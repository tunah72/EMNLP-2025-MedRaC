from typing import List, Dict
from method.method import Method
from evaluator import Evaluator
from model.model import LLM
from schema.schemas import prompt_style_to_schema

import os
import json
import logging
from pathlib import Path

import pandas as pd  # new dependency for convenient table handling

logger = logging.getLogger(__name__)


class Plain(Method):
    """Plain prompting strategy wrapper.

    1. ``generate_raw`` - run the model, collect raw answers/token counts
       and write them to ``<raw_json_dir>/<model>_<prompt_style>_raw.json``.

    2. ``evaluate`` - read the raw file, run every Evaluator object and
       append their results to a new file
       ``<eval_json_dir>/<model>_<prompt_style>_eval.json``.

    This separation makes it possible to cache expensive model calls and
    to re-run evaluation logic without touching the LLM again.
    """

    def __init__(
        self,
        prompt_style: str,
        llms: List[LLM],
        evaluators: List[Evaluator],
        **kwargs,
    ) -> None:
        valid_prompt_styles = {
            "direct": self.direct,
            "cot": self.cot,
            "oneshot": self.one_shot,
            "modular": self.modular,
            "modular_cot": self.modular_cot,
        }
        if prompt_style not in valid_prompt_styles:
            raise ValueError(f"Prompt style: {prompt_style} not supported.")

        self.prompt_style = prompt_style
        self.prompt_fn = valid_prompt_styles[prompt_style]
        self.evaluators = evaluators

        # Store latest run artefacts (optional, mainly for  backward compatibility)
        self.responses: Dict[str, List[str]] = {}
        self.answers: Dict[str, List[str]] = {}
        self.correctness: Dict[str, Dict[str, List[bool]]] = {}
        self.input_tokens: Dict[str, List[int]] = {}
        self.output_tokens: Dict[str, List[int]] = {}

        super().__init__(llms=llms, **kwargs)

    # ---------------------------------------------------------------------------------- #
    # Stage 1 – generation
    # ---------------------------------------------------------------------------------- #
    def generate_raw(
        self,
        test: bool = False,
        raw_json_dir: str = "raw_output",
    ) -> str:
        """Run the LLM(s) once, persist answers + token stats.

        Returns
        -------
        str
            Path of the raw JSON file (single-model use-case).  If multiple
            models are attached, the function still runs them all but returns
            the **first** path for convenience; a map is printed to the log.
        """

        # 1) load dataset ----------------------------------------------------------------
        if test:
            self.df = self.load_data_test()
        else:
            self.df = self.load_dataset()

        # 2) build prompts ---------------------------------------------------------------
        notes = self.df["Patient Note"].tolist()
        questions = self.df["Question"].tolist()
        calids = self.df["Calculator ID"].astype(str).tolist()

        prompts = self.prompt_fn(calids, notes, questions)

        os.makedirs(raw_json_dir, exist_ok=True)
        prompt_list = [
            {"system_msg": system, "user_msg": user}
            for system, user in prompts
        ]
        prompt_path = os.path.join(raw_json_dir, "prompts.json")
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(prompt_list, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(prompt_list)} prompts to {prompt_path}")
        # 3) ensure output dir -----------------------------------------------------------
        Path(raw_json_dir).mkdir(parents=True, exist_ok=True)

        # 4) query every model -----------------------------------------------------------
        model_to_path = {}
        for llm in self.llm_list:
            model_name = llm.model_name_full
            safe_model_name = model_name.replace("/", "_")  # file‑system safe

            schema = prompt_style_to_schema(self.prompt_style)
            generations = llm.generate(prompts, schema=schema)
            (
                self.responses[model_name],
                self.input_tokens[model_name],
                self.output_tokens[model_name],
            ) = map(list, zip(*generations))

            raw_records = self._build_records(
                model_name=model_name,
                include_evaluation=False,  # only raw fields
            )

            raw_path = (
                Path(raw_json_dir)
                / f"{safe_model_name}_{self.prompt_style}_raw.json"
            )
            self._dump_json(raw_records, raw_path)
            model_to_path[model_name] = str(raw_path)

            logger.info("Raw generation complete - %s saved (%d rows)",
                        raw_path, len(raw_records))

        # -------------------------------------------------------------------------------
        # For the common 1‑model scenario return its path so that calling‑site
        # code can remain simple.
        # -------------------------------------------------------------------------------
        if len(model_to_path) == 1:
            return next(iter(model_to_path.values()))
        return json.dumps(model_to_path, indent=2)

    # ---------------------------------------------------------------------------------- #
    # Stage 2 – evaluation
    # ---------------------------------------------------------------------------------- #
    def evaluate(
        self,
        raw_json_file: str,
        eval_json_dir: str = "eval_output/plain",
    ) -> str:
        """Read *raw* file, run evaluators, persist extended results."""
        str_raw_json_file = raw_json_file
        raw_json_file = Path(raw_json_file)
        if not raw_json_file.exists():
            raise FileNotFoundError(raw_json_file)

        # 1) read raw data
        with raw_json_file.open("r", encoding="utf‑8") as f:
            records = json.load(f)
        self.df = pd.DataFrame(records)

        # 2) infer model + bookkeeping 
        model_name = self.df["Model Name"].iloc[0]
        safe_model_name = model_name.replace("/", "_")

        self.responses[model_name] = self.df["LLM Original Answer"].tolist()
        self.input_tokens[model_name] = self.df["Input Tokens"].tolist()
        self.output_tokens[model_name] = self.df["Output Tokens"].tolist()
        self.correctness[model_name] = {}

        # 3) common fields for evaluators 
        ground_truths = self.df["Ground Truth Answer"].tolist()
        calids = self.df["Calculator ID"].astype(str).tolist()
        upper_limits = self.df["Upper Limit"].tolist()
        lower_limits = self.df["Lower Limit"].tolist()

        # 4) run every evaluator
        for evaluator in self.evaluators:
            # if (
            #     "LLM_Evaluator" in evaluator.get_evaluator_name()
            #     and self.prompt_style == "direct"
            # ):
            #     logger.debug("Skipping %s for direct style",
            #                  evaluator.get_evaluator_name())
            #     continue

            key, result_list = evaluator.check_correctness(
                responses=self.responses[model_name],
                ground_truths=ground_truths,
                calids=calids,
                upper_limits=upper_limits,
                lower_limits=lower_limits,
                relevant_entities=self.df["Relevant Entities"].tolist(),
                ground_truth_explanations=self.df["Ground Truth Explanation"].tolist(),
            )
            self.correctness[model_name][key] = result_list
            self.df[key] = result_list  # append column
            logger.info("Evaluator %s completed", evaluator.get_evaluator_name())

        # 5) token stats for LLM‑based evaluators 
        for evaluator in self.evaluators:
            if "LLM_Evaluator" in evaluator.get_evaluator_name():
                n_rows = len(self.df)
                logger.info(
                    "### LLM Evaluation Average Tokens - %s - input: %.2f | output: %.2f",
                    self.prompt_style,
                    evaluator.input_token_used / n_rows if n_rows else 0,
                    evaluator.output_token_used / n_rows if n_rows else 0,
                )

        # 6) persist full evaluation 
        Path(eval_json_dir).mkdir(parents=True, exist_ok=True)
        suffix = "_raw.json"
        if not str_raw_json_file.endswith(suffix):
            raise ValueError("input file must end with '_raw.json'")
        eval_path = str_raw_json_file[:-len(suffix)] + "_eval.json"

        # eval_path = (
        #     Path(eval_json_dir) /
        #     f"{safe_model_name}_{self.prompt_style}_eval.json"
        # )
        self._dump_json(self.df.to_dict(orient="records"), eval_path)
        logger.info("Evaluation written to %s", eval_path)

        return str(eval_path)

