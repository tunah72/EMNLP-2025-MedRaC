from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from schema.schemas import SelfRefineFB, CoTOutput
from method.method import Method            # base class with I/O helpers
from evaluator import Evaluator             # generic interface
from model.model import LLM                 # generic LLM interface

logger = logging.getLogger(__name__)
# ──────────────────────────────────────────────────────────────────────────────


class SelfRefine(Method):
    """
    Run one LLM in a *self-refinement* loop (feedback → refine) and offer two
    public entry points:

    • generate_raw() - produce model answers + token usage + history
    • evaluate()    - load a raw file and run all supplied evaluators

    All intermediate feedback / refined drafts are written into self.history
    per row using keys ``feedback<i>`` / ``refine<i>`` (i = 1, 2, …).
    """

    # ───────────────────────────── init ──────────────────────────────
    def __init__(
        self,
        refine_times: int,
        allow_early_stop: bool,
        prompt_style: str,
        llms: List[LLM],
        evaluators: List[Evaluator],
        **kwargs,
    ) -> None:
        # --- prompt dispatcher ------------------------------------------------
        valid_prompt_styles = {
            "direct":   self.direct,
            "cot":      self.cot,
            "oneshot":  self.one_shot,
            "modular":  self.modular,
        }
        if prompt_style not in valid_prompt_styles:
            raise ValueError(f"Prompt style '{prompt_style}' not supported.")
        self.prompt_style = prompt_style
        self.prompt_fn = valid_prompt_styles[prompt_style]

        # --- args -------------------------------------------------------------
        if not llms:
            raise ValueError("`llms` must contain at least one LLM.")
        if not evaluators:
            raise ValueError("`evaluators` list cannot be empty.")
        self.llm_list   = llms
        self.evaluators = evaluators

        self.refine_times     = refine_times
        self.allow_early_stop = allow_early_stop

        # --- record buffers used by _build_records ----------------------------
        self.responses:      Dict[str, List[str]]             = {}
        self.input_tokens:   Dict[str, List[int]]             = {}
        self.output_tokens:  Dict[str, List[int]]             = {}
        self.correctness:    Dict[str, Dict[str, List[bool]]] = {}
        self.history:        Dict[str, List[Dict[str, str]]]  = {}

        super().__init__(llms=llms, **kwargs)

    # ───────────────────── core refinement loop ─────────────────────
    def _run_refinement_loop(
        self,
        llm: LLM,
        rows_info: List[Tuple[int, Dict]],
        final_responses: List[str],
        tokens_usage: List[Dict[str, int]],
        history_all: List[List[Dict]],
    ) -> Tuple[List[str], List[Dict[str, int]], List[List[Dict]], List[int]]:
        """
        Identical logic to the original implementation but *mutates*:
        • final_responses - latest draft
        • tokens_usage    - cumulative token counts
        • history_all     - list-of-lists storing each turn
        Returns the same objects for convenience.
        """
        converged   = [False] * len(final_responses)
        iter_used   = [0]     * len(final_responses)

        for it in range(self.refine_times):
            active = [i for i, c in enumerate(converged) if not c]
            if not active:
                break

            # ---------- 1) feedback ------------------------------------------
            fb_prompts, fb_idx = [], []
            for i in active:
                row = rows_info[i][1]
                fb_prompts.append(
                    self._self_feedback_prompt(
                        note=row["Patient Note"],
                        question=row["Question"],
                        current_answer=final_responses[i],
                        prompt_style=self.prompt_style,
                        calculator_id=str(row["Calculator ID"])
                        if self.prompt_style == "oneshot"
                        else None,
                    )
                )
                fb_idx.append(i)

            for idx, (fb_resp, in_tok, out_tok) in zip(
                fb_idx, llm.generate(fb_prompts, batch_size=len(fb_prompts),schema=SelfRefineFB)
            ):
                fb_resp = fb_resp if isinstance(fb_resp, str) else json.dumps(fb_resp)
                if isinstance(fb_resp, str):
                    try:
                        fb_obj = json.loads(fb_resp)
                        response_obj = fb_obj
                    except (json.JSONDecodeError, TypeError):
                        response_obj = fb_resp
                else:
                    response_obj = fb_resp
                tokens_usage[idx]["Input Tokens"]  += in_tok
                tokens_usage[idx]["Output Tokens"] += out_tok
                history_all[idx].append(
                    {"iteration": it + 1, "type": "feedback", "response": response_obj}
                )

            # ---------- 2) decide refinement ---------------------------------
            to_refine: List[int] = []
            for idx in active:
                resp = history_all[idx][-1]["response"]
                has_errors: bool | None = None

                if isinstance(resp, dict):
                    raw = resp.get("has_errors", None)
                    if isinstance(raw, bool):
                        has_errors = raw
                    elif isinstance(raw, str):
                        has_errors = raw.lower() == "true"

                elif isinstance(resp, str):
                    try:
                        obj = json.loads(resp)
                        raw = obj.get("has_errors", None)
                        if isinstance(raw, bool):
                            has_errors = raw
                        elif isinstance(raw, str):
                            has_errors = raw.lower() == "true"
                    except (json.JSONDecodeError, TypeError):
                        pass

                    if has_errors is None:
                        m = re.search(r'"has_errors"\s*:\s*(true|false)', resp, re.I)
                        if m:
                            has_errors = (m.group(1).lower() == "true")

                if has_errors is False and self.allow_early_stop:
                    converged[idx] = True
                else:
                    to_refine.append(idx)

            # ---------- 3) refine -------------------------------------------
            if not to_refine:
                continue

            rf_prompts, rf_idx = [], []
            for idx in to_refine:
                row = rows_info[idx][1]
                rf_prompts.append(
                    self._self_refine_prompt(
                        note=row["Patient Note"],
                        question=row["Question"],
                        current_answer=final_responses[idx],
                        feedback=history_all[idx][-1]["response"],
                        prompt_style=self.prompt_style,
                        calculator_id=str(row["Calculator ID"])
                        if self.prompt_style == "oneshot"
                        else None,
                    )
                )
                rf_idx.append(idx)

            for idx, (rf_resp, in_tok, out_tok) in zip(
                rf_idx, llm.generate(rf_prompts, batch_size=len(rf_prompts),schema = CoTOutput,)
            ):
                rf_resp = rf_resp if isinstance(rf_resp, str) else json.dumps(rf_resp)
                # print(rf_resp)
                if isinstance(rf_resp, str):
                    try:
                        fb_obj = json.loads(rf_resp)
                        response_obj = fb_obj
                    except (json.JSONDecodeError, TypeError):
                        response_obj = rf_resp
                else:
                    response_obj = rf_resp

                final_responses[idx]                = response_obj
                tokens_usage[idx]["Input Tokens"]  += in_tok
                tokens_usage[idx]["Output Tokens"] += out_tok
                iter_used[idx]                     += 1
                history_all[idx].append(
                    {"iteration": it + 1, "type": "refine", "response": response_obj}
                )

        return final_responses, tokens_usage, history_all, iter_used

    # ─────────────────────────── generate_raw ───────────────────────────
    def generate_raw(
        self,
        test: bool = False,
        raw_json_dir: str = "raw_output/selfRefine",
    ) -> str:
        """
        Run the self-refinement pipeline once per LLM, **without** evaluation.
        Returns the path to the raw json file (string) or a JSON map if many
        models are run.
        """
        self.df = self.load_data_test() if test else self.load_dataset()

        notes      = self.df["Patient Note"].tolist()
        questions  = self.df["Question"].tolist()
        calids_str = self.df["Calculator ID"].astype(str).tolist()

        Path(raw_json_dir).mkdir(parents=True, exist_ok=True)
        model_to_path: Dict[str, str] = {}

        # one pass per model ---------------------------------------------------
        for llm in self.llm_list:
            mname = llm.get_model_name()
            logger.info("Self-Refine raw generation - model: %s", mname)

            # ---------- 0) build initial prompts -----------------------------
            raw_prompts = self.prompt_fn(calids_str, notes, questions)
            pt_pairs    = [p if isinstance(p, tuple) and len(p) == 2 else ("", p)
                           for p in raw_prompts]

            init_resps, tokens_usage = [], []
            for text, in_tok, out_tok in llm.generate(pt_pairs, batch_size=len(pt_pairs)):
                init_resps.append(text)
                tokens_usage.append({"Input Tokens": in_tok, "Output Tokens": out_tok})

            final_resps  = init_resps.copy()
            hist_per_row = [[{"iteration": 0, "type": "initial", "response": r}]
                            for r in init_resps]

            rows_info = list(enumerate(self.df.to_dict("records")))
            final_resps, tokens_usage, hist_per_row, _ = self._run_refinement_loop(
                llm, rows_info, final_resps, tokens_usage, hist_per_row
            )

            # ---------- 1) flatten history dict format -----------------------
            flat_history: List[Dict[str, str]] = []
            for row_hist in hist_per_row:
                row_dict: Dict[str, str] = {}
                for turn in row_hist:
                    if turn["type"] == "feedback":
                        key = f"feedback{turn['iteration']}"
                    elif turn["type"] == "refine":
                        key = f"refine{turn['iteration']}"
                    else:
                        # skip "initial"
                        continue
                    row_dict[key] = turn["response"]
                flat_history.append(row_dict)

            # ---------- 2) cache for _build_records --------------------------
            self.responses[mname]      = final_resps
            self.input_tokens[mname]   = [t["Input Tokens"]  for t in tokens_usage]
            self.output_tokens[mname]  = [t["Output Tokens"] for t in tokens_usage]
            self.correctness[mname]    = {}                  # filled in evaluate()
            self.history[mname]        = flat_history

            # ---------- 3) dump raw json -------------------------------------
            records   = self._build_records(mname, include_evaluation=False)
            raw_path  = (
                Path(raw_json_dir) /
                f"SelfRefine_{mname.replace('/', '_')}_{self.prompt_style}_raw.json"
            )
            self._dump_json(records, raw_path)
            logger.info("Raw file written → %s  (%d rows)", raw_path, len(records))

            model_to_path[mname] = str(raw_path)

        # single vs. multi model return ---------------------------------------
        if len(model_to_path) == 1:
            return next(iter(model_to_path.values()))
        return json.dumps(model_to_path, indent=2)

    # ───────────────────────────── evaluate ─────────────────────────────
    def evaluate(
        self,
        raw_json_file: str,
        eval_json_dir: str = "eval_output/selfRefine",
    ) -> str:
        """
        Load a raw json file, run **all** evaluators, attach correctness fields
        and dump *_eval.json*.
        """
        raw_path = Path(raw_json_file)
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        with raw_path.open("r", encoding="utf-8") as fp:
            records = json.load(fp)
        self.df = pd.DataFrame(records)

        mname      = self.df["Model Name"].iloc[0]
        safe_mname = mname.replace("/", "_")

        # --- rebuild buffers for _build_records -----------------------------
        self.responses[mname]      = self.df["LLM Original Answer"].tolist()
        self.input_tokens[mname]   = self.df["Input Tokens"].tolist()
        self.output_tokens[mname]  = self.df["Output Tokens"].tolist()
        self.history[mname]        = self.df["History"].tolist()
        self.correctness[mname]    = {}

        gts          = self.df["Ground Truth Answer"].tolist()
        calids_str   = self.df["Calculator ID"].astype(str).tolist()
        uppers       = self.df["Upper Limit"].tolist()
        lowers       = self.df["Lower Limit"].tolist()
        rel_entities = self.df.get("Relevant Entities", []).tolist()
        gt_expl      = self.df.get("Ground Truth Explanation", []).tolist()

        # --- iterate all evaluators ----------------------------------------
        for ev in self.evaluators:
            # optional: skip certain evals on incompatible prompt styles
            if (
                "LLM_Evaluator" in ev.get_evaluator_name()
                and self.prompt_style == "direct"
            ):
                logger.debug("Skip %s on direct style", ev.get_evaluator_name())
                continue

            key, result_list = ev.check_correctness(
                responses                 = self.responses[mname],
                ground_truths             = gts,
                calids                    = calids_str,
                upper_limits              = uppers,
                lower_limits              = lowers,
                relevant_entities         = rel_entities,
                ground_truth_explanations = gt_expl,
            )
            self.correctness[mname][key] = result_list
            self.df[key] = result_list
            logger.info("Evaluator %s finished.", ev.get_evaluator_name())

        # --- evaluator token stats (LLM-based only) -------------------------
        for ev in self.evaluators:
            if "LLM_Evaluator" in ev.get_evaluator_name():
                n = len(self.df)
                logger.info(
                    "### LLM Eval Avg Tokens - %s - in: %.2f | out: %.2f",
                    ev.get_evaluator_name(),
                    ev.input_token_used / n if n else 0,
                    ev.output_token_used / n if n else 0,
                )

        # --- dump eval file --------------------------------------------------
        Path(eval_json_dir).mkdir(parents=True, exist_ok=True)
        eval_path = (
            Path(eval_json_dir) /
            raw_path.name.replace("_raw.json", "_eval.json")
        )
        full_records = self._build_records(mname, include_evaluation=True)
        self._dump_json(full_records, eval_path)
        logger.info("Eval file written → %s", eval_path)

        return str(eval_path)


    # --------------- HELPER PROMPTS --------------------

    @staticmethod
    def _self_feedback_prompt(note: str, question: str, current_answer: str,
                             prompt_style: str, calculator_id: str = None) -> Tuple[str, str]:
        """
        Create a self-feedback prompt for medical calculation tasks.

        Args:
            note: The patient note text.
            question: The calculation question.
            current_answer: The model's current answer to evaluate.
            prompt_style: The style of prompt to use ("cot", "direct", or "oneshot").
            calculator_id: ID of the specific calculator (required for oneshot).

        Returns:
            Tuple containing (system_message, user_message).
        """
        if prompt_style == "cot" or prompt_style == "direct":
            system_msg = (
                "You are a medical calculation assistant. Your task is to evaluate whether the given medical calculation process is correct."
            )
            user_msg = (
                "Your task is to carefully review the provided answer by analyzing medical formula used and mathematical steps. "
                "Determine whether the entire calculation process is correct, and identify any mistakes."
                "Present your evaluation strictly in JSON with this schema:\n"
                "{\"reasoning\": \"<your detailed reasoning here>\", \"has_errors\": <True/False>}\n\n"
                "\"has_errors\" must be True if any mistake is found, otherwise False\n"

                f"Patient Note:\n{note}\n\n"
                f"Medical Calculation Task:\n{question}\n\n"
                f"Answer to Review:\n{current_answer}\n\n"
            )

        elif prompt_style == "oneshot":
            if calculator_id is None:
                raise ValueError("Calculator ID must be provided for oneshot prompts.")

            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "..", "data", "one_shot_finalized_explanation.json")
            try:
                with open(json_path, "r") as file:
                    one_shot_json = json.load(file)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError(f"Error loading one-shot examples: {e}")

            if calculator_id not in one_shot_json:
                raise ValueError(f"Calculator ID {calculator_id} not found in one-shot examples")

            example = one_shot_json[calculator_id]
            example_note = example["Patient Note"]
            example_output = {
                "step_by_step_thinking": example["Response"]["step_by_step_thinking"],
                "answer": example["Response"]["answer"]
            }

            system_msg = "You are a medical calculation assistant. Your task is to evaluate a medical calculation answer given an example"
            user_msg = (

                "Your task is to carefully review the provided answer by analyzing medical formula used and mathematical steps. "
                "Determine whether the entire calculation process is correct, and identify any mistakes."
                "Present your evaluation strictly in JSON with this schema:\n"
                "{\"step_by_step_thinking\": \"<your detailed reasoning here>\", \"has_errors\": <True/False>}\n\n"
                "\"has_errors\" must be True if any mistake is found, otherwise False\n"

                f"Patient Note:\n{note}\n\n"
                f"Medical Calculation Task:\n{question}\n\n"
                f"Your answer was:\n{current_answer}\n\n"

                "For guidance, consider the following example:\n"
                f"Example Patient Note:\n{example_note}\n\n"
                f"Example Output:\n{json.dumps(example_output)}"
            )
        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")

        return system_msg, user_msg

    @staticmethod
    def _self_refine_prompt(note: str, question: str, current_answer: str, feedback: str,
                           prompt_style: str, calculator_id: str = None) -> Tuple[str, str]:
        """
        Create a self-refinement prompt for medical calculation tasks.

        Args:
            note: The patient note text.
            question: The calculation question.
            current_answer: The model's current answer.
            feedback: The feedback on the current answer.
            prompt_style: The style of prompt to use ("cot", "direct", or "oneshot").
            calculator_id: ID of the specific calculator (required for oneshot).

        Returns:
            Tuple containing (system_message, user_message).
        """
        if prompt_style == "cot" or prompt_style == "direct":
            system_msg = (
                "You are an expert medical assistant who has received chain-of-thought feedback on your previous answer. "
                "Please review your previous reasoning and the feedback below. "
                "Think step-by-step: if the feedback indicates no errors, restate your original answer; "
                "if errors are found, generate a refined answer that corrects any issues in data extraction, reasoning, or calculation. "
                "Output your refined answer as a JSON dict with keys 'step_by_step_thinking' and 'answer'."
            )
            user_msg = (
                f"Patient Note:\n{note}\n\n"
                f"Medical Calculation Task:\n{question}\n\n"
                f"Your previous answer was:\n{current_answer}\n\n"
                f"Feedback:\n{feedback}\n\n"
                "Please provide your refined answer (with chain-of-thought) in the specified JSON format."
                '{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), '
                '"answer": str(short_and_direct_answer_of_the_question)}.'
            )
        elif prompt_style == "oneshot":
            if calculator_id is None:
                raise ValueError("Calculator ID must be provided for oneshot prompts.")

            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "..", "data", "one_shot_finalized_explanation.json")
            try:
                with open(json_path, "r") as file:
                    one_shot_json = json.load(file)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError(f"Error loading one-shot examples: {e}")

            if calculator_id not in one_shot_json:
                raise ValueError(f"Calculator ID {calculator_id} not found in one-shot examples")

            example = one_shot_json[calculator_id]
            example_note = example["Patient Note"]
            example_output = {
                "step_by_step_thinking": example["Response"]["step_by_step_thinking"],
                "answer": example["Response"]["answer"]
            }

            system_msg = (
                "You are a helpful assistant who has received chain-of-thought feedback on your previous answer. "
                "Please review your previous reasoning and the feedback below. "
                "Think step-by-step: if no errors are indicated, restate your original answer; otherwise, produce a refined answer that corrects any issues. "
                "Output your refined answer as a JSON dict with keys 'step_by_step_thinking' and 'answer'.\n\n"
                f"Example Patient Note:\n{example_note}\n\n"
                f"Example Output:\n{json.dumps(example_output)}"
            )
            user_msg = (
                f"Patient Note:\n{note}\n\n"
                f"Medical Calculation Task:\n{question}\n\n"
                f"Your previous answer was:\n{current_answer}\n\n"
                f"Feedback:\n{feedback}\n\n"
                "Please provide your refined answer (with chain-of-thought) in the specified JSON format."
            )
        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")

        return system_msg, user_msg
