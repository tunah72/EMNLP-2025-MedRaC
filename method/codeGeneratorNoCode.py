from model import LLM
from typing import Union, Tuple, List
import json
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from method import Method
import os
from evaluator import Evaluator
from schema.schemas import FormulaAndValues, Values, CalcAnswer
import logging
import pandas as pd
import re
from method import RAG
logger = logging.getLogger(__name__)

class CodeGeneratorNoCode(Method):
    """
    A class to generate and without code based on formulas and extracted values using an LLM model.
    """

    def __init__(
        self,
        llms: List[LLM],
        evaluators: List[Evaluator],
        model: LLM,
        use_rag: bool = False,
        doc_path="data/one_shot_finalized_explanation_formulas.txt",
        knowledge_base = "data/one_shot_finalized_explanation_formulas_embeddings"
    ):
        super().__init__(llms=llms)
        self.evaluators = evaluators
        self.model = model
        self.prompt_style = ("modular_cot_code_rag" if use_rag else "modular_cot_code") + self.model.model_name

        self.input_tokens_used = 0
        self.output_tokens_used = 0

        self.input_tokens = {}
        self.output_tokens = {}
        self.responses = {}
        self.correctness = {}
        self.formulas    = {}
        self.rag = RAG(
            doc_path=doc_path,
            embeddings_dir=knowledge_base,
        ) if use_rag else None
    
    def generate_raw(
        self,
        k = 1,
        test: bool = False,
        raw_json_dir: str = "raw_output/code",
        formula_json_path: str = "data/formula_new.json",
    ) -> str:
        """Run the LLM(s) once, persist answers + token stats for each model."""

        # 1) load dataset
        self.df = self.load_data_test() if test else self.load_dataset()

        notes     = self.df["Patient Note"].tolist()
        questions = self.df["Question"].tolist()
        calids    = self.df["Calculator ID"].astype(str).tolist()

        # 3) ensure output dir
        os.makedirs(raw_json_dir, exist_ok=True)

        model_to_path = {}
        for llm in self.llm_list:
            model_name      = llm.model_name_full
            safe_model_name = model_name.replace("/", "_")

            if self.rag:
                formulas = [
                    "\n".join(
                        self._safe_serialize(text)            # ← JSON-parse first
                        for text, _ in self.rag.retrieve(q, k=k)
                    )
                    for q in questions
                ]
                self.formulas[model_name] = formulas
                prompts = self._gen_extracted_values(notes, formulas, questions)
                schema = Values
                generations = llm.generate(prompts, schema=schema)
                extracted_values, in_toks, out_toks = map(list, zip(*generations))
                self.input_tokens[model_name]  = in_toks
                self.output_tokens[model_name] = out_toks
                calculations, answers = self.generate_no_code(formulas, extracted_values, questions)

            else:
                prompts = self._gen_formula_and_extracted_values(notes, questions)
                schema = FormulaAndValues
                generations = llm.generate(prompts, schema=schema)
                responses_list, in_toks, out_toks = map(list, zip(*generations))
                self.input_tokens[model_name]  = in_toks
                self.output_tokens[model_name] = out_toks

                self.formulas[model_name] = self._get_formulas(
                    calids=calids,
                    json_path=formula_json_path,
                )

                formulas, extracted_values = [], []
                for r in responses_list:
                    try:
                        d = r if isinstance(r, dict) else json.loads(r)

                        # ---------- split the two groups ----------
                        f_part = {}
                        if "formula" in d:
                            f_part["formula"] = d.pop("formula")
                        if "formula_reason" in d:
                            f_part["formula_reason"] = d.pop("formula_reason")

                        ev_part = {}
                        if "extracted_values" in d:
                            ev_part["extracted_values"] = d.pop("extracted_values")
                        if "extracted_values_reason" in d:
                            ev_part["extracted_values_reason"] = d.pop("extracted_values_reason")

                        # append; if either dict is empty, fall back to the raw response
                        formulas.append(f_part if f_part else r)
                        extracted_values.append(ev_part if ev_part else r)

                    except Exception:
                        # on any parsing failure, keep the raw response in both lists
                        formulas.append(r)
                        extracted_values.append(r)

                calculations, answers = self.generate_no_code(formulas, extracted_values, questions)

            self.responses[model_name] = [
                {"formula": f, "extracted_values": e, "calculation": c, "answer": a}
                for f, e, c, a in zip(formulas, extracted_values, calculations, answers)
            ]
            # print(self.formulas[model_name])
            # 7) dump raw records
            raw_records = self._build_records(model_name=model_name, include_evaluation=False)
            raw_path = os.path.join(
                raw_json_dir,
                f"{safe_model_name}_{self.prompt_style}_raw.json"
            )
            self._dump_json(raw_records, raw_path)
            model_to_path[model_name] = raw_path

            logger.info("Raw generation complete – %s saved (%d rows)",
                        raw_path, len(raw_records))

        # return the first (or only) path, or a JSON map if many
        if len(model_to_path) == 1:
            return next(iter(model_to_path.values()))
        return json.dumps(model_to_path, indent=2)
    
    def _safe_serialize(self, txt: str) -> str:
        """
        Try to json-loads `txt`; if it’s valid JSON, dump it back to a
        canonical string, otherwise just return the original text.
        """
        if not isinstance(txt, str):
            return str(txt)
        try:
            obj = json.loads(txt)
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            return txt

    
    def evaluate(
        self,
        raw_json_file: str,
        eval_json_dir: str = "eval_output/code",
    ) -> str:
        """Read *raw* file, run evaluators, persist extended results."""

        # ensure raw file exists
        if not os.path.exists(raw_json_file):
            raise FileNotFoundError(raw_json_file)

        # 1) read raw data
        with open(raw_json_file, "r", encoding="utf-8") as f:
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
            if (
                "LLM_Evaluator" in evaluator.get_evaluator_name()
                and self.prompt_style == "direct"
            ):
                logger.debug("Skipping %s for direct style",
                            evaluator.get_evaluator_name())
                continue

            key, result_list = evaluator.check_correctness(
                responses=self.responses[model_name],
                ground_truths=ground_truths,
                calids=calids,
                upper_limits=upper_limits,
                lower_limits=lower_limits,
                relevant_entities=self.df["Relevant Entities"].tolist(),
                ground_truth_explanations=self.df["Ground Truth Explanation"].tolist(),
                formulas = self.formulas[model_name],
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
        os.makedirs(eval_json_dir, exist_ok=True)
        eval_path = os.path.join(
            eval_json_dir,
            f"{safe_model_name}_{self.prompt_style}_eval.json"
        )
        self._dump_json(self.df.to_dict(orient="records"), eval_path)
        logger.info("Evaluation written to %s", eval_path)

        return eval_path


    def generate_no_code(
        self,
        formulas: List[Union[str, dict]],
        extracted_values: List[Union[str, dict]],
        questions: List[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Generates chain-of-thought calculation responses (no code),
        then parses them into 'calculations' and 'answers' lists.
        """
        # [same parsing of formulas/values and prompt gen as before...]
        parsed_formulas = []
        parsed_values = []
        for f, v in zip(formulas, extracted_values):
            if isinstance(f, str):
                try:    f = json.loads(f)
                except json.JSONDecodeError: pass
            if isinstance(v, str):
                try:    v = json.loads(v)
                except json.JSONDecodeError: pass
            parsed_formulas.append(f)
            parsed_values.append(v)

        prompts = self._gen_no_code_prompt(parsed_formulas, parsed_values, questions)
        gens = self.model.generate(prompts, CalcAnswer)  # List[Tuple[str, int, int]]

        # collect raw responses and update token usage
        responses = [r for r, _, _ in gens]
        self.input_tokens_used  += sum(it for _, it, _ in gens)
        self.output_tokens_used += sum(ot for _, _, ot in gens)

        calculations = []
        answers      = []

        for resp in responses:
            # By default, use the raw response
            calc = ans = resp

            if isinstance(resp, dict):
                # Already a parsed dict
                calc = resp.get("calculation", resp)
                ans  = resp.get("answer", resp)
            else:
                # Try parsing the string as JSON
                try:
                    parsed = json.loads(resp)
                    if isinstance(parsed, dict):
                        calc = parsed.get("calculation", resp)
                        ans  = parsed.get("answer", resp)
                except (json.JSONDecodeError, TypeError):
                    # Not JSON or unexpected type: keep raw text
                    pass

            calculations.append(calc)
            answers.append(ans)

        return calculations, answers

    def _gen_no_code_prompt(
        self,
        formulas: List[Union[str, dict]],
        extracted_values: List[Union[str, dict]],
        questions: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Creates system and user messages for the LLM to generate code.
        """
        prompts: List[Tuple[str, str]] = []
        system_msg = (
            "You are a calculation assistant. "
            "Please use chain of thought reasoning to compute the result of a given formula using provided values."
            "Return the response in the specified JSON format:\n"
                "{\"calculation\": str, \"answer\": str}\n\n"
                "- `calculation`: The steps you take to calculate the needed value.\n"
                "- `answer`: The final result in its simplest form."
            "Please return only the JSON dict and nothing else."
        )

        for formula, values, question in zip(formulas, extracted_values, questions):
            if formula == values:
                user_msg = (
                    f"Formula and Values: {json.dumps(formula)}\n"
                    f"Additionally, here's the question that might contain relevant information: {question}\n"
                )
            else:
                user_msg = (
                    f"Formula: {json.dumps(formula)}\n"
                    f"Values: {json.dumps(values)}\n"
                    f"Additionally, here's the question that might contain relevant information: {question}\n"
                )
            prompts.append((system_msg, user_msg))

        return prompts

    def _gen_formula_and_extracted_values(self, notes, questions):

        prompts = []
        
        for note, question in zip(notes, questions):
            system_msg = (
                "You are a reasoning assistant that follows a chain-of-thought approach to find important information in a given patient note. "
                "Follow these steps to extract necessary information:\n"
                "1. Reason about which formula(s) are applicable. Then identify the correct formula required for the calculation and state it explicitly.\n"
                "2. First reason about what values are needed then explain where these values appear in the text. Then, explicitly extract the values and map them to the formula variables.\n"
                "{\"formula_reason\": str, \"formula\": str, \"extracted_values_reason\": str, \"extracted_values\": dict}\n\n"
                "- `formula_reason`: The reasons that a formula is applicable.\n"
                "- `formula`: The explicit mathematical equation used for the calculation (e.g., `BMI = weight / height^2`).\n"
                "- `extracted_values_reason`: Justification for how each value was identified.\n"
                "- `extracted_values`: A dictionary mapping variable names to extracted values from the note (e.g., {\"weight\": \"70kg\", \"height\": \"1.75m\"}).\n"
            )
            
            user_msg = (
                f"Here is the patient note:\n"
                f"{note}\n\n"
                f"Here is the task:\n"
                f"{question}\n\n"
                "Please reason through each step carefully, providing justifications before stating the formula and extracted values. Return the response in the specified JSON format."
            )
            
            prompts.append((system_msg, user_msg))
        
        return prompts

    def _gen_extracted_values(self, notes, formulas, questions):

        prompts = []
        
        for note, formula, question in zip(notes, formulas, questions):
            system_msg = (
                "You are a reasoning assistant that follows a chain-of-thought approach to find important information in a given patient note. "
                "Follow these steps to extract necessary information:\n"
                "First reason about what values are needed then explain where these values appear in the text. Then, explicitly extract the values and map them to the formula variables.\n"
                "{\"extracted_values_reason\": str, \"extracted_values\": dict}\n\n"
                "- `extracted_values_reason`: Justification for how each value was identified.\n"
                "- `extracted_values`: A dictionary mapping variable names to extracted values from the note (e.g., {\"weight\": \"70kg\", \"height\": \"1.75m\"}).\n"
            )
            
            user_msg = (
                f"Here is the patient note:\n"
                f"{note}\n\n"
                f"Here is the question:\n"
                f"{question}\n\n"
                f"Here is the formula:\n"
                f"{formula}\n\n"
                "Please reason through each step carefully, providing justifications before stating extracted values. Return the response in the specified JSON format."
            )
            
            prompts.append((system_msg, user_msg))
        
        return prompts
    


    @staticmethod
    def _get_formulas(calids: List[str], json_path: str) -> List[str]:
        """
        Given a list of Calculator IDs, return a list of the corresponding formulas
        by looking them up in a JSON file. If any ID is not found, raise a ValueError.

        Parameters:
        calids (List[str]): A list of calculator ID strings to look up.
        json_path (str): Path to the JSON file containing formulas. Defaults to 'data/formula.json'.

        Returns:
        List[str]: A list of formulas matching each ID in `calids`.

        Raises:
        ValueError: If any ID in `calids` is not present in the JSON data.
        """
        # Load the JSON data from file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Build a mapping from Calculator ID to Formula
        id_to_formula = {
            item["Calculator ID"]: item["Formula"]
            for item in data
        }

        # Lookup each ID, raising an error if it's missing
        formulas = []
        for cid in calids:
            if cid not in id_to_formula:
                raise ValueError(f"Calculator ID '{cid}' not found in {json_path}")
            formulas.append(id_to_formula[cid])

        return formulas