from evaluator import RegEvaluator
from model import LLM, APIModel
from typing import List, Optional, Tuple
import json
import csv
import logging
from schema.schemas import *
from .baseEvaluator import Evaluator

logger = logging.getLogger(__name__)


class LLM_Evaluator(Evaluator):
    def __init__(self, model: LLM):
        """
        Initialize the evaluator with a given LLM instance.
        """
        self.model = model if model else APIModel("DeepSeek/deepseek-chat", temperature=0.1)
        self.input_token_used = 0
        self.output_token_used = 0

    
    def check_correctness(
        self,
        responses,
        ground_truths,
        ground_truth_explanations,
        relevant_entities,
        calids,                 
        upper_limits,          
        lower_limits,          
        formulas: Optional[List[str]] = None,
    ):
        """
        Evaluate each structured step.                                 
        For the *answer* step we now:
            1) parse the pure answer with an LLM,
            2) run RegEvaluator.check_correctness_parsed,
            3) embed the result back into the evaluation payload.
        """
        # ---------- unchanged: normalize responses into dict -------------
        num_samples = len(responses)
        parsed_responses = []
        for resp in responses:
            if isinstance(resp, str):
                try:
                    parsed = json.loads(resp)
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = resp
            parsed_responses.append(
                {
                    "formula": parsed.get("formula", resp),
                    "extracted_values": parsed.get("extracted_values", resp),
                    "calculation": parsed.get("calculation", resp),
                    "answer": parsed.get("final_answer", parsed.get("answer", resp)),
                }
            )

        # ----------- evaluate formula / values / calculation ------------
        all_evaluations = {f: [] for f in ["formula", "extracted_values", "calculation", "answer"]}

        for field, reference_list in [
            ("formula", formulas if formulas is not None else ground_truth_explanations),
            ("extracted_values", relevant_entities),
            ("calculation", ground_truth_explanations),
        ]:
            prompts = []
            for i in range(num_samples):
                if field == "formula":
                    f = parsed_responses[i]["formula"]
                    c = parsed_responses[i]["calculation"]
                    f_str = json.dumps(f) if isinstance(f, (dict, list)) else str(f)
                    c_str = json.dumps(c) if isinstance(c, (dict, list)) else str(c)
                    answer_val = f_str if f_str == c_str else f_str + " " + c_str
                else:
                    answer_val = parsed_responses[i][field]
                        
                sys_msg, usr_msg = self._gen_eval_prompt(
                    answer=answer_val,
                    reference=reference_list[i],
                    name_of_step=field,
                )
                prompts.append((sys_msg, usr_msg))

            outputs = self.model.generate(prompts=prompts, schema=EvaluationAspect)
            for response, in_tok, out_tok in outputs:
                if isinstance(response, (str, bytes, bytearray)):
                    try:
                        parsed = json.loads(response)
                    except json.JSONDecodeError:
                        parsed = response
                else:
                    parsed = response
                all_evaluations[field].append(parsed)
                self.input_token_used += in_tok
                self.output_token_used += out_tok

        # NEW answer-step pipeline 
        raw_answers = [item["answer"] for item in parsed_responses]
        ans_prompts = self._parse_ans_prompt(raw_answers)               
        parsed_out = self.model.generate(prompts=ans_prompts)            
        parsed_ans_list, in_tok_seq, out_tok_seq = zip(*parsed_out)      
        self.input_token_used += sum(in_tok_seq)                         
        self.output_token_used += sum(out_tok_seq)                       


        _, correctness_flags = RegEvaluator.check_correctness_parsed(    
            responses     = list(parsed_ans_list),
            ground_truths = ground_truths,
            calids        = calids,
            upper_limits  = upper_limits,
            lower_limits  = lower_limits,
        )

        for parsed_ans, gt, flag in zip(parsed_ans_list, ground_truths, correctness_flags):
            all_evaluations["answer"].append(
                {
                    "result": flag,
                    "explanation": (
                        f"Parsed Answer: {parsed_ans}, checked with ground truth "
                        f"answer {gt} by rule-based evaluator."
                    ),
                }
            )

        # ---------------- aggregate per-sample evaluations --------------
        evaluations = []
        for i in range(num_samples):
            evaluations.append(
                {
                    "formula": all_evaluations["formula"][i],
                    "extracted_values": all_evaluations["extracted_values"][i],
                    "calculation": all_evaluations["calculation"][i],
                    "answer": all_evaluations["answer"][i],
                }
            )

        return "LLM Evaluation", evaluations



    def _gen_eval_prompt(self, answer, reference, name_of_step):
        # System message is the same for all steps
        system_msg = (
            "You are a medical calculation assistant. Evaluate whether each step is correct by comparing it to the gold-standard reference."
        )

        # For calculation steps, omit the gold-standard reference entirely
        if name_of_step == "calculation":
            user_msg = (
                f"{name_of_step.capitalize()} to be evaluated:\n{answer}\n\n"
                "Note: Judge ONLY the mathematical correctness of each arithmetic "
                "step (addition, subtraction, multiplication, division, powers, "
                "roots, etc.). Do NOT assess whether the formula used is appropriate "
                "or whether the input values were correct or reasonable. Treat small rounding or "
                "decimal-precision differences as acceptable"
                'Respond in this JSON format:\n\n'
                '{"result": "Correct" or "Incorrect", "explanation": "Brief justification."}'
            )
            return system_msg, user_msg

        # For all other steps, include the gold-standard reference first
        user_msg = (
            f"{name_of_step.capitalize()} to be evaluated:\n{answer}\n\n"
            f"Gold-standard reference (fully correct):\n{reference}\n\n"
            "Determine if the given part is correct according to the Gold-standard reference. "
            'Respond in this JSON format:\n\n'
            '{"result": "Correct" or "Incorrect", "explanation": "Brief justification."}'
        )

        if name_of_step == "formula":
            user_msg += (
                "\n\n"
                "Note: Judge ONLY whether the mathematical formula or scoring standard invoked is appropriate. Do NOT evaluate:"
                "• the specific values plugged into the formula,"
                "• the correctness of any later calculations."
                "If the gold-standard reference lists multiple valid variants (e.g., male vs. female, different ethnicities), the answer is considered correct as long as it correctly applies ANY one of those variants. If the provided formula includes more detail than the gold-standard reference but the overlapping portion is consistent and correct, it should still be considered correct."
            )
        elif name_of_step == "extracted_values":
            user_msg += (
                "\n\n"
                "Note: Check if all variables given in the gold-standard reference are found or implied. "
                "Ignore any naming discrepancies, as long as the meaning is the same. "
                "It is ok if the answer has more variables than the gold-standard reference."
                "If the given answer has a different unit than the gold-standard answer, please do conversion first. "
                "Answers with reasonable rounding errors MUST be considered Correct."
            )
        elif name_of_step in ("answer", "final_answer"):
            user_msg += (
                "\n\n"
                "Note: You ONLY need to check whether the final numerical answer matches the provided gold-standard reference. "
                "The correctness of the intermediate steps does NOT matter. If one has a unit and the other does not, please ignore the unit. "
                "If the given answer has a different unit than the gold-standard answer, please do conversion first. "
                "Answers with rounding to the nearest integer and reasonable computational deviations MUST be considered Correct."
            )

        return system_msg, user_msg


    
    @staticmethod
    def evaluate_formula_prompts(
        solutions: List[str],
        ground_truths: List[str],
        upperlimits: List[str],
        lowerlimits: List[str],
        relevant_entities_list: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Generate prompts for evaluating the correctness of the formula or scoring criteria.
        """
        prompts = []
        for solution, ground_truth, upperlimit, lowerlimit, _ in zip(
            solutions, ground_truths, upperlimits, lowerlimits, relevant_entities_list
        ):
            # helper = f"Note: The final answer WITHIN the range ({lowerlimit}, {upperlimit}) is also considered correct."
            system_msg = (
                "You are a medical calculation assistant. Your task is to evaluate the medical calculation solution according to the given ground truth answer."
            )
            user_msg = (
                f"This is the ground truth answer:\n{ground_truth}\n"
                f"This is the solution to be evaluated:\n{solution}\n\n"
                "Your task is to check the formula or scoring criteria:\n"
                "Determine if the solution uses the correct medical formula(s) or scoring standard(s) as described in the ground truth answer.\n"
                "Only consider the mathematical formula functions and any related helper functions or scoring standard(s).\n\n"
                "Please provide your evaluation in a structured JSON format with two keys: 'result' (either 'Correct' or 'Incorrect') and a brief 'explanation'."
            )
            prompts.append((system_msg, user_msg))
        return prompts

    @staticmethod
    def evaluate_variables_prompts(
        solutions: List[str],
        ground_truths: List[str],
        upperlimits: List[str],
        lowerlimits: List[str],
        relevant_entities_list: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Generate prompts for evaluating the correctness of variable substitution.
        """
        prompts = []
        for solution, ground_truth, upperlimit, lowerlimit, relevant_entity in zip(
            solutions, ground_truths, upperlimits, lowerlimits, relevant_entities_list
        ):
            # helper = f"Note: The final answer WITHIN the range ({lowerlimit}, {upperlimit}) is also considered correct."
            system_msg = (
                "You are a medical calculation assistant. Your task is to evaluate the medical calculation solution according to the given information."
            )
            user_msg = (
                # f"This is the ground truth answer:\n{ground_truth}\n"
                f"This is the solution to be evaluated:\n{solution}\n\n"
                f"Relevant Entities:\n{relevant_entity}\n\n"
                "Your task is to compare the variable values used in the solution with the values specified in the relevant entities.\n"
                "Consider ONLY the variables provided in the relevant entities, not any derived or later calculated variables in the solution.\n\n"
                "Please provide your evaluation in a structured JSON format with two keys: 'result' (either 'Correct' or 'Incorrect') and a brief 'explanation'."
            )
            prompts.append((system_msg, user_msg))
        return prompts

    @staticmethod
    def evaluate_calculation_prompts(
        solutions: List[str],
        ground_truths: List[str],
        upperlimits: List[str],
        lowerlimits: List[str],
        relevant_entities_list: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Generate prompts for evaluating the correctness of the calculation process.
        """
        prompts = []
        for solution, ground_truth, upperlimit, lowerlimit, _ in zip(
            solutions, ground_truths, upperlimits, lowerlimits, relevant_entities_list
        ):
            # helper = f"Note: The final answer WITHIN the range ({lowerlimit}, {upperlimit}) is also considered correct."
            system_msg = (
                "You are a medical calculation assistant. Your task is to evaluate the medical calculation solution."
            )
            user_msg = (
                f"This is the solution to be evaluated:\n{solution}\n\n"
                "Your task is to evaluate only the arithmetic calculation process in the solution. "
                "Ignore any issues related to the selection of formulas, substituted values, or clinical appropriateness. "
                "Focus ONLY on whether each arithmetic step (e.g., addition, subtraction, multiplication, division, square root, etc.) is mathematically valid. "
                "If the calculated result has minor differences due to rounding or decimal approximations, consider them acceptable. "
                "Determine if each step is mathematically correct.\n\n"
                "Please provide your evaluation in a structured JSON format with two keys: 'result' (either 'Correct' or 'Incorrect') and a brief 'explanation' summarizing your reasoning."
            )

            prompts.append((system_msg, user_msg))
        return prompts

    @staticmethod
    def evaluate_final_answer_prompts(
        solutions: List[str],
        ground_truth_ans: List[str],
        upperlimits: List[str],
        lowerlimits: List[str],
        relevant_entities_list: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Generate prompts for evaluating the correctness of the final answer.
        """
        prompts = []
        for solution, ground_truth_an, upperlimit, lowerlimit, _ in zip(
            solutions, ground_truth_ans, upperlimits, lowerlimits, relevant_entities_list
        ):
            helper = f"Note: The final answer WITHIN the range ({lowerlimit}, {upperlimit}) is also considered correct. You MUST be careful with numbers when comparing the final answer with the ground truth answer."
            system_msg = (
                "You are a medical calculation assistant. Your task is to evaluate the final answer in the medical solution."
            )
            user_msg = (
                f"This is the ground truth answer:\n{ground_truth_an + helper}\n"
                f"This is the solution to be evaluated:\n{solution}\n\n"
                f"Remember, the final answer should fall within the range ({lowerlimit}, {upperlimit}).\n"
                "Your task is to compare the final numerical result in the solution with the ground truth answer. Only judge whether the final numerical result is correct.\n\n"
                "Please provide your evaluation in a structured JSON format with two keys: 'result' (either 'Correct' or 'Incorrect') and a brief 'explanation'."
            )
            prompts.append((system_msg, user_msg))
        return prompts
    
    @staticmethod
    def _parse_ans_prompt(ans: List[str]) -> List[Tuple[str, str]]:
        """
        Create zero-shot chain-of-thought (CoT) prompts for multiple medical calculation tasks.

        Args:
            calids: List of IDs of the specific calculators.
            notes: List of patient note texts.
            questions: List of calculation questions.

        Returns:
            List of tuples, where each tuple is (system_message, user_message).
        """
        prompts = []

        for an in ans:
            system_msg = (
                "You are a helpful assistant for extracting numeric or date information from medical calculator outputs."
            )
            user_msg = (
            f"""
            The given text is the output of a medical calculator. Please ONLY extract numeric or date information, as follows:
            - If the answer involves numeric calculation, identify the number representing the final result, and return only that number (without units or extra content).
            - If the answer involves a date, identify the final date and return it only, in the format MM/DD/YYYY (e.g., 08/31/2023 or 07/03/2000), with no other text.
            - If the answer describes age or duration (in the form of weeks and days), return a tuple clearly stating the weeks and days, e.g., (4 weeks, 3 days), (0 weeks, 5 days).
            \n Given text: {an}
            """
            )
            prompts.append((system_msg, user_msg))

        return prompts
    


    def get_evaluator_name(self) -> str:
        """
        Returns the name of the evaluator.
        """
        return f'LLM_Evaluator using {self.model.model_name_full}'
    