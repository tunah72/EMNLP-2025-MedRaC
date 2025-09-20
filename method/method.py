from model.model import LLM
from typing import List, Tuple, Optional, Dict
import pandas as pd
import logging
import abc
import json
import os
import ast
import codecs


logger = logging.getLogger(__name__)

class Method(abc.ABC):
    def __init__(self, llms: list[LLM], dataset_path: Optional[str] = None, n_data: Optional[int] = None, row_numbers: Optional[List[int]] = None, calculator_id: Optional[int] = None, batch_size: Optional[int] = 1):
        """
        Initialize a Method instance.
        
        Args:
            llm: List of model names (including company name) to use with this method
        """
        self.llm_list = llms
        self.dataset_path = dataset_path
        self.n_data = n_data
        self.row_numbers = row_numbers
        self.calculator_id = calculator_id
        self.batch_size = batch_size

    @staticmethod
    def _parse_json_if_possible(x):
        if not isinstance(x, str):
            return x

        s = x.strip()

        # 1) JSON or Python literal?
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    pass

        # 2) Only decode escape sequences if there *are* backslashes in the string
        if "\\" in s:
            # This will turn "\\n" → real newline, "\\u2265" → "≥", etc.
            try:
                return codecs.decode(s, "unicode_escape")
            except Exception:
                return s

        # 3) Otherwise leave real Unicode alone
        return s

    
    # @staticmethod
    def load_dataset(self, dataset_path: str= './data/test_data.csv') -> pd.DataFrame:
        """
        Load a complete dataset from a CSV file.
        
        Args:
            dataset_path: Path to the CSV dataset
            
        Returns:
            DataFrame containing the dataset
            
        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            pd.errors.EmptyDataError: If the dataset is empty
        """
        dataset_path = self.dataset_path or dataset_path
        logger.info(f"Loading dataset from '{dataset_path}'...")
        try:
            df = pd.read_csv(dataset_path, encoding='utf-8')
            df = df.map(self._parse_json_if_possible)
            logger.info(f"Dataset loaded successfully, total {len(df)} rows found.")
            return df
        except FileNotFoundError:
            logger.error(f"Dataset file not found at '{dataset_path}'")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Dataset file at '{dataset_path}' is empty")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def load_data_test(self, dataset_path: str = './data/test_data.csv', n_data: int = 2, 
                   row_numbers: Optional[List[int]] = None, 
                   calculator_id: Optional[int] = None) -> pd.DataFrame:
        """
        Load test data with specific filtering options.
        
        Args:
            dataset_path: Path to the dataset
            n_data: Number of data samples to load if using calculator_id or for random sampling
            row_numbers: Specific row numbers to load
            calculator_id: Specific calculator ID to filter by
            
        Returns:
            DataFrame containing the filtered test data
        """
        dataset_path = self.dataset_path or dataset_path
        n_data = self.n_data or n_data
        row_numbers = self.row_numbers or row_numbers
        calculator_id = self.calculator_id or calculator_id

        # Assert that row_numbers and calculator_id cannot both be provided 
        if row_numbers is not None and calculator_id is not None:
            logger.error("row_numbers and calculator_id cannot both be provided.")
            raise ValueError("row_numbers and calculator_id cannot both be provided.")
        
        # Load the full dataset with the specified encoding
        df = pd.read_csv(dataset_path, encoding='utf-8')
        df = df.map(self._parse_json_if_possible)
        
        if row_numbers is not None:
            return df.iloc[row_numbers]
        elif calculator_id is not None:
            return df[df['Calculator ID'] == calculator_id].head(n_data)
        else:
            # If both are None, sample n_data rows
            return df.sample(n=n_data)


    @abc.abstractmethod
    def generate_raw(self, test: bool = False, raw_json_dir: Optional[str] = None) -> str:
        """
        Generate raw JSON data for the method.
        
        Args:
            test: Whether to run in test mode
            raw_json_dir: Directory to save the raw JSON data
            
        Returns:
            Path to the generated raw JSON data
        """
        pass
        
    
    @abc.abstractmethod
    def evaluate(raw_json: List[dict], eval_json_dir: Optional[str] = None) -> str:
        """
        Evaluate the generated raw JSON data.
        
        Args:
            raw_json: List of raw JSON data to evaluate
            eval_json_dir: Directory to save the evaluation results
            
        Returns:
            Path to the evaluation results
        """
        pass
    


    @staticmethod 
    def direct(calids: List[str], notes: List[str], questions: List[str]) -> List[Tuple[str, str]]:
        """
        Create direct answer prompts for multiple medical calculation tasks.
        
        Args:
            calids: List of IDs of the specific calculators.
            notes: List of patient note texts.
            questions: List of calculation questions.
            
        Returns:
            List of tuples, where each tuple is (system_message, user_message).
        """
        prompts = []
        
        for calid, note, question in zip(calids, notes, questions):
            system_msg = (
                "You are a helpful assistant for calculating a score for a given patient note. "
                "Please output answer only without any other text. "
                "Your output should only contain a JSON dict formatted as "
                '{"answer": str(value which is the answer to the question)}.'
            )
            user_msg = (
                f"Here is the patient note:\n{note}\n\n"
                f"Here is the task:\n{question}\n\n"
                'Please directly output the answer formatted as '
                '"answer": str(value which is the answer to the question):'
            )   
            prompts.append((system_msg, user_msg))
            
        return prompts

    @staticmethod
    def cot(calids: List[int], notes: List[str], questions: List[str]) -> List[Tuple[str, str]]:
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
        
        for calid, note, question in zip(calids, notes, questions):
            system_msg = (
                "You are a helpful assistant for calculating a score for a given patient note. "
                "Please think step-by-step to solve the question and then generate the required score. "
                "Your output should only contain a JSON dict formatted as "
                '{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), '
                '"answer": str(short_and_direct_answer_of_the_question)}.'
            )
            user_msg = (
                f"Here is the patient note:\n{note}\n\n"
                f"Here is the task:\n{question}\n\n"
                'Please directly output your_step_by_step_thinking_procress_to_solve_the_question, '
                'and the answer formatted as "answer": str(short_and_direct_answer_of_the_question):'
            )
            prompts.append((system_msg, user_msg))
            
        return prompts

    @staticmethod
    def one_shot(calids: List[str], notes: List[str], questions: List[str]) -> List[Tuple[str, str]]:
        """
        Create one-shot prompts for multiple medical calculation tasks.
        
        Args:
            calids: List of IDs of the specific calculators.
            notes: List of patient note texts.
            questions: List of calculation questions.
            
        Returns:
            List of tuples, where each tuple is (system_message, user_message).
        """
        # Load one-shot examples once
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "..", "data", "one_shot_finalized_explanation.json")
        try:
            with open(json_path, "r") as file:
                one_shot_json = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading one-shot examples: {e}")
        
        prompts = []
        
        for calid, note, question in zip(calids, notes, questions):
            if str(calid) not in one_shot_json:
                raise ValueError(f"Calculator ID {calid} not found in one-shot examples")
            
            example = one_shot_json[str(calid)]
            example_note = example["Patient Note"]
            example_output = {
                "step_by_step_thinking": example["Response"]["step_by_step_thinking"], 
                "answer": example["Response"]["answer"]
            }
            
            system_msg = (
                "You are a helpful assistant for calculating a score for a given patient note. "
                "Please think step-by-step to solve the question and then generate the required score. "
                "Your output should only contain a JSON dict formatted as "
                '{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), '
                '"answer": str(short_and_direct_answer_of_the_question)}.'
            )
            system_msg += f"\n\nHere is an example patient note:\n\n{example_note}"
            system_msg += f"\n\nHere is an example task:\n\n{question}"
            system_msg += (
                f'\n\nPlease directly output the JSON dict formatted as '
                f'{{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), '
                f'"answer": str(value which is the answer to the question)}}:\n\n{json.dumps(example_output)}'
            )
            
            user_msg = (
                f"Here is the patient note:\n\n{note}\n\n"
                f"Here is the task:\n\n{question}\n\n"
                'Please directly output your_step_by_step_thinking_procress_to_solve_the_question, '
                'and the answer formatted as "answer": str(short_and_direct_answer_of_the_question):'
            )
            
            prompts.append((system_msg, user_msg))
            
        return prompts

    @staticmethod
    def modular(calids: List[str], notes: List[str], questions: List[str]) -> List[Tuple[str, str]]:
        """
        Create modular prompts for multiple medical calculation tasks.
        
        Args:
            calids: List of IDs of the specific calculators.
            notes: List of patient note texts.
            questions: List of calculation questions.
            
        Returns:
            List of tuples, where each tuple is (system_message, user_message).
        """
        prompts = []
        
        for calid, note, question in zip(calids, notes, questions):
            system_msg = (
                "You are a helpful assistant for calculating a score based on a given patient note. "
                "Follow these steps when answering a question:\n"
                "1. Identify the correct formula required for the calculation.\n"
                "2. Extract the necessary values from the patient note to fill in the formula.\n"
                "3. Perform the calculation and output the final answer.\n\n"
                "Your response should be formatted as a JSON dictionary:\n"
                '{"formula": str, "extracted_values": dict, "answer": str}\n\n'
                "- `formula` should be an equation written as `required_value = value1 [operation] value2` (e.g., `BMI = weight / height^2`).\n"
                "- `extracted_values` should be a dictionary where keys are variable names and values are extracted from the patient note (e.g., {\"weight\": \"70kg\", \"height\": \"1.75m\"}).\n"
                "- `answer` should be the shortest possible numerical or textual response to the question."
            )
            user_msg = (
                f"Here is the patient note:\n{note}\n\n"
                f"Here is the task:\n{question}\n\n"
                "Please extract the relevant formula, find the required values in the patient note, and calculate the answer. "
                "Return the response in the specified JSON format."
            )
            prompts.append((system_msg, user_msg))
            
        return prompts
    
    def modular_cot(self, calids: List[str], notes: List[str], questions: List[str]) -> List[Tuple[str, str]]:
        """
        Generate chain-of-thought (CoT) prompts for modular medical calculations.
        
        Args:
            calids: List of calculator IDs.
            notes: List of patient notes.
            questions: List of questions.
        
        Returns:
            List of tuples, where each tuple is (system_message, user_message).
        """
        prompts = []
        
        for calid, note, question in zip(calids, notes, questions):
            system_msg = (
                "You are a reasoning assistant that follows a chain-of-thought approach to calculate a score based on a given patient note. "
                "Follow these steps to answer the question:\n"
                "1. Reason about which formula(s) are applicable. Then identify the correct formula required for the calculation and state it explicitly.\n"
                "2. First reason about what values are needed then explain where these values appear in the text. Then, explicitly extract the values and map them to the formula variables.\n"
                "3. Put the extracted values into the formula and perform the calculation step by step. Then provide the final answer.\n"
                "4. Return the response in the specified JSON format:\n"
                "{\"formula_reason\": str, \"formula\": str, \"extracted_values_reason\": str, \"extracted_values\": dict, \"calculation_steps\": str, \"answer\": str}\n\n"
                "- `formula_reason`: The reasons that a formula is applicable.\n"
                "- `formula`: The explicit mathematical equation used for the calculation (e.g., `BMI = weight / height^2`).\n"
                "- `extracted_values_reason`: Justification for how each value was identified.\n"
                "- `extracted_values`: A dictionary mapping variable names to extracted values from the note (e.g., {\"weight\": \"70kg\", \"height\": \"1.75m\"}).\n"
                "- `calculation_steps`: A detailed step-by-step breakdown of how the extracted values are applied to the formula.\n"
                "- `answer`: The final result in its simplest form."
            )
            
            user_msg = (
                f"Here is the patient note:\n"
                f"{note}\n\n"
                f"Here is the task:\n"
                f"{question}\n\n"
                "Please reason through each step carefully, providing justifications before stating the formula, extracted values, and final answer. Return the response in the specified JSON format."
            )
            
            prompts.append((system_msg, user_msg))
        
        return prompts
        
    def _build_records(
        self,
        model_name: str,
        include_evaluation: bool,
    ) -> List[Dict]:
        """
        Create a list of per-row dictionaries ready for json.dump().
        Uses **positional** indexing (0 … N-1) - identical to the original
        implementation - so DataFrame index values can never go out of bounds.
        """

        if not hasattr(self, "df"):
            raise RuntimeError("Dataset not loaded. Call generate_raw() first.")

        # --- base fields always present 
        fields: Dict[str, List] = {
            "LLM Original Answer": self.responses[model_name],
            "Input Tokens":        self.input_tokens[model_name],
            "Output Tokens":       self.output_tokens[model_name],
        }

        # --- optional history field ---
        # if self.history is defined as a non-empty dict, include its entries
        # print(self.history)
        if hasattr(self, "history") and isinstance(self.history, dict) and self.history:
            # get per-row history list for this model (empty list if key missing)
            history = self.history.get(model_name)
            fields["History"] = history

        # --- optional evaluator outputs 
        if include_evaluation:
            fields.update(self.correctness[model_name])

        # --- shape validation 
        n_rows = len(self.df)
        for key, values in fields.items():
            if len(values) != n_rows:
                raise ValueError(
                    f"Field '{key}' length mismatch: {len(values)} vs {n_rows}"
                )

        # --- build records 
        records: List[Dict] = []
        for i in range(n_rows):                     # <- positional loop
            row = self.df.iloc[i]                   # safe positional access

            record = {
                "Model Name":             model_name,
                "Row Number":             row["Row Number"],
                "Calculator Name":        row["Calculator Name"],
                "Calculator ID":          row["Calculator ID"],
                "Category":               row["Category"],
                "Note ID":                row["Note ID"],
                "Patient Note":           row["Patient Note"],
                "Question":               row["Question"],
                "Ground Truth Answer":    row["Ground Truth Answer"],
                "Ground Truth Explanation": row["Ground Truth Explanation"],
                "Relevant Entities":      row["Relevant Entities"],
                "Upper Limit":            row["Upper Limit"],
                "Lower Limit":            row["Lower Limit"],
            }

            # append runtime‑generated fields
            for key, values in fields.items():
                record[key] = values[i]

            records.append(record)

        return records


    @staticmethod
    def _dump_json(data, path: os.PathLike) -> None:
        """Pretty-print JSON helper."""
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, default=str, indent=4)