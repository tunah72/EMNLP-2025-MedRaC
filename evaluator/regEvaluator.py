from typing import List, Union, Optional, Tuple
import os
import json
import re
import math
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from .baseEvaluator import Evaluator

logger = logging.getLogger(__name__)



class RegEvaluator(Evaluator):
    def __init__(self):
        """
        Initialize an RegEvaluator instance for assessing LLM responses against ground truth.
        """
        pass

    @staticmethod
    def _extract_answer(responses: List[Union[str, dict]], calids: List[Union[str, int]]) -> List[str]: 
        """
        Extract final answers from LLM responses.
        
        Args:
            responses: List of responses directly from the LLM
            calid: List of calculator IDs corresponding to each response
            
        Returns:
            List of extracted answers with units (not directly comparable to ground truth)
        """
        results = []
        
        for answer, cal_id in zip(responses, calids):
            cal_id = int(cal_id)
            
            # Try to parse as JSON first
            if isinstance(answer, dict):
                extracted_answer = answer.get("answer") or answer.get("Answer") or "Not Found"
            else:
                try:
                    data = json.loads(answer)
                    if isinstance(data, dict):
                        extracted_answer = data.get("answer") or data.get("Answer") or "Not Found"
                    else:
                        extracted_answer = "Not Found"      
                    # Explanation is extracted but not returned
                    # _ = data.get("step_by_step_thinking") or "No Explanation"
                except json.JSONDecodeError:
                    # Fall back to regex if JSON parsing fails
                    matches_answer = re.findall(r'(?:"[Aa]nswer":|\*\*[Aa]nswer:\*\*)\s*(?:"(.*?)"|([^,\}\s]+))', answer)
                    if matches_answer:
                        extracted_answer = matches_answer[-1][0] if matches_answer[-1][0] else matches_answer[-1][1]
                        extracted_answer = extracted_answer.strip().strip('"')
                    else:
                        matches_answer = re.findall(r'[Aa]nswer:\s*(?:"(.*?)"|([^,\}\s]+))', answer)
                        if matches_answer:
                            extracted_answer = matches_answer[-1][0] if matches_answer[-1][0] else matches_answer[-1][1]
                            extracted_answer = extracted_answer.strip().strip('"')
                        else:
                            extracted_answer = "Not Found"
                
            
            # Ensure answer is a string
            if not isinstance(extracted_answer, str):
                extracted_answer = str(extracted_answer)
            
            # Filter out template placeholders
            if extracted_answer in [
                "str(short_and_direct_answer_of_the_question)",
                "str(value which is the answer to the question)",
                "X.XX"
            ]:
                extracted_answer = "Not Found"
            
            # Process based on calculator type
            if cal_id in [13, 68]:  # Date formats
                match = re.search(r"^(0?[1-9]|1[0-2])\/(0?[1-9]|[12][0-9]|3[01])\/(\d{4})", extracted_answer)
                if match:
                    month = int(match.group(1))
                    day = int(match.group(2))
                    year = match.group(3)
                    answer_val = f"{month:02}/{day:02}/{year}"
                else:
                    answer_val = "N/A"
            elif cal_id in [69]:  # Weeks/days format
                match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", extracted_answer)
                if match:
                    weeks = match.group(1)
                    days = match.group(3)
                    answer_val = f"({weeks}, {days})"
                else:
                    answer_val = "N/A"
            elif cal_id in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51]:  # Count or score
                match = re.search(r"(\d+) out of", extracted_answer)
                if match:
                    answer_val = match.group(1)
                else:
                    match = re.search(r"-?\d+(, ?-?\d+)+", extracted_answer)
                    if match:
                        answer_val = str(len(match.group(0).split(",")))
                    else:
                        match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
                        if len(match) > 0:
                            answer_val = match[-1][0]
                        else:
                            answer_val = "N/A"
            elif cal_id in [2, 3, 5, 6, 7, 8, 9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44, 46, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]:  # Numerical calculations
                # Handle expressions like "str(123.45)"
                match = re.search(r"str\((.*)\)", extracted_answer)
                if match:
                    expression = match.group(1)
                    # Clean up the expression for safe eval
                    expression = (expression.replace("^", "**")
                                .replace("is odd", "% 2 == 1")
                                .replace("is even", "% 2 == 0")
                                .replace("sqrt", "math.sqrt")
                                .replace(".math", ""))
                    expression = expression.split('#')[0]
                    
                    # Balance parentheses
                    if expression.count('(') > expression.count(')'):
                        expression += ')' * (expression.count('(') - expression.count(')'))
                    elif expression.count(')') > expression.count('('):
                        expression = '(' * (expression.count(')') - expression.count('(')) + expression
                    
                    # Safely evaluate the expression
                    try:
                        answer_val = eval(expression, {"__builtins__": None}, {
                            "min": min, "pow": pow, "round": round, "abs": abs,
                            "int": int, "float": float, "math": math, "np": np, "numpy": np
                        })
                    except Exception as e:
                        logger.error(f"Error in evaluating expression: {expression}\n{e}")
                        answer_val = "N/A"
                else:
                    # Try to extract values with units
                    match = re.search(r"(-?\d+(\.\d+)?)\s*mL/min/1.73", extracted_answer)
                    if match:
                        answer_val = eval(match.group(1))
                    else:
                        match = re.findall(r"(-?\d+(\.\d+)?)\%", extracted_answer)
                        if len(match) > 0:
                            answer_val = eval(match[-1][0]) / 100
                        else:
                            match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
                            if len(match) > 0:
                                try:
                                    answer_val = eval(match[-1][0])
                                except:
                                    answer_val = "N/A"
                            else:
                                answer_val = "N/A"
                
                # Convert to string
                if answer_val != "N/A":
                    answer_val = str(answer_val)
            else:
                answer_val = extracted_answer
            
            # Standardize error values
            if answer_val == "N/A":
                answer_val = "Not Found"
            
            results.append(answer_val)
        
        return results
    
    @staticmethod
    def check_correctness(
        responses: List[str], 
        ground_truths: List[str], 
        calids: List[Union[str, int]],
        upper_limits: List[Union[str, float]], 
        lower_limits: List[Union[str, float]],
        **kwargs
    ) -> Tuple[str, List[str]]:
        """
        Check correctness of answers against ground truth.
        
        Args:
            responses: List of LLM responses
            ground_truths: List of ground truth answers
            calid: List of calculator IDs for each answer
            upper_limit: Upper bounds for acceptable answers
            lower_limit: Lower bounds for acceptable answers
            
        Returns:
            Tuple with key "Result" and list of "Correct"/"Incorrect" values.
        """
        results = []

        answers = RegEvaluator._extract_answer(responses, calids)

        for answer, truth, cal_id, upper, lower in zip(answers, ground_truths, calids, upper_limits, lower_limits):
            upper = str(upper)
            lower = str(lower)
            cal_id = int(cal_id)

            if answer == "Not Found" or answer == "N/A":
                results.append(0)
                continue

            try:
                if cal_id in [13, 68]:  # Date format
                    fmt = "%m/%d/%Y"
                    a = datetime.strptime(answer, fmt).strftime("%-m/%-d/%Y")
                    t = datetime.strptime(truth, fmt).strftime("%-m/%-d/%Y")
                    results.append(int(a == t))

                elif cal_id in [69]:  # Weeks/days
                    def parse_week_day(val):
                        m = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", val)
                        return eval(f"({m.group(1)}, {m.group(3)})") if m else None
                    results.append(int(parse_week_day(answer) == parse_week_day(truth)))

                elif cal_id in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69]:  # Int
                    results.append(int(round(eval(answer)) == eval(truth)))

                elif cal_id in [2, 3, 5, 6, 7, 8, 9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44, 46, 49,
                                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]:  # Float range
                    val = eval(answer)
                    results.append(int(eval(lower) <= val <= eval(upper)))

                else:
                    raise ValueError(f"Unknown calculator ID: {cal_id}")

            except Exception:
                results.append(0)

        return "Result", ["Correct" if r == 1 else "Incorrect" for r in results]


    @staticmethod


    def check_correctness_parsed(
            responses:   List[str],
            ground_truths: List[str],
            calids:      List[Union[str, int]],
            upper_limits: List[Union[str, float]],
            lower_limits: List[Union[str, float]],
            **kwargs
        ) -> Tuple[str, List[str]]:
        def _to_float(x) -> Optional[float]:
            NUM = re.compile(r'-?\d+\.?\d*(?:e-?\d+)?')
            m = NUM.search(str(x))
            return float(m.group()) if m else None

        def _decimal_places(txt: str) -> int:
            """
            Return the number of digits after the decimal point in the
            *literal* string. Trailing zeros count; scientific notation counts as 0.
            """
            m = re.search(r'-?\d+\.(\d+)', txt)
            return len(m.group(1)) if m else 0

        """Rule-based correctness checker."""
        results = []

        for ans, truth, cid, _, _ in zip(responses, ground_truths, calids,
                                        upper_limits, lower_limits):
            cid = int(cid)
            ans_s, truth_s = str(ans).strip(), str(truth).strip()

            # --- special N/A ---
            if ans_s in {"N/A", "Not Found", ""}:
                results.append(0)
                continue

            try:
                # dates
                if cid in {13, 68}:
                    fmt = "%m/%d/%Y"
                    ok = datetime.strptime(ans_s, fmt).date() == \
                        datetime.strptime(truth_s, fmt).date()
                    results.append(int(ok))

                # weeks + days
                elif cid == 69:
                    def _wkday(txt):
                        m = re.search(r'(\d+)\s*weeks?.*?(\d+)\s*days?', txt)
                        return tuple(map(int, m.groups())) if m else None
                    results.append(int(_wkday(ans_s) == _wkday(truth_s)))

                # integers
                elif cid in {4,15,16,17,18,20,21,25,27,28,29,32,33,36,43,45,48,51}:
                    ok = int(round(float(ans_s))) == int(round(float(truth_s)))
                    results.append(int(ok))

                # floats
                elif cid in {
                    2,3,5,6,7,8,9,10,11,19,22,23,24,26,
                    30,31,38,39,40,44,46,49,56,57,58,59,60,
                    61,62,63,64,65,66,67
                }:
                    val       = _to_float(ans_s)
                    truth_val = _to_float(truth_s)

                    if val is None or truth_val is None:
                        results.append(0)
                        continue

                    # Determine tolerance from *response* decimal places
                    d         = _decimal_places(ans_s)
                    tolerance = 0.5 / (10 ** d)  # e.g. d=2 → 0.005

                    # Primary check using response-based tolerance
                    ok = abs(val - truth_val) <= tolerance

                    # Secondary fallback: always allow two-decimal tolerance (±0.005)
                    if not ok and abs(val - truth_val) <= 0.005:
                        ok = True  # treat as correct if within 2-decimal tolerance

                    results.append(int(ok))

                else:
                    raise ValueError(f"Unknown calculator ID {cid}")

            except Exception:
                results.append(0)

        return "Result", ["Correct" if r else "Incorrect" for r in results]



    @staticmethod
    def compute_overall_accuracy(input_file_path: str, output_dir_path: str):
        """
        Compute accuracy for a single input file (JSON or JSONL) and write results.
        
        Args:
            input_file_path: Path to the JSONL or JSON file containing evaluation results
            output_dir_path: Directory where results will be written; will contain a
                            file named "results_<basename>.json"
        """
        file_name = os.path.basename(input_file_path)
        base_name = os.path.splitext(file_name)[0]  # filename without extension
        os.makedirs(output_dir_path, exist_ok=True)

        # Determine extension
        ext = Path(input_file_path).suffix.lower()

        # Load data
        datas = []
        with open(input_file_path, 'r', encoding='utf-8') as f:
            if ext == '.json':
                # entire file is a JSON array
                datas = json.load(f)
            elif ext == '.jsonl':
                # one JSON object per line
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    datas.append(json.loads(line))
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        # Accumulators
        category_accuracy = {}
        output_tokens_list = []
        input_tokens_list  = []

        # Process each record
        for data in datas:
            # If no token count is provided, set to -1
            input_tokens_list.append(data.get("Input Tokens", -1))
            output_tokens_list.append(data.get("Output Tokens", -1))

            category = data.get("Category")
            if category not in category_accuracy:
                category_accuracy[category] = []

            if str(data.get("Result")) in ["Correct", "1"]:
                category_accuracy[category].append(1)
            else:
                category_accuracy[category].append(0)

        # Compute per‑category stats
        category_stats = {}
        all_results = []
        for cat, results in category_accuracy.items():
            arr = np.array(results)
            mean = arr.mean() if len(arr) else 0.0
            std  = round(np.sqrt(mean * (1 - mean) / len(arr)), 2) if len(arr) else 0.0

            category_stats[cat] = {
                "average": round(mean * 100, 2),
                "std": std,
                "count": len(arr)
            }
            all_results.extend(results)

        # Overall stats
        all_arr = np.array(all_results)
        o_mean = all_arr.mean() if len(all_arr) else 0.0
        o_std  = round(np.sqrt(o_mean * (1 - o_mean) / len(all_arr)), 2) if len(all_arr) else 0.0
        category_stats["overall"] = {
            "average": round(o_mean * 100, 2),
            "std": o_std,
            "count": len(all_arr)
        }

        # Token averages
        category_stats["input_tokens_average"]  = int(round(np.mean(input_tokens_list)))  if input_tokens_list  else 0
        category_stats["output_tokens_average"] = int(round(np.mean(output_tokens_list))) if output_tokens_list else 0

        # Write out
        output_file = os.path.join(output_dir_path, f"results_{base_name}.json")
        with open(output_file, 'w', encoding='utf-8') as wf:
            json.dump(category_stats, wf, indent=4, ensure_ascii=False)

        logger.info(f"Accuracy statistics saved to '{output_file}'")
        return category_stats
    
    def compute_multifile_overall_accuracy(self, input_dir_path: str, output_dir_path: str):
        """
        Compute accuracy for all files in a directory.
        
        Args:
            input_dir_path: Directory containing the JSONL file containing evaluation results
            output_dir_path: Directory where results will be written, should automatically prepend "results_" before the individual input file names under the output directory
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir_path, exist_ok=True)
        
        # Get all JSONL files in the input directory
        input_files = [f for f in os.listdir(input_dir_path) if f.endswith('.jsonl')]
        
        if not input_files:
            logger.warning(f"No JSONL files found in directory: {input_dir_path}")
            return
        
        # Process each file individually
        processed_files = 0
        for file_name in input_files:
            input_file_path = os.path.join(input_dir_path, file_name)
            
            try:
                # Process each file using the compute_overall_accuracy method
                self.compute_overall_accuracy(input_file_path, output_dir_path)
                processed_files += 1
                
                logger.info(f"Processed file {processed_files}/{len(input_files)}: {file_name}")
                
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {str(e)}")
        
        logger.info(f"Completed processing {processed_files} out of {len(input_files)} files.")
    
    @staticmethod
    def parse_answer_agent(responses: List[Union[str, dict, None]]) -> List[str]:
        """
        Extract the core "answer" from a variety of agent outputs, with robust
        error handling and mandatory DeepSeek-style post-</think> slicing.

        Workflow for each item in `responses`:
        1. **DeepSeek slicing**:
        If the raw item is a string containing "</think>", drop everything
        up to and including the first occurrence of that marker, then collapse
        any excess whitespace.  Otherwise leave it as-is (later coerced to str).
        2. **Dictionary fast-path**:
        If the original item is a dict, immediately return its "answer" or
        "Answer" field (if present), or "Not Found" otherwise.
        3. **JSON parsing**:
        Attempt to `json.loads` the possibly-sliced string.  If this yields a
        dict, return its "answer"/"Answer" field (or "Not Found").
        4. **Regex extraction**:
        Apply two patterns in order:
            a) Primary: covers `"Answer": "…"`, `'answer':…`, `**Answer:** …`, etc.
            b) Fallback: plain `Answer: …`.
        5. **Fallback**:
        On any error, missing key, or empty match, produce the literal
        "Not Found".

        This function never raises: every input—regardless of type—yields exactly
        one string in the output list.  Unexpected exceptions at any stage are
        caught and treated as "Not Found".
        """

        results: List[str] = []
        deepseek_marker = "</think>"

        # Compile regexes once for efficiency
        primary_re = re.compile(
            r'(?:"[Aa]nswer"\s*:|\'[Aa]nswer\'\s*:|\*\*[Aa]nswer:\*\*|\*\*[Aa]nswer\*\*\s*:)\s*'
            r'(?:"([^"]*?)"|([^,\}\]\s]+))'
        )
        fallback_re = re.compile(
            r'[Aa]nswer\s*:\s*(?:"([^"]*?)"|([^,\}\]\s]+))'
        )

        for item in responses:
            # 1) If it's a string containing the DeepSeek marker, slice off the prefix
            if isinstance(item, str) and deepseek_marker in item:
                # Keep only what's after the first "</think>"
                sliced = item.split(deepseek_marker, 1)[1]
                working = " ".join(sliced.split())
            else:
                # Otherwise convert whatever it is (None, number, dict, etc.) to a string
                working = "" if item is None else str(item)

            # Default result if all else fails
            extracted = "Not Found"

            # 2) If the original was a dict, take its answer field immediately
            if isinstance(item, dict):
                extracted = item.get("answer") or item.get("Answer") or "Not Found"
                results.append(extracted or "Not Found")
                continue

            # 3) Try JSON parsing on the (possibly sliced) string
            try:
                parsed = json.loads(working)
                if isinstance(parsed, dict):
                    extracted = parsed.get("answer") or parsed.get("Answer") or "Not Found"
            except Exception:
                # On JSON errors (including non-dict values), fall back to regex
                pass

            # 4a) Primary regex if still missing
            if extracted in ("Not Found", "", None):
                matches = primary_re.findall(working)
                if matches:
                    # choose last match, prefer group 1
                    extracted = matches[-1][0] or matches[-1][1] or "Not Found"

            # 4b) Fallback regex if still missing
            if extracted in ("Not Found", "", None):
                matches = fallback_re.findall(working)
                if matches:
                    extracted = matches[-1][0] or matches[-1][1] or "Not Found"

            # 5) Final cleanup: strip extra quotes/whitespace
            extracted = (extracted or "").strip().strip('"') or "Not Found"
            results.append(extracted)

        return results
    
    def get_evaluator_name(self) -> str:
        """
        Get the name of the evaluator.
        
        Returns:
            str: Name of the evaluator
        """
        return "RegEvaluator"

    
