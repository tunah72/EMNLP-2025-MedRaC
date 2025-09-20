import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Union, Set
from collections import Counter

from evaluator import RegEvaluator


def extract_row_numbers(json_path):
    """
    Read a JSON file containing a list of entries and extract the Row Number for:
      1. Entries whose Calculator ID is "13".
      2. Entries whose Ground Truth Answer is a negative decimal (contains a decimal point), e.g. "-1.222" or "-1.0".
         Negative integers like "-1" or "-2" are ignored.
      3. Entries whose Calculator ID is "28" or "11".
      4. Entries whose Row Number is 451 or 236.
    Returns:
        A list of unique Row Number values (as strings).
    Raises:
        ValueError: if the file is not valid JSON or not a list at the root.
    """
    TARGET_CALCULATOR_ID = "13"
    ADDITIONAL_CALCULATOR_IDS = {"28", "11", "36"}
    ADDITIONAL_ROW_NUMS = {"451", "236"}
    NEG_DECIMAL_PATTERN = re.compile(r"^-\d+\.\d+$")
    row_numbers = set()

    # Load JSON from file
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data_list = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    # Ensure the root is a list
    if not isinstance(data_list, list):
        raise ValueError("Expected JSON root to be a list of entries")

    # Iterate over each entry in the list
    for entry in data_list:
        calculator_id = str(entry.get("Calculator ID", "")).strip()
        ground_truth = str(entry.get("Ground Truth Answer", "")).strip()
        row_num = entry.get("Row Number")

        # Skip if Row Number is missing
        if row_num is None:
            continue

        row_str = str(row_num).strip()

        # Condition 1: Calculator ID equals TARGET_CALCULATOR_ID
        if calculator_id == TARGET_CALCULATOR_ID:
            row_numbers.add(row_str)

        # Condition 2: Ground Truth Answer is a negative decimal
        if NEG_DECIMAL_PATTERN.match(ground_truth):
            row_numbers.add(row_str)

        # Condition 3: Additional calculator IDs
        if calculator_id in ADDITIONAL_CALCULATOR_IDS:
            row_numbers.add(row_str)

        # Condition 4: Additional row numbers
        if row_str in ADDITIONAL_ROW_NUMS:
            row_numbers.add(row_str)

    # Return as a list of strings
    return list(row_numbers)




# Make sure extract_row_numbers is defined or imported in this module:
# from your_module import extract_row_numbers

def clean_json_files_in_directory(root_path: str):
    """
    Recursively traverse `root_path`, find all .json files, and for each:
      1. Load the JSON (expected to be a list of entry dicts).
      2. Identify rows to remove by calling extract_row_numbers(file_path).
      3. Remove those entries from the list.
      4. Overwrite the original file with the cleaned list.
      5. Print status messages in English:
         - Processing file: <filepath>
         - Total rows: <total>
         - Problematic rows: <problem_count>
         - Rows remaining: <remaining>
    """
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if not fname.lower().endswith(".json"):
                continue

            file_path = os.path.join(dirpath, fname)
            print(f"Processing file: {file_path}")

            # Load JSON data
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except (IOError, json.JSONDecodeError) as err:
                print(f"  - Failed to read or parse JSON: {err}")
                continue

            # Expect the root JSON to be a list of dicts
            if not isinstance(entries, list):
                print("  - JSON root is not a list, skipping file.")
                continue

            total = len(entries)

            # Determine which row numbers to remove
            # extract_row_numbers should return a list of strings
            to_remove: Set[str] = set(extract_row_numbers(file_path))
            problem_count = len(to_remove)

            # Filter out problematic entries
            cleaned = [
                entry for entry in entries
                if str(entry.get("Row Number")) not in to_remove
            ]
            remaining = len(cleaned)

            # Overwrite the file with cleaned data
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(cleaned, f, ensure_ascii=False, indent=2)
            except IOError as err:
                print(f"  - Failed to write cleaned data: {err}")
                continue

            # Print summary
            print(f"  Total rows: {total}")
            print(f"  Problematic rows: {problem_count}")
            print(f"  Rows remaining: {remaining}\n")











