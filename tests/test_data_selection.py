import unittest
from pathlib import Path

import pandas as pd

from seminar_demo.data import apply_exclusion_policy, load_exclusion_policy, select_samples


ROOT = Path(__file__).resolve().parents[1]


class DataSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(ROOT / "data" / "test_data.csv", encoding="utf-8")
        cls.policy = load_exclusion_policy(ROOT / "config" / "paper_exclusions.json")

    def test_policy_reproduces_940_row_cohort(self):
        retained, excluded = apply_exclusion_policy(self.frame, self.policy)
        self.assertEqual(len(excluded), 108)
        self.assertEqual(len(retained), 940)
        self.assertEqual(excluded["Row Number"].nunique(), 108)

    def test_default_selection_is_fixed_equation_and_rule_pair(self):
        selected = select_samples(self.frame, self.policy)
        self.assertEqual(selected["DataFrame Index"].tolist(), [0, 40])
        self.assertEqual(selected["Calculator ID"].astype(str).tolist(), ["2", "4"])

    def test_excluded_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "excluded by policy"):
            select_samples(self.frame, self.policy, row_indices=[180])

    def test_calculator_selection_is_deterministic(self):
        first = select_samples(self.frame, self.policy, calculator_ids=["2", "4"])
        second = select_samples(self.frame, self.policy, calculator_ids=["2", "4"])
        self.assertEqual(first["DataFrame Index"].tolist(), second["DataFrame Index"].tolist())


if __name__ == "__main__":
    unittest.main()
