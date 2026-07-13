import unittest
from pathlib import Path

from seminar_demo.pipeline import first_failed_step, validate_index_artifacts


ROOT = Path(__file__).resolve().parents[1]


class PipelineHelperTests(unittest.TestCase):
    def test_first_failed_step_uses_feca_order(self):
        evaluation = {
            "formula": {"result": "Correct"},
            "extracted_values": {"result": "Incorrect"},
            "calculation": {"result": "Incorrect"},
            "answer": {"result": "Correct"},
        }
        self.assertEqual(first_failed_step(evaluation), "extracted_values")

    def test_first_failed_step_none_when_all_correct(self):
        evaluation = {
            step: {"result": "Correct"}
            for step in ("formula", "extracted_values", "calculation", "answer")
        }
        self.assertIsNone(first_failed_step(evaluation))

    def test_checked_in_index_hashes(self):
        result = validate_index_artifacts(
            ROOT / "data" / "one_shot_finalized_explanation_formulas_embeddings"
        )
        self.assertEqual(set(result), {"index.faiss", "index.pkl"})


if __name__ == "__main__":
    unittest.main()
