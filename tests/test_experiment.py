import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from seminar_demo.data import load_exclusion_policy
from seminar_demo.experiment import (
    ExperimentConfig,
    aggregate_experiment,
    aggregate_method_metrics,
    load_experiment_samples,
    run_mini_experiment,
)


ROOT = Path(__file__).resolve().parents[1]


def _method_record(rule="Correct", evaluation=None):
    return {
        "status": "success",
        "rule_evaluation": {"status": "success", "result": rule},
        "stepwise_evaluation": {
            "status": "success" if evaluation else "not_run",
            "evaluation": evaluation,
        },
        "token_usage": {"generation_input": 10, "generation_output": 5},
        "request_counts": {"chat_attempts": 1, "embedding_attempts": 0},
    }


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(ROOT / "data" / "test_data.csv")
        cls.policy = load_exclusion_policy(ROOT / "config" / "paper_exclusions.json")

    def test_committed_manifest_is_deterministic_and_mixed(self):
        payload, selected = load_experiment_samples(
            ROOT / "config" / "mini_experiment.json", self.frame, self.policy
        )
        self.assertEqual(len(selected), 10)
        self.assertEqual(
            selected["DataFrame Index"].tolist(),
            [0, 40, 20, 120, 60, 200, 80, 100, 160, 280],
        )
        families = [item["family"] for item in payload["samples"]]
        self.assertEqual(families.count("equation"), 7)
        self.assertEqual(families.count("rule"), 3)

    def test_aggregation_computes_strict_conditional_and_first_error(self):
        all_correct = {
            step: {"result": "Correct"}
            for step in ("formula", "extracted_values", "calculation", "answer")
        }
        extraction_failure = {
            "formula": {"result": "Correct"},
            "extracted_values": {"result": "Incorrect"},
            "calculation": {"result": "Incorrect"},
            "answer": {"result": "Correct"},
        }
        samples = [
            {"methods": {"plain_cot": _method_record(evaluation=all_correct)}},
            {"methods": {"plain_cot": _method_record("Incorrect", extraction_failure)}},
        ]
        metrics = aggregate_method_metrics(samples, "plain_cot")
        self.assertEqual(metrics["rule_based_final_answer_accuracy"]["value"], 0.5)
        self.assertEqual(metrics["strict_all_steps_accuracy"]["value"], 0.5)
        self.assertEqual(
            metrics["conditional_correctness"]["extracted_values"]["value"], 0.5
        )
        self.assertEqual(
            metrics["conditional_correctness"]["calculation"]["denominator"], 1
        )
        self.assertEqual(
            metrics["first_error_attribution"]["counts"]["extracted_values"], 1
        )
        self.assertEqual(metrics["first_error_attribution"]["denominator"], 1)
        self.assertEqual(
            metrics["first_error_attribution"]["rates"]["extracted_values"]["value"],
            1.0,
        )
        self.assertEqual(metrics["masked_reasoning_failures"]["count"], 1)
        self.assertEqual(metrics["masked_reasoning_failures"]["value"], 0.5)

    def test_first_error_rates_condition_only_on_strict_failures(self):
        all_correct = {
            step: {"result": "Correct"}
            for step in ("formula", "extracted_values", "calculation", "answer")
        }
        formula_failure = {
            "formula": {"result": "Incorrect"},
            "extracted_values": {"result": "Correct"},
            "calculation": {"result": "Correct"},
            "answer": {"result": "Correct"},
        }
        calculation_failure = {
            "formula": {"result": "Correct"},
            "extracted_values": {"result": "Correct"},
            "calculation": {"result": "Incorrect"},
            "answer": {"result": "Incorrect"},
        }
        samples = [
            {"methods": {"plain_cot": _method_record(evaluation=all_correct)}},
            {"methods": {"plain_cot": _method_record(evaluation=formula_failure)}},
            {"methods": {"plain_cot": _method_record(evaluation=calculation_failure)}},
        ]
        metrics = aggregate_method_metrics(samples, "plain_cot")
        attribution = metrics["first_error_attribution"]
        self.assertEqual(attribution["complete_denominator"], 3)
        self.assertEqual(attribution["denominator"], 2)
        self.assertEqual(attribution["rates"]["formula"]["value"], 0.5)
        self.assertEqual(attribution["rates"]["calculation"]["value"], 0.5)
        self.assertAlmostEqual(
            sum(
                item["value"]
                for item in attribution["rates"].values()
                if item["value"] is not None
            ),
            1.0,
        )

    def test_first_error_rates_are_null_without_strict_failures(self):
        all_correct = {
            step: {"result": "Correct"}
            for step in ("formula", "extracted_values", "calculation", "answer")
        }
        samples = [
            {"methods": {"plain_cot": _method_record(evaluation=all_correct)}}
        ]
        metrics = aggregate_method_metrics(samples, "plain_cot")
        attribution = metrics["first_error_attribution"]
        self.assertEqual(attribution["denominator"], 0)
        self.assertTrue(
            all(item["value"] is None for item in attribution["rates"].values())
        )
        self.assertEqual(metrics["masked_reasoning_failures"]["count"], 0)

    def test_missing_stepwise_metrics_are_null_not_zero(self):
        samples = [{"methods": {"plain_cot": _method_record()}}]
        metrics = aggregate_method_metrics(samples, "plain_cot")
        self.assertEqual(metrics["stepwise_complete_samples"], 0)
        self.assertIsNone(metrics["strict_all_steps_accuracy"]["value"])
        self.assertIsNone(metrics["step_accuracy"]["formula"]["value"])

    def test_paired_comparison_excludes_method_failures(self):
        success = _method_record("Correct")
        incorrect = _method_record("Incorrect")
        failed = _method_record("Incorrect")
        failed["status"] = "failed"
        failed["rule_evaluation"] = {"status": "not_run", "result": None}
        samples = [
            {"methods": {"plain_cot": incorrect, "medrac_rag": success}},
            {"methods": {"plain_cot": success, "medrac_rag": failed}},
        ]
        metrics = aggregate_experiment(samples)["paired_comparison"]
        self.assertEqual(metrics["eligible_samples"], 1)
        self.assertEqual(metrics["excluded_due_to_method_failure"], 1)
        self.assertEqual(metrics["plain_cot_accuracy"]["value"], 0.0)
        self.assertEqual(metrics["medrac_rag_accuracy"]["value"], 1.0)
        self.assertEqual(metrics["outcomes"]["medrac_only_correct"], 1)

    @patch("seminar_demo.experiment.validate_canonical_index")
    def test_dry_run_generates_complete_experiment_artifacts(self, validate_index):
        validate_index.return_value = {
            "document_count": 55,
            "embedding_model": "openai/text-embedding-3-small",
        }
        with tempfile.TemporaryDirectory() as temporary:
            config = ExperimentConfig(
                dataset_path=ROOT / "data" / "test_data.csv",
                formula_path=ROOT / "data" / "formula_new.json",
                exclusion_path=ROOT / "config" / "paper_exclusions.json",
                sample_manifest_path=ROOT / "config" / "mini_experiment.json",
                index_dir=Path(temporary) / "index",
                output_root=Path(temporary) / "artifacts",
                generator_model="openai/gpt-4.1",
                evaluator_model="openai/gpt-4.1",
                embedding_model="openai/text-embedding-3-small",
                temperature=0.0,
                seed=42,
                use_llm_evaluation=False,
                dry_run=True,
                limit=1,
                timeout_seconds=2.0,
                command=["python", "scripts/run_mini_experiment.py", "--dry-run"],
            )
            output = run_mini_experiment(config, self.policy)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"manifest.json", "samples.json", "metrics.json", "report.md"},
            )
            manifest = json.loads((output / "manifest.json").read_text())
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(manifest["status"], "dry_run_validated")
            self.assertEqual(manifest["dataset"]["selected_dataframe_indices"], [0])
            self.assertIsNone(
                metrics["plain_cot"]["rule_based_final_answer_accuracy"]["value"]
            )


if __name__ == "__main__":
    unittest.main()
