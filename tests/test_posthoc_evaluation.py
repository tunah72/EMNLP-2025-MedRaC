import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seminar_demo.posthoc_evaluation import (
    PosthocEvaluationConfig,
    build_evaluation_plan,
    run_posthoc_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]
STEPS = ("formula", "extracted_values", "calculation", "answer")


def _record():
    return {
        "status": "success",
        "error": None,
        "public_structured_output": {"answer": "25.24"},
        "final_answer": "25.24",
        "rule_evaluation": {"status": "success", "result": "Correct"},
        "stepwise_evaluation": {"status": "not_run", "evaluation": None},
        "first_failed_step": None,
        "token_usage": {
            "embedding_input": 0,
            "generation_input": 10,
            "generation_output": 5,
            "code_input": 0,
            "code_output": 0,
            "evaluation_input": 0,
            "evaluation_output": 0,
        },
        "request_counts": {"chat_attempts": 1, "embedding_attempts": 0},
    }


def _sample():
    return {
        "dataframe_index": 0,
        "row_number": 1,
        "calculator_id": "2",
        "calculator_name": "Creatinine Clearance",
        "family": "equation",
        "methods": {"plain_cot": _record(), "medrac_rag": _record()},
    }


def _write_source(path: Path) -> None:
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"run_id": "parent-run"}), encoding="utf-8"
    )
    (path / "samples.json").write_text(
        json.dumps([_sample()]), encoding="utf-8"
    )
    (path / "metrics.json").write_text("{}", encoding="utf-8")
    (path / "report.md").write_text("# Parent\n", encoding="utf-8")


def _config(source: Path, output: Path | None, **overrides):
    values = {
        "source_artifact": source,
        "output_dir": output,
        "dataset_path": ROOT / "data" / "test_data.csv",
        "formula_path": ROOT / "data" / "formula_new.json",
        "evaluator_model": "openai/gpt-4.1",
        "evaluator_temperature": 0.0,
        "seed": 42,
        "batch_indices": [0],
        "methods": ["plain_cot", "medrac_rag"],
        "dry_run": False,
        "command": ["python", "scripts/evaluate_experiment_artifact.py"],
    }
    values.update(overrides)
    return PosthocEvaluationConfig(**values)


class PosthocEvaluationTests(unittest.TestCase):
    def test_plan_counts_four_requests_per_method_sample(self):
        samples = [_sample()]
        plan = build_evaluation_plan(samples)
        self.assertEqual(plan["pending_evaluation_units"], 2)
        self.assertEqual(plan["expected_chat_requests"], 8)

        samples[0]["methods"]["plain_cot"]["stepwise_evaluation"] = {
            "status": "success",
            "evaluation": {step: {"result": "Correct"} for step in STEPS},
        }
        resumed = build_evaluation_plan(samples)
        self.assertEqual(resumed["pending_evaluation_units"], 1)
        self.assertEqual(resumed["expected_chat_requests"], 4)

    def test_dry_run_makes_no_output_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "derived"
            _write_source(source)
            result = run_posthoc_evaluation(
                _config(source, output, dry_run=True)
            )
            self.assertEqual(result["api_requests_made"], 0)
            self.assertEqual(result["plan"]["expected_chat_requests"], 8)
            self.assertFalse(output.exists())

    @patch("seminar_demo.posthoc_evaluation.require_github_token")
    @patch("seminar_demo.posthoc_evaluation.GitHubModelsLLM")
    @patch("seminar_demo.posthoc_evaluation._stepwise_evaluate")
    def test_two_batches_resume_to_completed_artifact(
        self, stepwise, model_class, require_token
    ):
        evaluation = {step: {"result": "Correct"} for step in STEPS}
        stepwise.return_value = (evaluation, 100, 20, 4)
        model_class.return_value.requests_made = 0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "derived"
            _write_source(source)

            first = _config(source, output, methods=["plain_cot"])
            self.assertEqual(run_posthoc_evaluation(first), output.resolve())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "in_progress")
            self.assertEqual(
                manifest["progress"]["completed_evaluation_units"], 1
            )

            second = _config(source, output, methods=["medrac_rag"])
            self.assertEqual(run_posthoc_evaluation(second), output.resolve())
            manifest = json.loads((output / "manifest.json").read_text())
            metrics = json.loads((output / "metrics.json").read_text())
            samples = json.loads((output / "samples.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                manifest["progress"]["completed_evaluation_units"], 2
            )
            self.assertEqual(metrics["plain_cot"]["strict_all_steps_accuracy"]["value"], 1.0)
            self.assertEqual(metrics["medrac_rag"]["strict_all_steps_accuracy"]["value"], 1.0)
            self.assertEqual(len(samples[0]["methods"]["plain_cot"]["posthoc_evaluation"]["runs"]), 1)
            self.assertEqual(len(samples[0]["methods"]["medrac_rag"]["posthoc_evaluation"]["runs"]), 1)
            self.assertEqual(stepwise.call_count, 2)
            self.assertEqual(require_token.call_count, 2)


if __name__ == "__main__":
    unittest.main()
