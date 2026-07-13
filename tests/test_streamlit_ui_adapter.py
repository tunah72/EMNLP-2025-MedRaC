import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from method.medRaC import MedRaC
from method.plain import Plain
from seminar_demo.experiment import run_medrac_sample, single_sample_batch_args
from seminar_demo.safe_execution import SafeExecutionResult
from seminar_demo.ui_adapter import (
    derive_stepwise_status,
    discover_replay_runs,
    medrac_only_download_payload,
    normalize_artifact_sample,
    normalize_calculator_id,
    validate_live_configuration,
)


class _FakeRag:
    last_embedding_input_tokens = 9

    def __init__(self):
        self.queries = []

    def retrieve(self, query, k=1):
        self.queries.append((query, k))
        return [("result = value", 0.91)]


class _FakeGenerator:
    def __init__(self, code="result = 4"):
        self.code = code
        self.calls = []

    def generate(self, prompts, schema=None, show_progress=False):
        self.calls.append((prompts, schema, show_progress))
        if schema is not None:
            return [({"extracted_values": [{"key": "value", "value": "4"}]}, 11, 3)]
        return [(self.code, 7, 2)]


def _row(note="separate patient note", question="separate required question"):
    return pd.Series(
        {
            "DataFrame Index": 0,
            "Row Number": 1,
            "Calculator ID": 2,
            "Calculator Name": "Example calculator",
            "Patient Note": note,
            "Question": question,
            "Ground Truth Answer": "4",
            "Upper Limit": None,
            "Lower Limit": None,
            "Ground Truth Explanation": "result = 4",
            "Relevant Entities": "value = 4",
        }
    )


class StreamlitUIAdapterTests(unittest.TestCase):
    def test_normalize_calculator_id(self):
        self.assertEqual(normalize_calculator_id(2), "2")
        self.assertEqual(normalize_calculator_id(2.0), "2")
        self.assertEqual(normalize_calculator_id(" 2 "), "2")
        with self.assertRaises(ValueError):
            normalize_calculator_id(" ")

    def test_patient_note_clinical_note_and_question_provenance(self):
        normalized = normalize_artifact_sample(
            {
                "Calculator ID": 4,
                "Clinical Note": "original clinical text",
                "Question": "original question text",
            }
        )
        self.assertEqual(normalized["patient_note"], "original clinical text")
        self.assertEqual(normalized["patient_note_source"], "Clinical Note")
        self.assertEqual(normalized["question"], "original question text")
        self.assertEqual(normalized["question_source"], "Question")

    def test_missing_question_is_preserved_as_missing_for_replay(self):
        normalized = normalize_artifact_sample(
            {"calculator_id": "2", "input": {"patient_note": "note only"}}
        )
        self.assertTrue(normalized["question_missing"])
        self.assertIsNone(normalized["question"])

    def test_live_batch_rejects_missing_question(self):
        with self.assertRaisesRegex(ValueError, "Question is required"):
            single_sample_batch_args(_row(question=""))

    def test_plain_prompt_receives_three_fields_in_correct_positions(self):
        row = _row(note="NOTE_SENTINEL", question="QUESTION_SENTINEL")
        args = single_sample_batch_args(row)
        baseline = Plain(prompt_style="cot", llms=[], evaluators=[])
        baseline.prompt_fn = Mock(wraps=baseline.prompt_fn)
        prompts = baseline.prompt_fn(*args)
        baseline.prompt_fn.assert_called_once_with(
            ["2"], ["NOTE_SENTINEL"], ["QUESTION_SENTINEL"]
        )
        self.assertIn("NOTE_SENTINEL", prompts[0][1])
        self.assertIn("QUESTION_SENTINEL", prompts[0][1])

    def test_runner_propagates_question_and_never_calls_unrestricted_exec(self):
        row = _row(note="NOTE_SENTINEL", question="QUESTION_SENTINEL")
        rag = _FakeRag()
        generator = _FakeGenerator()
        with (
            patch(
                "seminar_demo.experiment.execute_safely",
                return_value=SafeExecutionResult(status="success", result=4),
            ),
            patch("seminar_demo.experiment._rule_evaluate", return_value="Correct"),
            patch.object(MedRaC, "execute_code", side_effect=AssertionError("unsafe bypass")) as unsafe,
        ):
            record = run_medrac_sample(
                row,
                "result = 4",
                generator,
                None,
                rag,
                use_llm_evaluation=False,
                timeout_seconds=5.0,
            )
        self.assertEqual(record["status"], "success")
        self.assertEqual(rag.queries, [("QUESTION_SENTINEL", 1)])
        extraction_user = generator.calls[0][0][0][1]
        code_user = generator.calls[1][0][0][1]
        self.assertIn("NOTE_SENTINEL", extraction_user)
        self.assertIn("QUESTION_SENTINEL", extraction_user)
        self.assertIn("QUESTION_SENTINEL", code_user)
        unsafe.assert_not_called()

    def test_unsafe_code_is_rejected_without_unrestricted_exec(self):
        rag = _FakeRag()
        generator = _FakeGenerator("import os\nresult = os.listdir('.')")
        with patch.object(
            MedRaC, "execute_code", side_effect=AssertionError("unsafe bypass")
        ) as unsafe:
            record = run_medrac_sample(
                _row(),
                "result = 4",
                generator,
                None,
                rag,
                use_llm_evaluation=False,
                timeout_seconds=1.0,
            )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["execution"]["status"], "rejected")
        self.assertEqual(record["execution"]["validation_status"], "rejected")
        unsafe.assert_not_called()

    def test_stepwise_derivation_requires_complete_success(self):
        evaluation = {
            "formula": {"result": "Correct"},
            "extracted_values": {"result": "Incorrect"},
            "calculation": {"result": "Incorrect"},
            "answer": {"result": "Correct"},
        }
        result = derive_stepwise_status(
            {"status": "success", "evaluation": evaluation}
        )
        self.assertTrue(result["complete"])
        self.assertFalse(result["strict_correct"])
        self.assertEqual(result["first_failed_step"], "extracted_values")
        not_run = derive_stepwise_status({"status": "not_run", "evaluation": None})
        self.assertIsNone(not_run["strict_correct"])
        self.assertIsNone(not_run["first_failed_step"])

    def test_experiment_download_removes_plain_branch(self):
        payload = medrac_only_download_payload(
            {
                "calculator_id": "2",
                "methods": {
                    "plain_cot": {"public_structured_output": {"step_by_step_thinking": "hidden"}},
                    "medrac_rag": {"final_answer": 4},
                },
            }
        )
        encoded = json.dumps(payload)
        self.assertNotIn("plain_cot", encoded)
        self.assertNotIn("step_by_step_thinking", encoded)
        self.assertEqual(payload["medrac_rag"]["final_answer"], 4)

    def test_artifact_discovery_loads_valid_and_skips_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "demo" / "valid"
            valid.mkdir(parents=True)
            (valid / "manifest.json").write_text(
                json.dumps({"run_id": "valid", "status": "completed"})
            )
            (valid / "samples.json").write_text(
                json.dumps([{"calculator_id": "2", "input": {"question": "q"}}])
            )
            incomplete = root / "experiments" / "incomplete"
            incomplete.mkdir(parents=True)
            runs, skipped = discover_replay_runs(root)
            self.assertEqual([run.path.name for run in runs], ["valid"])
            self.assertEqual(skipped, [incomplete])

    def test_configuration_validation_requires_token(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("seminar_demo.ui_adapter.load_project_env", return_value=False),
            patch("seminar_demo.ui_adapter.load_live_dataset", return_value=({}, pd.DataFrame())),
            patch("seminar_demo.ui_adapter.validate_canonical_index", return_value={}),
        ):
            status = validate_live_configuration()
        self.assertFalse(status.ready)
        self.assertFalse(status.token_configured)
        self.assertTrue(status.dataset_ready)
        self.assertTrue(status.index_ready)


if __name__ == "__main__":
    unittest.main()
