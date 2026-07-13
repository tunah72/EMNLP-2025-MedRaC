from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .canonical_index import CanonicalFormulaIndex, validate_canonical_index
from .data import load_exclusion_policy
from .experiment import load_experiment_samples, run_medrac_sample
from .github_models import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    GitHubModelsLLM,
    load_project_env,
)
from .pipeline import STEP_ORDER, _dependency_versions, _formula_map


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class ReplayRun:
    path: Path
    artifact_type: str
    manifest: dict[str, Any]
    samples: list[dict[str, Any]]

    @property
    def label(self) -> str:
        status = self.manifest.get("status", "unknown")
        return f"{self.path.name} · {self.artifact_type} · {status}"


@dataclass(frozen=True)
class LiveConfigurationStatus:
    ready: bool
    token_configured: bool
    dataset_ready: bool
    index_ready: bool
    messages: tuple[str, ...]


def normalize_calculator_id(value: Any) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError("Calculator ID is required")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    result = str(value).strip()
    if result.endswith(".0") and result[:-2].isdigit():
        result = result[:-2]
    if not result:
        raise ValueError("Calculator ID is required")
    return result


def _first_field(
    sample: dict[str, Any], candidates: tuple[tuple[str, ...], ...]
) -> tuple[Any, str | None]:
    for path in candidates:
        value: Any = sample
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            if value is not None:
                return value, ".".join(path)
    return None, None


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def derive_stepwise_status(
    stepwise: dict[str, Any] | None, explicit_first_failed: Any = None
) -> dict[str, Any]:
    payload = stepwise or {}
    evaluation = payload.get("evaluation")
    complete = payload.get("status") == "success" and isinstance(evaluation, dict)
    if complete:
        complete = all(
            isinstance(evaluation.get(step), dict)
            and evaluation[step].get("result") in {"Correct", "Incorrect"}
            for step in STEP_ORDER
        )
    first_failed = None
    strict_correct: bool | None = None
    if complete:
        strict_correct = all(
            evaluation[step]["result"] == "Correct" for step in STEP_ORDER
        )
        if explicit_first_failed in STEP_ORDER:
            first_failed = explicit_first_failed
        else:
            first_failed = next(
                (
                    step
                    for step in STEP_ORDER
                    if evaluation[step]["result"] == "Incorrect"
                ),
                None,
            )
    return {
        "status": payload.get("status", "not_run"),
        "evaluation": evaluation,
        "complete": bool(complete),
        "strict_correct": strict_correct,
        "first_failed_step": first_failed,
    }


def normalize_artifact_sample(
    sample: dict[str, Any], manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    manifest = manifest or {}
    calculator_value, calculator_source = _first_field(
        sample,
        (("calculator_id",), ("Calculator ID",), ("input", "calculator_id")),
    )
    calculator_id = normalize_calculator_id(calculator_value)
    patient_note, patient_note_source = _first_field(
        sample,
        (
            ("input", "patient_note"),
            ("input", "Patient Note"),
            ("input", "clinical_note"),
            ("input", "Clinical Note"),
            ("patient_note",),
            ("Patient Note",),
            ("clinical_note",),
            ("Clinical Note",),
        ),
    )
    question, question_source = _first_field(
        sample,
        (("input", "question"), ("input", "Question"), ("question",), ("Question",)),
    )

    method_record = sample
    if isinstance(sample.get("methods"), dict):
        method_record = sample["methods"].get("medrac_rag") or {}
    elif "LLM Original Answer" in sample:
        method_record = _as_mapping(sample.get("LLM Original Answer"))

    retrieval = dict(method_record.get("retrieval") or {})
    if not retrieval and method_record.get("formula") is not None:
        retrieval = {
            "status": "legacy_structured_output",
            "formula": method_record.get("formula"),
            "score": None,
        }
    retrieval_query_source = "stored"
    if retrieval.get("query") is None and retrieval.get("status") == "success":
        retrieval["query"] = question
        retrieval_query_source = "derived_from_verified_runner"
    if retrieval.get("rank") is None and retrieval.get("status") == "success":
        retrieval["rank"] = 1

    extraction = dict(method_record.get("extraction") or {})
    if not extraction and method_record.get("extracted_values") is not None:
        extraction = {
            "status": "legacy_structured_output",
            "extracted_values": method_record.get("extracted_values"),
        }
    code_generation = dict(method_record.get("code_generation") or {})
    if not code_generation and method_record.get("calculation") is not None:
        code_generation = {
            "status": "legacy_structured_output",
            "code": method_record.get("calculation"),
        }
    execution = dict(method_record.get("execution") or {})
    if execution and "validation_status" not in execution:
        if execution.get("status") == "rejected":
            execution["validation_status"] = "rejected"
        elif execution.get("status") in {"success", "error", "timeout"}:
            execution["validation_status"] = "passed"
        else:
            execution["validation_status"] = "not_run"

    final_answer = method_record.get("final_answer")
    if final_answer is None:
        final_answer = method_record.get("answer")
    stepwise = derive_stepwise_status(
        method_record.get("stepwise_evaluation"),
        method_record.get("first_failed_step"),
    )
    configuration = manifest.get("configuration") or {}
    index_metadata = manifest.get("canonical_index") or manifest.get("index") or {}
    index_warning = None
    if "one_shot_finalized_explanation_formulas_embeddings" in str(
        configuration.get("index_dir", "")
    ):
        index_warning = (
            "Legacy trusted-pickle index: 776 indexed blocks out of 785 corpus blocks."
        )

    return {
        "dataframe_index": sample.get("dataframe_index", sample.get("DataFrame Index")),
        "row_number": sample.get("row_number", sample.get("Row Number")),
        "calculator_id": calculator_id,
        "calculator_name": sample.get("calculator_name", sample.get("Calculator Name")),
        "category": sample.get("category", sample.get("Category")),
        "family": sample.get("family"),
        "patient_note": patient_note,
        "patient_note_source": patient_note_source,
        "question": question,
        "question_source": question_source,
        "question_missing": not isinstance(question, str) or not question.strip(),
        "retrieval": retrieval,
        "retrieval_query_source": retrieval_query_source,
        "extraction": extraction,
        "code_generation": code_generation,
        "execution": execution,
        "final_answer": final_answer,
        "ground_truth_answer": sample.get(
            "ground_truth_answer", sample.get("Ground Truth Answer")
        ),
        "rule_evaluation": method_record.get("rule_evaluation") or {},
        "stepwise": stepwise,
        "token_usage": method_record.get("token_usage") or {},
        "request_counts": method_record.get("request_counts") or {},
        "runtime_error": method_record.get("error"),
        "method_status": method_record.get("status", "unknown"),
        "configuration": configuration,
        "index_metadata": index_metadata,
        "index_warning": index_warning,
        "result_provenance": manifest.get("result_provenance", "unknown"),
    }


def medrac_only_download_payload(sample: dict[str, Any]) -> dict[str, Any]:
    """Return a sample download without any Plain CoT branch."""
    if not isinstance(sample.get("methods"), dict):
        return sample
    metadata = {key: value for key, value in sample.items() if key != "methods"}
    metadata["method"] = "medrac_rag"
    metadata["medrac_rag"] = sample["methods"].get("medrac_rag")
    return metadata


def normalized_sample_markdown(
    normalized: dict[str, Any], *, run_id: str, status: str
) -> str:
    stepwise = normalized["stepwise"]
    return "\n".join(
        [
            f"# MedRaC RAG Sample — {run_id}",
            "",
            "> Research demonstration only. This is not a clinical application.",
            "",
            f"- Run status: **{status}**",
            f"- Provenance: **{normalized['result_provenance']}**",
            f"- Calculator ID: `{normalized['calculator_id']}`",
            f"- Calculator: {normalized.get('calculator_name') or 'Unavailable'}",
            f"- Final answer: `{normalized.get('final_answer')}`",
            f"- Ground truth: `{normalized.get('ground_truth_answer')}`",
            f"- Rule evaluation: `{normalized['rule_evaluation'].get('result')}`",
            f"- Step-wise status: `{stepwise['status']}`",
            f"- Strict all-steps correct: `{stepwise['strict_correct']}`",
            f"- First failed step: `{stepwise['first_failed_step']}`",
            "",
            "## Patient Note",
            "",
            str(normalized.get("patient_note") or "Unavailable"),
            "",
            "## Question",
            "",
            str(normalized.get("question") or "Unavailable"),
            "",
            "## Retrieved Formula",
            "",
            str(normalized["retrieval"].get("formula") or "Unavailable"),
            "",
            "## Extracted Values",
            "",
            "```json",
            json.dumps(
                normalized["extraction"].get("extracted_values"),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            "```",
            "",
            "No provider-hidden chain-of-thought is displayed.",
        ]
    ) + "\n"


def discover_replay_runs(
    artifacts_root: Path = ROOT / "artifacts",
) -> tuple[list[ReplayRun], list[Path]]:
    runs: list[ReplayRun] = []
    skipped: list[Path] = []
    for artifact_type in ("demo", "experiments"):
        parent = artifacts_root / artifact_type
        if not parent.is_dir():
            continue
        for path in sorted(parent.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            manifest_path = path / "manifest.json"
            samples_path = path / "samples.json"
            if not manifest_path.is_file() or not samples_path.is_file():
                skipped.append(path)
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                samples = json.loads(samples_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict) or not isinstance(samples, list):
                    raise ValueError("Artifact root types are invalid")
            except (OSError, ValueError, json.JSONDecodeError):
                skipped.append(path)
                continue
            runs.append(ReplayRun(path, artifact_type, manifest, samples))
    runs.sort(key=lambda run: run.manifest.get("started_at", run.path.name), reverse=True)
    return runs, skipped


def load_live_dataset(
    root: Path = ROOT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_csv(root / "data" / "test_data.csv", encoding="utf-8")
    policy = load_exclusion_policy(root / "config" / "paper_exclusions.json")
    return load_experiment_samples(
        root / "config" / "mini_experiment.json", frame, policy
    )


def validate_live_configuration(root: Path = ROOT) -> LiveConfigurationStatus:
    messages: list[str] = []
    load_project_env()
    token_configured = bool(os.getenv("GITHUB_TOKEN"))
    if not token_configured:
        messages.append("GITHUB_TOKEN is not configured.")
    try:
        load_live_dataset(root)
        dataset_ready = True
    except Exception as exc:
        dataset_ready = False
        messages.append(f"Dataset or deterministic manifest failed validation: {exc}")
    try:
        validate_canonical_index(
            root / "data" / "github_models_canonical_formulas_index",
            formula_path=root / "data" / "formula_new.json",
        )
        index_ready = True
    except Exception as exc:
        index_ready = False
        messages.append(f"Canonical index failed validation: {exc}")
    return LiveConfigurationStatus(
        ready=token_configured and dataset_ready and index_ready,
        token_configured=token_configured,
        dataset_ready=dataset_ready,
        index_ready=index_ready,
        messages=tuple(messages),
    )


def _git_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _live_markdown(manifest: dict[str, Any], sample: dict[str, Any]) -> str:
    stepwise = derive_stepwise_status(
        sample.get("stepwise_evaluation"), sample.get("first_failed_step")
    )
    return "\n".join(
        [
            f"# Streamlit MedRaC RAG Demo — {manifest['run_id']}",
            "",
            "> Research demonstration only. This is not a clinical application.",
            "",
            f"- Status: **{manifest['status']}**",
            f"- Provenance: **{manifest['result_provenance']}**",
            f"- Calculator ID: `{sample['calculator_id']}`",
            f"- Calculator: {sample['calculator_name']}",
            f"- Retrieval score: `{sample['retrieval'].get('score')}`",
            f"- Safe execution: `{sample['execution'].get('status')}`",
            f"- Final answer: `{sample.get('final_answer')}`",
            f"- Rule evaluation: `{sample['rule_evaluation'].get('result')}`",
            f"- Step-wise status: `{stepwise['status']}`",
            f"- Strict all-steps correct: `{stepwise['strict_correct']}`",
            f"- First failed step: `{stepwise['first_failed_step']}`",
            "",
            "## Patient Note",
            "",
            str(sample["input"]["patient_note"]),
            "",
            "## Question",
            "",
            str(sample["input"]["question"]),
            "",
            "No provider-hidden chain-of-thought is stored or displayed.",
        ]
    ) + "\n"


def run_live_dataset_sample(
    dataframe_index: int,
    *,
    use_llm_evaluation: bool,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    root: Path = ROOT,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> ReplayRun:
    status = validate_live_configuration(root)
    if not status.ready:
        raise EnvironmentError(" ".join(status.messages) or "Live configuration is incomplete")
    sample_manifest, selected = load_live_dataset(root)
    indices = selected["DataFrame Index"].astype(int).tolist()
    if int(dataframe_index) not in indices:
        raise ValueError("The selected row is not in the deterministic demo manifest")
    position = indices.index(int(dataframe_index))
    row = selected.iloc[position]
    declared = sample_manifest["samples"][position]
    formulas = _formula_map(root / "data" / "formula_new.json")
    calculator_id = normalize_calculator_id(row["Calculator ID"])

    generator = GitHubModelsLLM(DEFAULT_CHAT_MODEL, temperature=0.0, seed=42)
    evaluator_model = (
        GitHubModelsLLM(DEFAULT_CHAT_MODEL, temperature=0.0, seed=42)
        if use_llm_evaluation
        else None
    )
    rag = CanonicalFormulaIndex(
        root / "data" / "github_models_canonical_formulas_index",
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )
    started = datetime.now(timezone.utc)
    method_record = run_medrac_sample(
        row,
        formulas[calculator_id],
        generator,
        evaluator_model,
        rag,
        use_llm_evaluation=use_llm_evaluation,
        timeout_seconds=timeout_seconds,
        progress_callback=progress_callback,
    )
    ended = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ") + "-streamlit"
    output_dir = root / "artifacts" / "demo" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    sample = {
        "dataframe_index": int(row["DataFrame Index"]),
        "row_number": int(row["Row Number"]),
        "calculator_id": calculator_id,
        "calculator_name": row["Calculator Name"],
        "category": row["Category"],
        "output_type": row["Output Type"],
        "family": declared["family"],
        "selection_rationale": declared["rationale"],
        "input": {
            "patient_note": row["Patient Note"],
            "question": row["Question"],
        },
        "ground_truth_answer": row["Ground Truth Answer"],
        "canonical_reference_formula": formulas[calculator_id],
        "retrieval": method_record["retrieval"],
        "extraction": method_record["extraction"],
        "code_generation": method_record["code_generation"],
        "execution": method_record["execution"],
        "final_answer": method_record["final_answer"],
        "rule_evaluation": method_record["rule_evaluation"],
        "stepwise_evaluation": method_record["stepwise_evaluation"],
        "first_failed_step": method_record["first_failed_step"],
        "token_usage": method_record["token_usage"],
        "request_counts": method_record["request_counts"],
        "status": method_record["status"],
        "error": method_record["error"],
    }
    index_manifest = validate_canonical_index(
        root / "data" / "github_models_canonical_formulas_index",
        formula_path=root / "data" / "formula_new.json",
    )
    manifest = {
        "run_id": run_id,
        "status": "completed" if method_record["status"] == "success" else "completed_with_failure",
        "result_provenance": "locally reproduced Streamlit MedRaC run",
        "git_sha": _git_sha(root),
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python_version": platform.python_version(),
        "dependency_versions": _dependency_versions(),
        "configuration": {
            "provider": "github-models",
            "generator_model": DEFAULT_CHAT_MODEL,
            "evaluator_model": DEFAULT_CHAT_MODEL,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "temperature": 0.0,
            "seed": 42,
            "use_rag": True,
            "use_llm_evaluation": use_llm_evaluation,
            "execution_mode": "safe_ast_child_process",
            "timeout_seconds": timeout_seconds,
            "ui_mode": "dataset_live_demo",
        },
        "dataset": {
            "sample_manifest_path": str(root / "config" / "mini_experiment.json"),
            "selected_dataframe_indices": [int(row["DataFrame Index"])],
            "selected_row_numbers": [int(row["Row Number"])],
            "selected_calculator_ids": [calculator_id],
        },
        "canonical_index": index_manifest,
        "output_paths": {
            "manifest": str(output_dir / "manifest.json"),
            "samples": str(output_dir / "samples.json"),
            "report": str(output_dir / "report.md"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "samples.json").write_text(
        json.dumps([sample], ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        _live_markdown(manifest, sample), encoding="utf-8"
    )
    return ReplayRun(output_dir, "demo", manifest, [sample])
