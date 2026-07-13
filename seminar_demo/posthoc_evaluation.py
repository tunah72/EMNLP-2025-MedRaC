from __future__ import annotations

import copy
import hashlib
import inspect
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from evaluator import LLM_Evaluator

from .experiment import (
    METHODS,
    aggregate_experiment,
    _stepwise_evaluate,
)
from .github_models import GitHubModelsLLM, require_github_token, validate_openai_catalog_model
from .pipeline import STEP_ORDER, _dependency_versions, _formula_map, _git_sha, _sha256, first_failed_step


PROMPT_VERSION = "paper-appendix-h1-released-code-v1"
REQUESTS_PER_EVALUATION = 4


@dataclass(frozen=True)
class PosthocEvaluationConfig:
    source_artifact: Path
    output_dir: Path | None
    dataset_path: Path
    formula_path: Path
    evaluator_model: str
    evaluator_temperature: float
    seed: int
    batch_indices: Sequence[int] | None
    methods: Sequence[str]
    dry_run: bool
    command: Sequence[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prompt_sha256() -> str:
    source = "\n".join(
        (
            inspect.getsource(LLM_Evaluator._gen_eval_prompt),
            inspect.getsource(LLM_Evaluator._parse_ans_prompt),
        )
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_evaluation_plan(
    samples: Sequence[dict[str, Any]],
    *,
    batch_indices: Sequence[int] | None = None,
    methods: Sequence[str] = METHODS,
) -> dict[str, Any]:
    invalid_methods = sorted(set(methods) - set(METHODS))
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {invalid_methods}")
    if not methods:
        raise ValueError("At least one method is required")

    by_index = {int(sample["dataframe_index"]): sample for sample in samples}
    if len(by_index) != len(samples):
        raise ValueError("Source artifact contains duplicate DataFrame indices")
    selected_indices = (
        [int(index) for index in batch_indices]
        if batch_indices is not None
        else list(by_index)
    )
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("Batch DataFrame indices must be unique")
    missing = sorted(set(selected_indices) - set(by_index))
    if missing:
        raise ValueError(f"Batch indices are absent from source artifact: {missing}")

    tasks = []
    already_complete = []
    unavailable = []
    for index in selected_indices:
        sample = by_index[index]
        for method in methods:
            record = sample["methods"][method]
            unit = {"dataframe_index": index, "method": method}
            if record.get("status") != "success":
                unavailable.append({**unit, "status": record.get("status")})
            elif record.get("stepwise_evaluation", {}).get("status") == "success":
                already_complete.append(unit)
            else:
                tasks.append(unit)
    return {
        "selected_indices": selected_indices,
        "methods": list(methods),
        "tasks": tasks,
        "already_complete": already_complete,
        "unavailable": unavailable,
        "pending_evaluation_units": len(tasks),
        "expected_chat_requests": len(tasks) * REQUESTS_PER_EVALUATION,
        "requests_per_evaluation_unit": REQUESTS_PER_EVALUATION,
    }


def _source_metadata(source: Path) -> dict[str, Any]:
    required = ["manifest.json", "samples.json", "metrics.json", "report.md"]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Source artifact is incomplete: {missing}")
    manifest = _load_json(source / "manifest.json")
    return {
        "path": str(source.resolve()),
        "run_id": manifest["run_id"],
        "manifest_sha256": _sha256(source / "manifest.json"),
        "samples_sha256": _sha256(source / "samples.json"),
    }


def _target_units(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"dataframe_index": int(sample["dataframe_index"]), "method": method}
        for sample in samples
        for method in METHODS
        if sample["methods"][method].get("status") == "success"
    ]


def _progress(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    targets = _target_units(samples)
    lookup = {int(sample["dataframe_index"]): sample for sample in samples}
    completed = failed = 0
    for unit in targets:
        record = lookup[unit["dataframe_index"]]["methods"][unit["method"]]
        status = record.get("stepwise_evaluation", {}).get("status")
        completed += int(status == "success")
        failed += int(status == "failed")
    return {
        "target_evaluation_units": len(targets),
        "completed_evaluation_units": completed,
        "failed_evaluation_units": failed,
        "pending_evaluation_units": len(targets) - completed,
        "expected_total_evaluator_requests": len(targets) * REQUESTS_PER_EVALUATION,
    }


def _initial_manifest(
    config: PosthocEvaluationConfig,
    source_metadata: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "run_id": output_dir.name,
        "status": "in_progress",
        "result_provenance": "post-hoc LLM evaluation of stored generation outputs",
        "git_sha": _git_sha(),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python_version": platform.python_version(),
        "dependency_versions": _dependency_versions(),
        "parent_artifact": source_metadata,
        "configuration": {
            "provider": "github-models",
            "paid_usage_fallback": False,
            "evaluation_only": True,
            "evaluator_model": config.evaluator_model,
            "evaluator_temperature": config.evaluator_temperature,
            "seed": config.seed,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": _prompt_sha256(),
            "requests_per_evaluation_unit": REQUESTS_PER_EVALUATION,
            "target_methods": list(METHODS),
        },
        "batches": [],
        "progress": {},
        "output_paths": {
            "manifest": str((output_dir / "manifest.json").resolve()),
            "samples": str((output_dir / "samples.json").resolve()),
            "metrics": str((output_dir / "metrics.json").resolve()),
            "report": str((output_dir / "report.md").resolve()),
        },
    }


def _validate_resume(
    manifest: dict[str, Any],
    config: PosthocEvaluationConfig,
    source_metadata: dict[str, Any],
) -> None:
    if manifest.get("parent_artifact") != source_metadata:
        raise ValueError("Source artifact changed since the derived artifact was created")
    expected = manifest["configuration"]
    observed = {
        "evaluator_model": config.evaluator_model,
        "evaluator_temperature": config.evaluator_temperature,
        "seed": config.seed,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": _prompt_sha256(),
    }
    for key, value in observed.items():
        if expected.get(key) != value:
            raise ValueError(
                f"Cannot resume with changed {key}: {expected.get(key)!r} != {value!r}"
            )


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    samples: Sequence[dict[str, Any]],
) -> None:
    progress = manifest["progress"]
    lines = [
        f"# Post-hoc Step-wise Evaluation — {manifest['run_id']}",
        "",
        f"- Status: **{manifest['status']}**",
        f"- Parent artifact: `{manifest['parent_artifact']['run_id']}`",
        f"- Evaluator: `{manifest['configuration']['evaluator_model']}`",
        f"- Prompt: `{manifest['configuration']['prompt_version']}`",
        f"- Progress: `{progress['completed_evaluation_units']}/{progress['target_evaluation_units']}` method/sample units",
        "",
        "This derived artifact evaluates stored outputs only. It does not regenerate answers, rebuild embeddings, retrieve formulas, or execute generated code.",
        "",
        "## Reasoning metrics",
        "",
        "| Method | F | E | C | A | Strict | Masked failures / A-correct |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = metrics[method]
        step_values = [item["step_accuracy"][step]["value"] for step in STEP_ORDER]
        strict = item["strict_all_steps_accuracy"]["value"]
        masked = item["masked_reasoning_failures"]
        formatted = ["N/A" if value is None else f"{value:.3f}" for value in step_values]
        strict_text = "N/A" if strict is None else f"{strict:.3f}"
        lines.append(
            f"| `{method}` | {' | '.join(formatted)} | {strict_text} | "
            f"{masked['count']}/{masked['answer_correct_denominator']} |"
        )

    lines.extend(
        [
            "",
            "## Conditional Correctness",
            "",
            "| Method | CC-F | CC-E | CC-C | CC-A |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        conditional = metrics[method]["conditional_correctness"]
        values = [conditional[step]["value"] for step in STEP_ORDER]
        rendered = ["N/A" if value is None else f"{value:.3f}" for value in values]
        lines.append(f"| `{method}` | {' | '.join(rendered)} |")

    lines.extend(
        [
            "",
            "## First Error Attribution Rate",
            "",
            "FE is conditioned on strict failure (`not kappa`); all-correct cases are excluded from its denominator.",
            "",
            "| Method | Failed denominator | FE-F | FE-E | FE-C | FE-A |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        attribution = metrics[method]["first_error_attribution"]
        values = [attribution["rates"][step]["value"] for step in STEP_ORDER]
        rendered = ["N/A" if value is None else f"{value:.3f}" for value in values]
        lines.append(
            f"| `{method}` | {attribution['denominator']} | {' | '.join(rendered)} |"
        )

    lines.extend(["", "## Per-sample step verdicts", "", "| Index | Method | F | E | C | A | First failed |", "|---:|---|---|---|---|---|---|"])
    for sample in samples:
        for method in METHODS:
            record = sample["methods"][method]
            evaluation = record.get("stepwise_evaluation", {}).get("evaluation")
            if not isinstance(evaluation, dict):
                verdicts = ["N/A"] * len(STEP_ORDER)
            else:
                verdicts = [evaluation.get(step, {}).get("result", "N/A") for step in STEP_ORDER]
            lines.append(
                f"| {sample['dataframe_index']} | `{method}` | {' | '.join(verdicts)} | "
                f"{record.get('first_failed_step')} |"
            )
    if manifest["status"] != "completed":
        lines.extend(
            [
                "",
                "The evaluation is incomplete. Aggregate values use only complete F/E/C/A verdicts and must not be presented as final results yet.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _checkpoint(
    output_dir: Path,
    manifest: dict[str, Any],
    samples: Sequence[dict[str, Any]],
) -> None:
    progress = _progress(samples)
    manifest["progress"] = progress
    manifest["updated_at"] = _utc_now()
    if progress["completed_evaluation_units"] == progress["target_evaluation_units"]:
        manifest["status"] = "completed"
    elif progress["failed_evaluation_units"]:
        manifest["status"] = "in_progress_with_failures"
    else:
        manifest["status"] = "in_progress"
    metrics = aggregate_experiment(samples)
    _atomic_json(output_dir / "manifest.json", manifest)
    _atomic_json(output_dir / "samples.json", samples)
    _atomic_json(output_dir / "metrics.json", metrics)
    _write_report(output_dir / "report.md", manifest, metrics, samples)


def run_posthoc_evaluation(config: PosthocEvaluationConfig) -> Path | dict[str, Any]:
    validate_openai_catalog_model(config.evaluator_model)
    source = config.source_artifact.resolve()
    source_metadata = _source_metadata(source)
    source_samples = _load_json(source / "samples.json")

    if config.dry_run:
        return {
            "source_artifact": source_metadata,
            "evaluator_model": config.evaluator_model,
            "evaluator_temperature": config.evaluator_temperature,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": _prompt_sha256(),
            "plan": build_evaluation_plan(
                source_samples,
                batch_indices=config.batch_indices,
                methods=config.methods,
            ),
            "api_requests_made": 0,
        }
    if config.output_dir is None:
        raise ValueError("output_dir is required unless --dry-run is used")

    output_dir = config.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and not manifest_path.is_file():
        raise FileExistsError(
            f"Output directory exists but is not a resumable artifact: {output_dir}"
        )
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        _validate_resume(manifest, config, source_metadata)
        samples = _load_json(output_dir / "samples.json")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        samples = copy.deepcopy(source_samples)
        manifest = _initial_manifest(config, source_metadata, output_dir)
        _checkpoint(output_dir, manifest, samples)

    plan = build_evaluation_plan(
        samples,
        batch_indices=config.batch_indices,
        methods=config.methods,
    )
    batch = {
        "started_at": _utc_now(),
        "command": list(config.command),
        "selected_indices": plan["selected_indices"],
        "methods": plan["methods"],
        "planned_evaluation_units": plan["pending_evaluation_units"],
        "expected_chat_requests": plan["expected_chat_requests"],
        "completed_evaluation_units": 0,
        "chat_attempts": 0,
        "status": "running",
    }
    manifest["batches"].append(batch)
    _checkpoint(output_dir, manifest, samples)
    if not plan["tasks"]:
        batch["status"] = "no_pending_work"
        batch["completed_at"] = _utc_now()
        _checkpoint(output_dir, manifest, samples)
        return output_dir

    require_github_token()
    evaluator_model = GitHubModelsLLM(
        config.evaluator_model,
        temperature=config.evaluator_temperature,
        seed=config.seed,
    )
    frame = pd.read_csv(config.dataset_path, encoding="utf-8")
    formulas = _formula_map(config.formula_path)
    samples_by_index = {int(sample["dataframe_index"]): sample for sample in samples}

    for task in plan["tasks"]:
        index = task["dataframe_index"]
        method = task["method"]
        sample = samples_by_index[index]
        row = frame.iloc[index]
        if str(row["Calculator ID"]) != str(sample["calculator_id"]):
            raise ValueError(f"Dataset identity mismatch at DataFrame index {index}")
        record = sample["methods"][method]
        response = record.get("public_structured_output")
        if not isinstance(response, dict):
            raise ValueError(f"Stored structured output is unavailable for {index}/{method}")
        reference_formula = formulas[str(row["Calculator ID"])] if method == "medrac_rag" else None
        before_requests = evaluator_model.requests_made
        started_at = _utc_now()
        try:
            evaluation, input_tokens, output_tokens, attempts = _stepwise_evaluate(
                response,
                row,
                evaluator_model,
                formula=reference_formula,
            )
            record["stepwise_evaluation"] = {
                "status": "success",
                "evaluation": evaluation,
            }
            record["first_failed_step"] = first_failed_step(evaluation)
            record["token_usage"]["evaluation_input"] += input_tokens
            record["token_usage"]["evaluation_output"] += output_tokens
            record["request_counts"]["chat_attempts"] += attempts
            batch["completed_evaluation_units"] += 1
            batch["chat_attempts"] += attempts
            run = {
                "started_at": started_at,
                "completed_at": _utc_now(),
                "status": "success",
                "evaluator_model": config.evaluator_model,
                "prompt_version": PROMPT_VERSION,
                "chat_attempts": attempts,
            }
        except Exception as exc:
            attempts = evaluator_model.requests_made - before_requests + 1
            record["stepwise_evaluation"] = {
                "status": "failed",
                "evaluation": None,
                "error": {"type": type(exc).__name__, "message": str(exc)[:500]},
            }
            record["request_counts"]["chat_attempts"] += attempts
            batch["chat_attempts"] += attempts
            run = {
                "started_at": started_at,
                "completed_at": _utc_now(),
                "status": "failed",
                "evaluator_model": config.evaluator_model,
                "prompt_version": PROMPT_VERSION,
                "chat_attempts": attempts,
                "error": {"type": type(exc).__name__, "message": str(exc)[:500]},
            }
            record.setdefault("posthoc_evaluation", {"runs": []})["runs"].append(run)
            batch["status"] = "stopped_on_failure"
            batch["completed_at"] = _utc_now()
            _checkpoint(output_dir, manifest, samples)
            raise RuntimeError(
                f"Post-hoc evaluation stopped at index {index}/{method}: {exc}"
            ) from exc
        record.setdefault("posthoc_evaluation", {"runs": []})["runs"].append(run)
        _checkpoint(output_dir, manifest, samples)

    batch["status"] = "completed"
    batch["completed_at"] = _utc_now()
    _checkpoint(output_dir, manifest, samples)
    return output_dir
