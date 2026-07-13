from __future__ import annotations

import json
import platform
import random
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from evaluator import LLM_Evaluator, RegEvaluator
from method.medRaC import MedRaC
from method.plain import Plain
from schema.schemas import CoTOutput, Values

from .canonical_index import CanonicalFormulaIndex, validate_canonical_index
from .data import ExclusionPolicy, apply_exclusion_policy, select_samples
from .github_models import (
    GitHubModelsLLM,
    require_github_token,
    validate_openai_catalog_model,
)
from .pipeline import (
    STEP_ORDER,
    _code_prompts,
    _dependency_versions,
    _formula_map,
    _git_sha,
    _normalise_payload,
    _sha256,
    first_failed_step,
)
from .safe_execution import execute_safely


METHODS = ("plain_cot", "medrac_rag")


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_path: Path
    formula_path: Path
    exclusion_path: Path
    sample_manifest_path: Path
    index_dir: Path
    output_root: Path
    generator_model: str
    evaluator_model: str
    embedding_model: str
    temperature: float
    seed: int
    use_llm_evaluation: bool
    dry_run: bool
    limit: int | None
    timeout_seconds: float
    command: Sequence[str]


def load_experiment_samples(
    config_path: Path,
    frame: pd.DataFrame,
    policy: ExclusionPolicy,
    *,
    limit: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    declared = payload["samples"]
    expected = int(payload["expected_sample_count"])
    if len(declared) != expected:
        raise ValueError(f"Expected {expected} declared samples, found {len(declared)}")
    indices = [int(item["dataframe_index"]) for item in declared]
    if len(indices) != len(set(indices)):
        raise ValueError("Experiment DataFrame indices must be unique")
    selected = select_samples(frame, policy, row_indices=indices)
    for item, (_, row) in zip(declared, selected.iterrows()):
        observed = {
            "dataframe_index": int(row["DataFrame Index"]),
            "row_number": int(row["Row Number"]),
            "calculator_id": str(row["Calculator ID"]),
        }
        expected_identity = {
            "dataframe_index": int(item["dataframe_index"]),
            "row_number": int(item["row_number"]),
            "calculator_id": str(item["calculator_id"]),
        }
        if observed != expected_identity:
            raise ValueError(
                f"Experiment manifest identity mismatch: {expected_identity} != {observed}"
            )
        if item["family"] not in {"equation", "rule"}:
            raise ValueError(f"Unsupported calculator family: {item['family']}")
    if limit is not None:
        if limit < 1 or limit > expected:
            raise ValueError(f"limit must be between 1 and {expected}")
        declared = declared[:limit]
        selected = selected.iloc[:limit].reset_index(drop=True)
    effective = {**payload, "samples": declared, "effective_sample_count": len(declared)}
    return effective, selected


def _empty_method_record() -> dict[str, Any]:
    return {
        "status": "not_run",
        "error": None,
        "public_structured_output": None,
        "final_answer": None,
        "retrieval": {"status": "not_applicable", "formula": None, "score": None},
        "extraction": {"status": "not_applicable", "extracted_values": None},
        "code_generation": {"status": "not_applicable", "code": None},
        "execution": {
            "status": "not_applicable",
            "result": None,
            "error": None,
            "execution_mode": None,
        },
        "rule_evaluation": {"status": "not_run", "result": None},
        "stepwise_evaluation": {"status": "not_run", "evaluation": None},
        "first_failed_step": None,
        "token_usage": {
            "embedding_input": 0,
            "generation_input": 0,
            "generation_output": 0,
            "code_input": 0,
            "code_output": 0,
            "evaluation_input": 0,
            "evaluation_output": 0,
        },
        "request_counts": {"chat_attempts": 0, "embedding_attempts": 0},
    }


def _error_payload(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:500]}


def _rule_evaluate(response: Any, row: pd.Series) -> str:
    _, results = RegEvaluator.check_correctness(
        responses=[response],
        ground_truths=[row["Ground Truth Answer"]],
        calids=[str(row["Calculator ID"])],
        upper_limits=[row["Upper Limit"]],
        lower_limits=[row["Lower Limit"]],
    )
    return results[0]


def _stepwise_evaluate(
    response: dict[str, Any],
    row: pd.Series,
    evaluator_model: GitHubModelsLLM,
    *,
    formula: str | None,
) -> tuple[dict[str, Any], int, int, int]:
    before_requests = evaluator_model.requests_made
    evaluator = LLM_Evaluator(evaluator_model)
    try:
        _, evaluations = evaluator.check_correctness(
            responses=[response],
            ground_truths=[row["Ground Truth Answer"]],
            ground_truth_explanations=[row["Ground Truth Explanation"]],
            relevant_entities=[row["Relevant Entities"]],
            calids=[str(row["Calculator ID"])],
            upper_limits=[row["Upper Limit"]],
            lower_limits=[row["Lower Limit"]],
            formulas=[formula] if formula is not None else None,
        )
    except Exception:
        attempts = evaluator_model.requests_made - before_requests + 1
        raise
    attempts = evaluator_model.requests_made - before_requests
    return evaluations[0], evaluator.input_token_used, evaluator.output_token_used, attempts


def _run_plain_sample(
    row: pd.Series,
    generator: GitHubModelsLLM,
    evaluator_model: GitHubModelsLLM | None,
    *,
    use_llm_evaluation: bool,
) -> dict[str, Any]:
    record = _empty_method_record()
    record["retrieval"]["status"] = "not_applicable"
    record["extraction"]["status"] = "not_applicable"
    record["code_generation"]["status"] = "not_applicable"
    record["execution"]["status"] = "not_applicable"
    baseline = Plain(prompt_style="cot", llms=[generator], evaluators=[])
    prompts = baseline.prompt_fn(
        [str(row["Calculator ID"])], [row["Patient Note"]], [row["Question"]]
    )
    try:
        record["request_counts"]["chat_attempts"] += 1
        payload, input_tokens, output_tokens = generator.generate(
            prompts, schema=CoTOutput, show_progress=False
        )[0]
        parsed = _normalise_payload(payload)
        record["public_structured_output"] = parsed
        record["final_answer"] = parsed.get("answer")
        record["token_usage"]["generation_input"] = input_tokens
        record["token_usage"]["generation_output"] = output_tokens
        result = _rule_evaluate(parsed, row)
        record["rule_evaluation"] = {"status": "success", "result": result}
        record["status"] = "success"
        if use_llm_evaluation and evaluator_model is not None:
            evaluation, in_tokens, out_tokens, attempts = _stepwise_evaluate(
                parsed, row, evaluator_model, formula=None
            )
            record["request_counts"]["chat_attempts"] += attempts
            record["token_usage"]["evaluation_input"] = in_tokens
            record["token_usage"]["evaluation_output"] = out_tokens
            record["stepwise_evaluation"] = {
                "status": "success",
                "evaluation": evaluation,
            }
            record["first_failed_step"] = first_failed_step(evaluation)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = _error_payload(exc)
    return record


def _run_medrac_sample(
    row: pd.Series,
    canonical_formula: str,
    generator: GitHubModelsLLM,
    evaluator_model: GitHubModelsLLM | None,
    rag: CanonicalFormulaIndex,
    *,
    use_llm_evaluation: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    record = _empty_method_record()
    record["retrieval"]["status"] = "not_run"
    record["extraction"]["status"] = "not_run"
    record["code_generation"]["status"] = "not_run"
    record["execution"] = {
        "status": "not_run",
        "result": None,
        "error": None,
        "execution_mode": "safe_ast_child_process",
    }
    response: dict[str, Any] = {}
    try:
        record["request_counts"]["embedding_attempts"] += 1
        formula, score = rag.retrieve(row["Question"], k=1)[0]
        record["retrieval"] = {"status": "success", "formula": formula, "score": score}
        record["token_usage"]["embedding_input"] = rag.last_embedding_input_tokens

        prompts = MedRaC._gen_extracted_values(
            None, [row["Patient Note"]], [formula], [row["Question"]]
        )
        record["request_counts"]["chat_attempts"] += 1
        payload, input_tokens, output_tokens = generator.generate(
            prompts, schema=Values, show_progress=False
        )[0]
        parsed = _normalise_payload(payload)
        values = parsed.get("extracted_values")
        if values is None:
            raise ValueError("MedRaC extraction response has no extracted_values")
        record["extraction"] = {"status": "success", "extracted_values": values}
        record["token_usage"]["generation_input"] = input_tokens
        record["token_usage"]["generation_output"] = output_tokens

        record["request_counts"]["chat_attempts"] += 1
        code_payload, code_input, code_output = generator.generate(
            _code_prompts([formula], [values], [row["Question"]]),
            show_progress=False,
        )[0]
        code = code_payload if isinstance(code_payload, str) else json.dumps(code_payload)
        record["code_generation"] = {"status": "success", "code": code}
        record["token_usage"]["code_input"] = code_input
        record["token_usage"]["code_output"] = code_output
        execution = execute_safely(code, timeout_seconds=timeout_seconds)
        record["execution"] = execution.to_dict()
        if execution.status != "success":
            raise RuntimeError(f"Safe execution failed: {execution.error}")
        record["final_answer"] = execution.result
        response = {
            "formula": formula,
            "extracted_values": values,
            "calculation": code,
            "answer": str(execution.result),
        }
        record["public_structured_output"] = response
        result = _rule_evaluate(response, row)
        record["rule_evaluation"] = {"status": "success", "result": result}
        record["status"] = "success"

        if use_llm_evaluation and evaluator_model is not None:
            evaluation, in_tokens, out_tokens, attempts = _stepwise_evaluate(
                response, row, evaluator_model, formula=canonical_formula
            )
            record["request_counts"]["chat_attempts"] += attempts
            record["token_usage"]["evaluation_input"] = in_tokens
            record["token_usage"]["evaluation_output"] = out_tokens
            record["stepwise_evaluation"] = {
                "status": "success",
                "evaluation": evaluation,
            }
            record["first_failed_step"] = first_failed_step(evaluation)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = _error_payload(exc)
    return record


def _ratio(correct: int, denominator: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "denominator": denominator,
        "value": correct / denominator if denominator else None,
    }


def aggregate_method_metrics(samples: Sequence[dict[str, Any]], method: str) -> dict[str, Any]:
    records = [sample["methods"][method] for sample in samples]
    rule_results = [
        record["rule_evaluation"]["result"]
        for record in records
        if record["rule_evaluation"]["status"] == "success"
    ]
    complete_evaluations = []
    for record in records:
        stepwise = record["stepwise_evaluation"]
        evaluation = stepwise.get("evaluation")
        if stepwise.get("status") != "success" or not isinstance(evaluation, dict):
            continue
        if all(
            isinstance(evaluation.get(step), dict)
            and evaluation[step].get("result") in {"Correct", "Incorrect"}
            for step in STEP_ORDER
        ):
            complete_evaluations.append(evaluation)

    step_accuracy = {
        step: _ratio(
            sum(evaluation[step]["result"] == "Correct" for evaluation in complete_evaluations),
            len(complete_evaluations),
        )
        for step in STEP_ORDER
    }
    strict_correct = sum(
        all(evaluation[step]["result"] == "Correct" for step in STEP_ORDER)
        for evaluation in complete_evaluations
    )
    conditional: dict[str, dict[str, Any]] = {}
    for index, step in enumerate(STEP_ORDER):
        eligible = [
            evaluation
            for evaluation in complete_evaluations
            if all(
                evaluation[previous]["result"] == "Correct"
                for previous in STEP_ORDER[:index]
            )
        ]
        conditional[step] = _ratio(
            sum(evaluation[step]["result"] == "Correct" for evaluation in eligible),
            len(eligible),
        )
    first_errors = {step: 0 for step in STEP_ORDER}
    all_correct = 0
    answer_correct = 0
    masked_reasoning_failures = 0
    for evaluation in complete_evaluations:
        failed = first_failed_step(evaluation)
        is_answer_correct = evaluation["answer"]["result"] == "Correct"
        answer_correct += int(is_answer_correct)
        if failed is None:
            all_correct += 1
        else:
            first_errors[failed] += 1
            masked_reasoning_failures += int(is_answer_correct)

    strict_failures = len(complete_evaluations) - all_correct

    token_keys = records[0]["token_usage"].keys() if records else []
    request_keys = records[0]["request_counts"].keys() if records else []
    return {
        "selected_samples": len(records),
        "completed_samples": sum(record["status"] == "success" for record in records),
        "failed_samples": sum(record["status"] == "failed" for record in records),
        "skipped_samples": sum(record["status"] == "skipped" for record in records),
        "rule_based_final_answer_accuracy": _ratio(
            sum(result == "Correct" for result in rule_results), len(rule_results)
        ),
        "stepwise_complete_samples": len(complete_evaluations),
        "step_accuracy": step_accuracy,
        "strict_all_steps_accuracy": _ratio(strict_correct, len(complete_evaluations)),
        "conditional_correctness": conditional,
        "first_error_attribution": {
            "counts": first_errors,
            "rates": {
                step: _ratio(count, strict_failures)
                for step, count in first_errors.items()
            },
            "all_correct": all_correct,
            "denominator": strict_failures,
            "complete_denominator": len(complete_evaluations),
        },
        "masked_reasoning_failures": {
            "count": masked_reasoning_failures,
            "answer_correct_denominator": answer_correct,
            "value": (
                masked_reasoning_failures / answer_correct
                if answer_correct
                else None
            ),
        },
        "token_usage": {
            key: sum(int(record["token_usage"].get(key, 0)) for record in records)
            for key in token_keys
        },
        "request_counts": {
            key: sum(int(record["request_counts"].get(key, 0)) for record in records)
            for key in request_keys
        },
    }


def aggregate_experiment(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = {method: aggregate_method_metrics(samples, method) for method in METHODS}
    paired = []
    for sample in samples:
        plain = sample["methods"]["plain_cot"]["rule_evaluation"]
        medrac = sample["methods"]["medrac_rag"]["rule_evaluation"]
        if plain["status"] == "success" and medrac["status"] == "success":
            paired.append((plain["result"], medrac["result"]))
    metrics["paired_comparison"] = {
        "eligible_samples": len(paired),
        "excluded_due_to_method_failure": len(samples) - len(paired),
        "plain_cot_accuracy": _ratio(
            sum(plain == "Correct" for plain, _ in paired), len(paired)
        ),
        "medrac_rag_accuracy": _ratio(
            sum(medrac == "Correct" for _, medrac in paired), len(paired)
        ),
        "outcomes": {
            "both_correct": sum(p == "Correct" and m == "Correct" for p, m in paired),
            "plain_only_correct": sum(p == "Correct" and m != "Correct" for p, m in paired),
            "medrac_only_correct": sum(p != "Correct" and m == "Correct" for p, m in paired),
            "both_incorrect": sum(p != "Correct" and m != "Correct" for p, m in paired),
        },
    }
    return metrics


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    samples: Sequence[dict[str, Any]],
) -> None:
    lines = [
        f"# Mini Experiment — {manifest['run_id']}",
        "",
        f"- Status: **{manifest['status']}**",
        f"- Provenance: **{manifest['result_provenance']}**",
        f"- Samples: `{len(samples)}` fixed rows",
        f"- Generator/evaluator: `{manifest['configuration']['generator_model']}` / "
        f"`{manifest['configuration']['evaluator_model']}`",
        f"- LLM step-wise evaluation: `{manifest['configuration']['use_llm_evaluation']}`",
        "",
        "This is a local seminar-scale comparison, not a reproduction of the paper's reported results and not a statistical-significance claim.",
        "",
        "## Aggregate results",
        "",
        "| Method | Rule correct/denominator | Accuracy | Completed | Failed | Chat attempts | Embedding attempts |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = metrics[method]
        rule = item["rule_based_final_answer_accuracy"]
        value = "N/A" if rule["value"] is None else f"{rule['value']:.3f}"
        lines.append(
            f"| `{method}` | {rule['correct']}/{rule['denominator']} | {value} | "
            f"{item['completed_samples']} | {item['failed_samples']} | "
            f"{item['request_counts'].get('chat_attempts', 0)} | "
            f"{item['request_counts'].get('embedding_attempts', 0)} |"
        )
    paired = metrics["paired_comparison"]
    plain_paired = paired["plain_cot_accuracy"]
    medrac_paired = paired["medrac_rag_accuracy"]
    lines.extend(
        [
            "",
            "## Paired comparison",
            "",
            f"Only rows with successful rule evaluation for both methods are included: "
            f"`{paired['eligible_samples']}/{len(samples)}`. "
            f"Plain CoT: `{plain_paired['correct']}/{plain_paired['denominator']}`; "
            f"MedRaC: `{medrac_paired['correct']}/{medrac_paired['denominator']}`. "
            f"Excluded due to a method failure: `{paired['excluded_due_to_method_failure']}`.",
            "",
            f"Outcome counts: `{json.dumps(paired['outcomes'], sort_keys=True)}`.",
        ]
    )
    if not manifest["configuration"]["use_llm_evaluation"]:
        lines.extend(
            [
                "",
                "Step-wise F/E/C/A, strict correctness, Conditional Correctness, and First Error Attribution are unavailable because `--llm-evaluation` was disabled. They are stored as null/zero-denominator metrics, not fabricated as zero accuracy.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Step-wise reasoning results",
                "",
                "| Method | F | E | C | A | Strict | Masked failures / A-correct |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            item = metrics[method]
            step_values = [item["step_accuracy"][step]["value"] for step in STEP_ORDER]
            rendered = ["N/A" if value is None else f"{value:.3f}" for value in step_values]
            strict = item["strict_all_steps_accuracy"]["value"]
            strict_text = "N/A" if strict is None else f"{strict:.3f}"
            masked = item["masked_reasoning_failures"]
            lines.append(
                f"| `{method}` | {' | '.join(rendered)} | {strict_text} | "
                f"{masked['count']}/{masked['answer_correct_denominator']} |"
            )
        lines.extend(
            [
                "",
                "First Error Attribution Rate is conditioned on strict failure (`not kappa`). All-correct cases are excluded from its denominator; when there are no strict failures, FE rates are `null`, not zero.",
                "",
                "| Method | Strict failures | FE-F | FE-E | FE-C | FE-A |",
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
    lines.extend(["", "## Paired sample results", "", "| Index | ID | Family | Plain answer | Plain rule | MedRaC answer | MedRaC rule |", "|---:|---:|---|---|---|---|---|"])
    for sample in samples:
        plain = sample["methods"]["plain_cot"]
        medrac = sample["methods"]["medrac_rag"]
        lines.append(
            f"| {sample['dataframe_index']} | {sample['calculator_id']} | {sample['family']} | "
            f"{plain['final_answer']} | {plain['rule_evaluation']['result']} | "
            f"{medrac['final_answer']} | {medrac['rule_evaluation']['result']} |"
        )
    failures = [
        (sample, method, sample["methods"][method]["error"])
        for sample in samples
        for method in METHODS
        if sample["methods"][method]["status"] == "failed"
    ]
    lines.extend(["", "## Failures", ""])
    if failures:
        for sample, method, error in failures:
            lines.append(
                f"- Index `{sample['dataframe_index']}` / Calculator ID "
                f"`{sample['calculator_id']}` / `{method}`: "
                f"`{error['type']}` — {error['message']}"
            )
    else:
        lines.append("No failed method/sample executions.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_mini_experiment(config: ExperimentConfig, policy: ExclusionPolicy) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    validate_openai_catalog_model(config.generator_model)
    validate_openai_catalog_model(config.evaluator_model)
    validate_openai_catalog_model(config.embedding_model, embedding=True)
    frame = pd.read_csv(config.dataset_path, encoding="utf-8")
    retained, excluded = apply_exclusion_policy(frame, policy)
    sample_manifest, selected = load_experiment_samples(
        config.sample_manifest_path, frame, policy, limit=config.limit
    )
    formulas_by_id = _formula_map(config.formula_path)
    index_manifest = validate_canonical_index(
        config.index_dir, formula_path=config.formula_path
    )

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ") + (
        "-dry-run" if config.dry_run else ""
    )
    output_dir = config.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    samples: list[dict[str, Any]] = []
    for declared, (_, row) in zip(sample_manifest["samples"], selected.iterrows()):
        samples.append(
            {
                "dataframe_index": int(row["DataFrame Index"]),
                "row_number": int(row["Row Number"]),
                "calculator_id": str(row["Calculator ID"]),
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
                "methods": {method: _empty_method_record() for method in METHODS},
            }
        )

    if not config.dry_run:
        require_github_token()
        generator = GitHubModelsLLM(
            config.generator_model,
            temperature=config.temperature,
            seed=config.seed,
        )
        evaluator_model = (
            GitHubModelsLLM(config.evaluator_model, temperature=0.0, seed=config.seed)
            if config.use_llm_evaluation
            else None
        )
        rag = CanonicalFormulaIndex(
            config.index_dir, embedding_model=config.embedding_model
        )
        for sample, (_, row) in zip(samples, selected.iterrows()):
            sample["methods"]["plain_cot"] = _run_plain_sample(
                row,
                generator,
                evaluator_model,
                use_llm_evaluation=config.use_llm_evaluation,
            )
            sample["methods"]["medrac_rag"] = _run_medrac_sample(
                row,
                formulas_by_id[str(row["Calculator ID"])],
                generator,
                evaluator_model,
                rag,
                use_llm_evaluation=config.use_llm_evaluation,
                timeout_seconds=config.timeout_seconds,
            )

    metrics = aggregate_experiment(samples)
    ended = datetime.now(timezone.utc)
    any_failed = any(
        sample["methods"][method]["status"] == "failed"
        for sample in samples
        for method in METHODS
    )
    manifest = {
        "run_id": run_id,
        "status": (
            "dry_run_validated"
            if config.dry_run
            else "completed_with_failures" if any_failed else "completed"
        ),
        "result_provenance": (
            "inferred but unverified" if config.dry_run else "locally reproduced"
        ),
        "git_sha": _git_sha(),
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python_version": platform.python_version(),
        "dependency_versions": _dependency_versions(),
        "command": " ".join(shlex.quote(item) for item in config.command),
        "configuration": {
            "provider": "github-models",
            "paid_usage_fallback": False,
            "baseline": "Plain(cot)",
            "proposed_method": "MedRaC(use_rag=True)",
            "generator_model": config.generator_model,
            "evaluator_model": config.evaluator_model,
            "embedding_model": config.embedding_model,
            "temperature": config.temperature,
            "evaluator_temperature": 0.0,
            "seed": config.seed,
            "use_llm_evaluation": config.use_llm_evaluation,
            "execution_mode": "safe_ast_child_process",
            "timeout_seconds": config.timeout_seconds,
            "dry_run": config.dry_run,
            "limit": config.limit,
        },
        "dataset": {
            "source_rows": len(frame),
            "excluded_rows": len(excluded),
            "retained_rows": len(retained),
            "sample_manifest_path": str(config.sample_manifest_path),
            "sample_manifest_sha256": _sha256(config.sample_manifest_path),
            "selected_dataframe_indices": [s["dataframe_index"] for s in samples],
            "selected_row_numbers": [s["row_number"] for s in samples],
            "selected_calculator_ids": [s["calculator_id"] for s in samples],
            "families": {
                family: sum(s["family"] == family for s in samples)
                for family in ("equation", "rule")
            },
        },
        "canonical_index": index_manifest,
        "output_paths": {
            "manifest": str(output_dir / "manifest.json"),
            "samples": str(output_dir / "samples.json"),
            "metrics": str(output_dir / "metrics.json"),
            "report": str(output_dir / "report.md"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(output_dir / "report.md", manifest, metrics, samples)
    return output_dir
