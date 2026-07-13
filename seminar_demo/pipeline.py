from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from evaluator import LLM_Evaluator, RegEvaluator
from method.medRaC import MedRaC
from method.rag import RAG
from model import APIModel
from schema.schemas import FormulaAndValues, Values

from .canonical_index import canonical_documents, validate_canonical_index, CanonicalFormulaIndex
from .data import ExclusionPolicy, apply_exclusion_policy, select_samples
from .github_models import (
    GitHubModelsLLM,
    require_github_token,
    validate_openai_catalog_model,
)
from .safe_execution import execute_safely


INDEX_HASHES = {
    "index.faiss": "ef8b80ab8566e28b73a78536ec373f767a9bbbb8cde4465c310ad8c080328a9a",
    "index.pkl": "fd4de9d5ba416f2004d6c35a6408ed3c12abdb0903c7dbf36b6f44469179dbd8",
}
STEP_ORDER = ("formula", "extracted_values", "calculation", "answer")


@dataclass(frozen=True)
class DemoConfig:
    dataset_path: Path
    formula_path: Path
    index_dir: Path
    output_root: Path
    row_indices: Sequence[int] | None
    calculator_ids: Sequence[str] | None
    provider: str
    generator_model: str
    evaluator_model: str
    embedding_model: str
    temperature: float
    seed: int
    use_rag: bool
    use_llm_evaluation: bool
    dry_run: bool
    trust_checked_in_index: bool
    timeout_seconds: float = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_index_artifacts(index_dir: Path) -> dict[str, Any]:
    observed = {}
    for name, expected in INDEX_HASHES.items():
        path = index_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Hash mismatch for checked-in index artifact: {path}")
        observed[name] = {"path": str(path), "sha256": actual}
    return observed


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _dependency_versions() -> dict[str, str]:
    packages = (
        "faiss-cpu",
        "langchain",
        "langchain-community",
        "langchain-openai",
        "numpy",
        "openai",
        "pandas",
        "pydantic",
        "tiktoken",
    )
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _formula_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["Calculator ID"]): item["Formula"] for item in payload}


def _normalise_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            return parsed if isinstance(parsed, dict) else {"raw": payload}
        except json.JSONDecodeError:
            return {"raw": payload}
    return {"raw": str(payload)}


def _code_prompts(
    formulas: Sequence[str], values: Sequence[Any], questions: Sequence[str]
) -> list[tuple[str, str]]:
    system = (
        "Write a short Python calculation using only numeric constants, basic "
        "arithmetic, comparisons, conditionals, abs, min, max, pow, round, and "
        "approved math functions. Do not import modules, access files or the "
        "network, call input, use reflection, or print. Assign the final answer "
        "to a variable named result. Return only fenced Python code."
    )
    return [
        (
            system,
            "Formula:\n"
            f"{formula}\n\nExtracted values:\n"
            f"{json.dumps(value, ensure_ascii=False, default=str)}\n\n"
            f"Question:\n{question}\n\nAssign the final answer to result.",
        )
        for formula, value, question in zip(formulas, values, questions)
    ]


def first_failed_step(evaluation: dict[str, Any] | None) -> str | None:
    if not evaluation:
        return None
    for step in STEP_ORDER:
        value = evaluation.get(step)
        if isinstance(value, dict) and value.get("result") == "Incorrect":
            return step
    return None


def _base_sample(row: pd.Series, canonical_formula: str) -> dict[str, Any]:
    return {
        "dataframe_index": int(row["DataFrame Index"]),
        "row_number": int(row["Row Number"]),
        "calculator_id": str(row["Calculator ID"]),
        "calculator_name": row["Calculator Name"],
        "category": row["Category"],
        "output_type": row["Output Type"],
        "input": {
            "patient_note": row["Patient Note"],
            "question": row["Question"],
        },
        "canonical_reference_formula": canonical_formula,
        "retrieval": {"status": "not_run", "formula": None, "score": None},
        "extraction": {"status": "not_run", "extracted_values": None},
        "code_generation": {"status": "not_run", "code": None},
        "execution": {
            "status": "not_run",
            "result": None,
            "error": None,
            "execution_mode": "safe_ast_child_process",
        },
        "final_answer": None,
        "rule_evaluation": {"status": "not_run", "result": None},
        "stepwise_evaluation": {"status": "not_run", "evaluation": None},
        "first_failed_step": None,
        "token_usage": {
            "retrieval_embedding_input": 0,
            "extraction_input": 0,
            "extraction_output": 0,
            "code_input": 0,
            "code_output": 0,
        },
    }


def _write_report(path: Path, manifest: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    lines = [
        f"# MedRaC Pipeline Demo — {manifest['run_id']}",
        "",
        f"- Status: **{manifest['status']}**",
        f"- Dry run: `{manifest['configuration']['dry_run']}`",
        f"- Generator: `{manifest['configuration']['generator_model']}`",
        f"- Evaluator: `{manifest['configuration']['evaluator_model']}`",
        f"- RAG enabled: `{manifest['configuration']['use_rag']}`",
        f"- Execution mode: `safe_ast_child_process`",
        "",
        "No provider-hidden reasoning is stored. Dry-run stages are explicitly marked `not_run`.",
        "",
    ]
    for sample in samples:
        lines.extend(
            [
                f"## Row {sample['row_number']} — {sample['calculator_name']}",
                "",
                f"- DataFrame index: `{sample['dataframe_index']}`",
                f"- Calculator ID/category: `{sample['calculator_id']}` / `{sample['category']}`",
                f"- Retrieval: `{sample['retrieval']['status']}`; score: `{sample['retrieval']['score']}`",
                f"- Extraction: `{sample['extraction']['status']}`",
                f"- Code generation: `{sample['code_generation']['status']}`",
                f"- Execution: `{sample['execution']['status']}`",
                f"- Final answer: `{sample['final_answer']}`",
                f"- Rule evaluation: `{sample['rule_evaluation']['result']}`",
                f"- First failed step: `{sample['first_failed_step']}`",
                "",
                "### Question",
                "",
                sample["input"]["question"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_demo(config: DemoConfig, policy: ExclusionPolicy) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    frame = pd.read_csv(config.dataset_path, encoding="utf-8")
    retained, excluded = apply_exclusion_policy(frame, policy)
    selected = select_samples(
        frame,
        policy,
        row_indices=config.row_indices,
        calculator_ids=config.calculator_ids,
    )
    formulas_by_id = _formula_map(config.formula_path)
    if config.provider == "github-models":
        validate_openai_catalog_model(config.generator_model)
        validate_openai_catalog_model(config.evaluator_model)
        validate_openai_catalog_model(config.embedding_model, embedding=True)
        canonical_documents(config.formula_path)
        try:
            index_artifacts: dict[str, Any] = {
                "status": "ready",
                "manifest": validate_canonical_index(
                    config.index_dir, formula_path=config.formula_path
                ),
            }
        except FileNotFoundError as exc:
            if not config.dry_run or not config.use_rag:
                if config.use_rag:
                    raise
            index_artifacts = {"status": "missing", "reason": str(exc)}
    elif config.provider == "openai":
        index_artifacts = {
            "status": "ready",
            "artifacts": validate_index_artifacts(config.index_dir),
        }
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ") + ("-dry-run" if config.dry_run else "")
    output_dir = config.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    samples = [
        _base_sample(row, formulas_by_id[str(row["Calculator ID"])])
        for _, row in selected.iterrows()
    ]

    if not config.dry_run:
        if config.provider == "openai" and config.use_rag and not config.trust_checked_in_index:
            raise ValueError("Live RAG requires --trust-checked-in-index")
        if config.provider == "github-models":
            require_github_token()
            generator: Any = GitHubModelsLLM(
                config.generator_model,
                temperature=config.temperature,
                seed=config.seed,
            )
            rag: Any = CanonicalFormulaIndex(
                config.index_dir,
                embedding_model=config.embedding_model,
            ) if config.use_rag else None
        else:
            if not os.getenv("OPENAI_API_KEY"):
                raise EnvironmentError("OPENAI_API_KEY is required for the OpenAI provider")
            generator = APIModel(config.generator_model, temperature=config.temperature)
            rag = RAG(
                embedding_model=config.embedding_model,
                embeddings_dir=str(config.index_dir),
            ) if config.use_rag else None

        formulas: list[str] = []
        if rag is not None:
            for sample in samples:
                formula, score = rag.retrieve(sample["input"]["question"], k=1)[0]
                formulas.append(formula)
                sample["retrieval"] = {
                    "status": "success",
                    "formula": formula,
                    "score": score,
                }
                sample["token_usage"]["retrieval_embedding_input"] = int(
                    getattr(rag, "last_embedding_input_tokens", 0)
                )
            prompts = MedRaC._gen_extracted_values(
                None,
                [sample["input"]["patient_note"] for sample in samples],
                formulas,
                [sample["input"]["question"] for sample in samples],
            )
            generations = generator.generate(prompts, schema=Values)
        else:
            prompts = MedRaC._gen_formula_and_extracted_values(
                None,
                [sample["input"]["patient_note"] for sample in samples],
                [sample["input"]["question"] for sample in samples],
            )
            generations = generator.generate(prompts, schema=FormulaAndValues)

        extracted_values: list[Any] = []
        for sample, (payload, input_tokens, output_tokens) in zip(samples, generations):
            parsed = _normalise_payload(payload)
            if rag is None:
                formula = parsed.get("formula")
                formulas.append(str(formula))
                sample["retrieval"] = {
                    "status": "generated_without_rag",
                    "formula": formula,
                    "score": None,
                }
            values = parsed.get("extracted_values")
            extracted_values.append(values)
            sample["extraction"] = {
                "status": "success" if values is not None else "parse_error",
                "extracted_values": values,
            }
            sample["token_usage"]["extraction_input"] = input_tokens
            sample["token_usage"]["extraction_output"] = output_tokens

        code_generations = generator.generate(
            _code_prompts(
                formulas,
                extracted_values,
                [sample["input"]["question"] for sample in samples],
            )
        )
        for sample, (code_payload, input_tokens, output_tokens) in zip(samples, code_generations):
            code = code_payload if isinstance(code_payload, str) else json.dumps(code_payload)
            sample["code_generation"] = {"status": "success", "code": code}
            sample["token_usage"]["code_input"] = input_tokens
            sample["token_usage"]["code_output"] = output_tokens
            execution = execute_safely(code, timeout_seconds=config.timeout_seconds)
            sample["execution"] = execution.to_dict()
            sample["final_answer"] = execution.result if execution.status == "success" else None

        responses = [
            {
                "formula": sample["retrieval"]["formula"],
                "extracted_values": sample["extraction"]["extracted_values"],
                "calculation": sample["code_generation"]["code"],
                "answer": (
                    str(sample["final_answer"])
                    if sample["final_answer"] is not None
                    else None
                ),
            }
            for sample in samples
        ]
        calids = [sample["calculator_id"] for sample in samples]
        _, rule_results = RegEvaluator.check_correctness(
            responses=responses,
            ground_truths=selected["Ground Truth Answer"].tolist(),
            calids=calids,
            upper_limits=selected["Upper Limit"].tolist(),
            lower_limits=selected["Lower Limit"].tolist(),
        )
        for sample, result in zip(samples, rule_results):
            sample["rule_evaluation"] = {"status": "success", "result": result}

        if config.use_llm_evaluation:
            evaluator_model = (
                GitHubModelsLLM(config.evaluator_model, temperature=0.0, seed=config.seed)
                if config.provider == "github-models"
                else APIModel(config.evaluator_model, temperature=0.0)
            )
            evaluator = LLM_Evaluator(evaluator_model)
            _, evaluations = evaluator.check_correctness(
                responses=responses,
                ground_truths=selected["Ground Truth Answer"].tolist(),
                ground_truth_explanations=selected["Ground Truth Explanation"].tolist(),
                relevant_entities=selected["Relevant Entities"].tolist(),
                calids=calids,
                upper_limits=selected["Upper Limit"].tolist(),
                lower_limits=selected["Lower Limit"].tolist(),
                formulas=[formulas_by_id[calculator_id] for calculator_id in calids],
            )
            for sample, evaluation in zip(samples, evaluations):
                sample["stepwise_evaluation"] = {
                    "status": "success",
                    "evaluation": evaluation,
                }
                sample["first_failed_step"] = first_failed_step(evaluation)

    ended = datetime.now(timezone.utc)
    manifest = {
        "run_id": run_id,
        "status": (
            "dry_run_preflight_missing_index"
            if config.dry_run and config.use_rag and index_artifacts["status"] == "missing"
            else "dry_run_validated" if config.dry_run else "completed"
        ),
        "result_provenance": "locally reproduced" if not config.dry_run else "inferred but unverified",
        "git_sha": _git_sha(),
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "python_version": platform.python_version(),
        "dependency_versions": _dependency_versions(),
        "configuration": {
            "provider": config.provider,
            "paid_usage_fallback": False,
            "dataset_path": str(config.dataset_path),
            "formula_path": str(config.formula_path),
            "index_dir": str(config.index_dir),
            "generator_model": config.generator_model,
            "evaluator_model": config.evaluator_model,
            "embedding_model": config.embedding_model,
            "temperature": config.temperature,
            "seed": config.seed,
            "use_rag": config.use_rag,
            "use_llm_evaluation": config.use_llm_evaluation,
            "dry_run": config.dry_run,
            "execution_mode": "safe_ast_child_process",
        },
        "dataset": {
            "source_rows": len(frame),
            "excluded_rows": len(excluded),
            "retained_rows": len(retained),
            "selected_dataframe_indices": [sample["dataframe_index"] for sample in samples],
            "selected_row_numbers": [sample["row_number"] for sample in samples],
            "selected_calculator_ids": [sample["calculator_id"] for sample in samples],
        },
        "index": index_artifacts,
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
        json.dumps(samples, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _write_report(output_dir / "report.md", manifest, samples)
    return output_dir
