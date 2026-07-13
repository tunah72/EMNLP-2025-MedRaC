#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seminar_demo.data import load_exclusion_policy
from seminar_demo.github_models import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL
from seminar_demo.pipeline import DemoConfig, run_demo


def _csv_ints(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc


def _csv_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one value")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic and transparent MedRaC seminar pipeline"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--row-indices", type=_csv_ints, help="Zero-based CSV DataFrame indices")
    selection.add_argument("--calculator-ids", type=_csv_strings, help="Calculator IDs; first retained row per ID")
    parser.add_argument(
        "--provider", choices=("github-models", "openai"), default="github-models"
    )
    parser.add_argument("--generator-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--evaluator-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="Override provider-specific index directory",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "demo")
    parser.add_argument("--no-rag", action="store_true", help="Generate a formula instead of retrieving it")
    parser.add_argument("--llm-evaluation", action="store_true", help="Enable F/E/C/A evaluation requests")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write planned artifacts without API calls")
    parser.add_argument("--trust-checked-in-index", action="store_true", help="Acknowledge trusted pickle loading for live RAG")
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.index_dir is not None:
        index_dir = args.index_dir
    elif args.provider == "github-models":
        index_dir = ROOT / "data" / "github_models_canonical_formulas_index"
    else:
        index_dir = ROOT / "data" / "one_shot_finalized_explanation_formulas_embeddings"
    config = DemoConfig(
        dataset_path=ROOT / "data" / "test_data.csv",
        formula_path=ROOT / "data" / "formula_new.json",
        index_dir=index_dir,
        output_root=args.output_dir,
        row_indices=args.row_indices,
        calculator_ids=args.calculator_ids,
        provider=args.provider,
        generator_model=args.generator_model,
        evaluator_model=args.evaluator_model,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        seed=args.seed,
        use_rag=not args.no_rag,
        use_llm_evaluation=args.llm_evaluation,
        dry_run=args.dry_run,
        trust_checked_in_index=args.trust_checked_in_index,
        timeout_seconds=args.timeout_seconds,
    )
    policy = load_exclusion_policy(ROOT / "config" / "paper_exclusions.json")
    output_dir = run_demo(config, policy)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
