#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seminar_demo.data import load_exclusion_policy
from seminar_demo.experiment import ExperimentConfig, run_mini_experiment
from seminar_demo.github_models import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic Plain CoT versus MedRaC mini experiment"
    )
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=ROOT / "config" / "mini_experiment.json",
    )
    parser.add_argument("--generator-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--evaluator-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Use a deterministic prefix for smoke testing")
    parser.add_argument("--llm-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "experiments"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        dataset_path=ROOT / "data" / "test_data.csv",
        formula_path=ROOT / "data" / "formula_new.json",
        exclusion_path=ROOT / "config" / "paper_exclusions.json",
        sample_manifest_path=args.sample_manifest,
        index_dir=ROOT / "data" / "github_models_canonical_formulas_index",
        output_root=args.output_dir,
        generator_model=args.generator_model,
        evaluator_model=args.evaluator_model,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        seed=args.seed,
        use_llm_evaluation=args.llm_evaluation,
        dry_run=args.dry_run,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        command=[sys.executable, *sys.argv],
    )
    policy = load_exclusion_policy(config.exclusion_path)
    output_dir = run_mini_experiment(config, policy)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
