#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seminar_demo.experiment import METHODS
from seminar_demo.github_models import DEFAULT_CHAT_MODEL
from seminar_demo.posthoc_evaluation import PosthocEvaluationConfig, run_posthoc_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate stored Plain CoT and MedRaC outputs without regenerating them"
    )
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New derived artifact directory; reuse the same path to resume",
    )
    parser.add_argument("--evaluator-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--evaluator-temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-indices",
        type=int,
        nargs="+",
        help="Evaluate only these source DataFrame indices in this resumable batch",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Methods to evaluate in this batch",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = PosthocEvaluationConfig(
        source_artifact=args.source_artifact,
        output_dir=args.output_dir,
        dataset_path=ROOT / "data" / "test_data.csv",
        formula_path=ROOT / "data" / "formula_new.json",
        evaluator_model=args.evaluator_model,
        evaluator_temperature=args.evaluator_temperature,
        seed=args.seed,
        batch_indices=args.batch_indices,
        methods=args.methods,
        dry_run=args.dry_run,
        command=[sys.executable, *sys.argv],
    )
    result = run_posthoc_evaluation(config)
    if isinstance(result, dict):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
