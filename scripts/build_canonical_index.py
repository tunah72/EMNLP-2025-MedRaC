#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seminar_demo.canonical_index import build_canonical_index, canonical_documents
from seminar_demo.github_models import DEFAULT_EMBEDDING_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a separate FAISS index from the 55 canonical formulas"
    )
    parser.add_argument(
        "--formula-path", type=Path, default=ROOT / "data" / "formula_new.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "github_models_canonical_formulas_index",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate 55 source records without API calls"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        documents = canonical_documents(args.formula_path)
        print(json.dumps({
            "status": "validated_without_api",
            "document_count": len(documents),
            "embedding_model": args.embedding_model,
            "output_dir": str(args.output_dir),
        }, indent=2))
        return 0
    manifest = build_canonical_index(
        args.formula_path,
        args.output_dir,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
