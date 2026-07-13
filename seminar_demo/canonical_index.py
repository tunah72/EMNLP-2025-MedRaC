from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from .github_models import (
    DEFAULT_EMBEDDING_MODEL,
    github_models_client,
    validate_openai_catalog_model,
)


INDEX_FILE = "index.faiss"
DOCUMENTS_FILE = "documents.json"
MANIFEST_FILE = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_documents(formula_path: Path) -> list[dict[str, str]]:
    payload = json.loads(formula_path.read_text(encoding="utf-8"))
    if len(payload) != 55:
        raise ValueError(f"Expected 55 canonical formulas, found {len(payload)}")
    documents: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in payload:
        calculator_id = str(item["Calculator ID"])
        if calculator_id in seen:
            raise ValueError(f"Duplicate Calculator ID: {calculator_id}")
        seen.add(calculator_id)
        question = str(item["Question"]).strip()
        formula = str(item["Formula"]).strip()
        documents.append(
            {
                "calculator_id": calculator_id,
                "question": question,
                "formula": formula,
                "embedding_text": (
                    f"Calculator ID: {calculator_id}\nQuestion: {question}\nFormula: {formula}"
                ),
            }
        )
    return documents


def build_canonical_index(
    formula_path: Path,
    output_dir: Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    client: Any | None = None,
) -> dict[str, Any]:
    validate_openai_catalog_model(embedding_model, embedding=True)
    documents = canonical_documents(formula_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty index directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_client = client or github_models_client()
    response = embedding_client.embeddings.create(
        model=embedding_model,
        input=[document["embedding_text"] for document in documents],
        encoding_format="float",
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    vectors = np.asarray([item.embedding for item in ordered], dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != 55:
        raise ValueError(f"Expected embedding matrix with 55 rows, found {vectors.shape}")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path = output_dir / INDEX_FILE
    documents_path = output_dir / DOCUMENTS_FILE
    faiss.write_index(index, str(index_path))
    documents_path.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "format_version": 1,
        "provider": "github-models-free-tier",
        "embedding_model": embedding_model,
        "knowledge_base": "55 canonical formulas from data/formula_new.json",
        "document_count": len(documents),
        "vector_dimension": int(vectors.shape[1]),
        "similarity": "cosine via normalized inner product",
        "source_formula_path": str(formula_path),
        "source_formula_sha256": _sha256(formula_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_input_tokens": int(
            getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        ),
        "artifacts": {
            INDEX_FILE: _sha256(index_path),
            DOCUMENTS_FILE: _sha256(documents_path),
        },
    }
    (output_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def validate_canonical_index(
    index_dir: Path, *, formula_path: Path | None = None
) -> dict[str, Any]:
    manifest_path = index_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Canonical index is missing: {manifest_path}; run scripts/build_canonical_index.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("document_count") != 55:
        raise ValueError("Canonical index manifest must contain exactly 55 documents")
    for name in (INDEX_FILE, DOCUMENTS_FILE):
        path = index_dir / name
        expected = manifest["artifacts"][name]
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Canonical index hash mismatch: {path}")
    if formula_path is not None and _sha256(formula_path) != manifest["source_formula_sha256"]:
        raise ValueError("Canonical index was built from a different formula_new.json")
    index = faiss.read_index(str(index_dir / INDEX_FILE))
    documents = json.loads((index_dir / DOCUMENTS_FILE).read_text(encoding="utf-8"))
    if index.ntotal != 55 or len(documents) != 55:
        raise ValueError("Canonical index must contain 55 vectors and documents")
    if index.d != manifest["vector_dimension"]:
        raise ValueError("Canonical index dimension differs from manifest")
    return manifest


@dataclass
class CanonicalFormulaIndex:
    index_dir: Path
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    client: Any | None = None

    def __post_init__(self) -> None:
        validate_openai_catalog_model(self.embedding_model, embedding=True)
        manifest = validate_canonical_index(self.index_dir)
        if manifest["embedding_model"] != self.embedding_model:
            raise ValueError("Query embedding model differs from index embedding model")
        self.index = faiss.read_index(str(self.index_dir / INDEX_FILE))
        self.documents = json.loads(
            (self.index_dir / DOCUMENTS_FILE).read_text(encoding="utf-8")
        )
        self.client = self.client or github_models_client()
        self.last_embedding_input_tokens = 0

    def retrieve(self, query: str, k: int = 1) -> list[tuple[str, float]]:
        if k < 1:
            raise ValueError("k must be >= 1")
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query],
            encoding_format="float",
        )
        self.last_embedding_input_tokens = int(
            getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        )
        vector = np.asarray([response.data[0].embedding], dtype="float32")
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, min(k, len(self.documents)))
        return [
            (self.documents[int(index)]["formula"], float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]
