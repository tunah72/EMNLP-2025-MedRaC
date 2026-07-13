import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from schema.schemas import Values
from seminar_demo.canonical_index import (
    CanonicalFormulaIndex,
    build_canonical_index,
    canonical_documents,
    validate_canonical_index,
)
from seminar_demo.github_models import GitHubModelsLLM, require_github_token


ROOT = Path(__file__).resolve().parents[1]


class _EmbeddingsAPI:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        vectors = []
        for index, _ in enumerate(inputs):
            vector = [0.0] * 8
            vector[index % len(vector)] = 1.0
            vectors.append(SimpleNamespace(index=index, embedding=vector))
        return SimpleNamespace(data=vectors)


class _ChatCompletionsAPI:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class GitHubModelsCompatibilityTests(unittest.TestCase):
    def test_canonical_knowledge_base_has_55_unique_formulas(self):
        documents = canonical_documents(ROOT / "data" / "formula_new.json")
        self.assertEqual(len(documents), 55)
        self.assertEqual(len({item["calculator_id"] for item in documents}), 55)

    def test_build_load_and_query_separate_safe_index(self):
        client = SimpleNamespace(embeddings=_EmbeddingsAPI())
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "canonical-index"
            manifest = build_canonical_index(
                ROOT / "data" / "formula_new.json", index_dir, client=client
            )
            self.assertEqual(manifest["document_count"], 55)
            self.assertFalse((index_dir / "index.pkl").exists())
            validate_canonical_index(
                index_dir, formula_path=ROOT / "data" / "formula_new.json"
            )
            retriever = CanonicalFormulaIndex(index_dir, client=client)
            result = retriever.retrieve("query", k=1)
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0][0])

    def test_generation_uses_github_model_id_and_json_schema(self):
        payload = {
            "extracted_values_reason": "structured public field",
            "extracted_values": [{"key": "age", "value": "70"}],
        }
        chat = _ChatCompletionsAPI(json.dumps(payload))
        client = SimpleNamespace(chat=SimpleNamespace(completions=chat))
        model = GitHubModelsLLM("openai/gpt-4.1", client=client)
        result = model.generate([("system", "user")], schema=Values)
        self.assertEqual(result[0], (payload, 10, 5))
        request = chat.calls[0]
        self.assertEqual(request["model"], "openai/gpt-4.1")
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual(request["seed"], 42)

    def test_non_openai_catalog_model_is_rejected(self):
        with self.assertRaises(ValueError):
            GitHubModelsLLM("meta/llama", client=object())

    def test_token_loader_does_not_depend_on_stdin_caller_discovery(self):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}, clear=False):
            self.assertEqual(require_github_token(), "test-token")


if __name__ == "__main__":
    unittest.main()
