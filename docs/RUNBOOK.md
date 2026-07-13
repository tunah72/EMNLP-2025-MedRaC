# API Demo Runbook

## Supported environment

Verified target: macOS 26.5.2 on Apple ARM64 with Python 3.11. The API demo does
not require CUDA, a GPU, vLLM, a local model server, or local model weights.
Other operating systems may work but are not yet verified.

## Create the isolated environment

From the repository root:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-demo.txt
```

Do not install the original `requirements.txt` for the API-only seminar path.

## Environment variables

```bash
cp .env.example .env
```

The default path uses `GITHUB_TOKEN`, not `OPENAI_API_KEY`. Create a fine-grained
GitHub personal access token with `models:read`, place it in `.env`, and never
print or commit it. `OPENAI_API_KEY` is only for the optional legacy direct
OpenAI path.

## GitHub Models free-tier compatibility

### 1. Keep paid usage disabled

Before any inference request, open GitHub **Settings → Billing and licensing**
and confirm paid usage for GitHub Models is not enabled. The API has no
per-request `free_only` switch. This demo never changes billing settings and
never falls back to another provider; when paid usage is disabled, GitHub blocks
requests after the included quota is exhausted.

### 2. Validate the catalog and token without inference

Load `.env` into the current shell without displaying it:

```bash
set -a
source .env
set +a
```

Then check that the token can read the two required catalog entries:

```bash
curl -fsS \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  https://models.github.ai/catalog/models \
| .venv/bin/python -c '
import json, sys
ids = {item["id"] for item in json.load(sys.stdin)}
required = {"openai/gpt-4.1", "openai/text-embedding-3-small"}
missing = required - ids
assert not missing, f"Missing catalog models: {sorted(missing)}"
print("GitHub Models catalog/token: OK")
'
```

This catalog check does not generate text or embeddings. A `401`/`403` means
the PAT is missing or does not have `models:read`.

For a minimal embedding permission probe, do not call bare `load_dotenv()` from
`python - <<'PY'`; some `python-dotenv` versions cannot discover a caller file
in stdin mode. The project client loads the repository `.env` explicitly:

```bash
.venv/bin/python - <<'PY'
from seminar_demo.github_models import github_models_client

client = github_models_client()
response = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input=["permission probe"],
)
assert len(response.data[0].embedding) == 1536
print("GitHub Models embedding access: OK")
PY
```

### 3. Build the separate 55-formula index

First validate the source without an API call:

```bash
.venv/bin/python scripts/build_canonical_index.py --dry-run
```

Then build the index. This sends the 55 canonical embedding inputs in one
embedding request and writes only the new directory:

```bash
.venv/bin/python scripts/build_canonical_index.py
```

Expected files:

```text
data/github_models_canonical_formulas_index/
├── index.faiss
├── documents.json
└── manifest.json
```

The builder refuses to overwrite a non-empty output directory. It never edits
or loads `data/one_shot_finalized_explanation_formulas_embeddings/`.

Validate the built index locally, without provider calls:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from seminar_demo.canonical_index import validate_canonical_index

manifest = validate_canonical_index(
    Path("data/github_models_canonical_formulas_index"),
    formula_path=Path("data/formula_new.json"),
)
assert manifest["document_count"] == 55
assert manifest["embedding_model"] == "openai/text-embedding-3-small"
print("canonical index: OK, vectors=55, dimension=", manifest["vector_dimension"])
PY
```

### 4. Run offline verification and dry-run

```bash
.venv/bin/python -m compileall -q -f scripts seminar_demo tests
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/demo_pipeline.py --help
.venv/bin/python scripts/demo_pipeline.py --dry-run --row-indices 0,40 --seed 42
```

After the real index exists, the dry-run manifest must contain:

```text
configuration.provider = github-models
configuration.generator_model = openai/gpt-4.1
configuration.evaluator_model = openai/gpt-4.1
configuration.embedding_model = openai/text-embedding-3-small
configuration.paid_usage_fallback = false
index.status = ready
index.manifest.document_count = 55
```

If the index has not been built, dry-run succeeds as a preflight but records
`dry_run_preflight_missing_index`; a live RAG run fails before any generation.

### 5. One-sample online smoke test

Run without LLM step-wise evaluation first:

```bash
.venv/bin/python scripts/demo_pipeline.py \
  --provider github-models \
  --row-indices 0 \
  --generator-model openai/gpt-4.1 \
  --evaluator-model openai/gpt-4.1 \
  --embedding-model openai/text-embedding-3-small \
  --seed 42
```

This normally makes three free-tier requests for one sample: one query
embedding, one value-extraction generation, and one code generation. It does
not use the direct OpenAI API. Confirm the printed artifact directory, then:

```bash
RUN_DIR="$(find artifacts/demo -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
.venv/bin/python - "$RUN_DIR" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
manifest = json.loads((run / "manifest.json").read_text())
samples = json.loads((run / "samples.json").read_text())
assert manifest["status"] == "completed"
assert manifest["configuration"]["provider"] == "github-models"
assert manifest["configuration"]["paid_usage_fallback"] is False
assert samples[0]["retrieval"]["status"] == "success"
assert samples[0]["retrieval"]["score"] is not None
assert samples[0]["extraction"]["status"] == "success"
assert samples[0]["code_generation"]["status"] == "success"
assert samples[0]["execution"]["execution_mode"] == "safe_ast_child_process"
assert samples[0]["rule_evaluation"]["status"] == "success"
print("one-sample pipeline smoke: OK")
PY
```

Only after that succeeds, optionally add `--llm-evaluation`. It adds four
evaluation requests per sample and may hit the free daily quota sooner.

### 6. Optional F–E–C–A step-wise evaluation

`rule_evaluation` checks only whether the final numerical/categorical answer is
acceptable. `stepwise_evaluation` diagnoses the intermediate pipeline stages:

| Field | Meaning | Released evaluator behavior |
|---|---|---|
| `formula` (F) | Was the selected medical formula/scoring rule appropriate? | LLM judge compares the retrieved formula, together with generated calculation text when different, against the canonical formula. |
| `extracted_values` (E) | Were the required patient values extracted correctly? | LLM judge compares extracted values against the dataset's `Relevant Entities`. |
| `calculation` (C) | Was the arithmetic or scoring computation valid? | LLM judge checks mathematical correctness without receiving the gold calculation reference. |
| `answer` (A) | Is the final answer acceptable? | LLM parses/normalizes the answer, then `RegEvaluator` applies the repository's rule-based correctness check. |

The field is initialized as:

```json
"stepwise_evaluation": {
  "status": "not_run",
  "evaluation": null
}
```

This is expected when `--llm-evaluation` is absent; it is not a pipeline error
and must not be interpreted as an incorrect F/E/C/A result. Enable it for one
sample first:

```bash
.venv/bin/python scripts/demo_pipeline.py \
  --provider github-models \
  --row-indices 0 \
  --generator-model openai/gpt-4.1 \
  --evaluator-model openai/gpt-4.1 \
  --embedding-model openai/text-embedding-3-small \
  --llm-evaluation \
  --seed 42
```

Successful output has one structured result per stage:

```json
"stepwise_evaluation": {
  "status": "success",
  "evaluation": {
    "formula": {"result": "Correct", "explanation": "..."},
    "extracted_values": {"result": "Correct", "explanation": "..."},
    "calculation": {"result": "Correct", "explanation": "..."},
    "answer": {"result": "Correct", "explanation": "..."}
  }
},
"first_failed_step": null
```

`first_failed_step` is the earliest `Incorrect` stage in F -> E -> C -> A
order. It remains `null` when all four stages are correct or when step-wise
evaluation was not run. Distinguish those cases using
`stepwise_evaluation.status`.

For one MedRaC sample, enabling this option adds four chat requests: three LLM
judges for F/E/C and one LLM answer-normalization request. The A verdict itself
is then computed locally by `RegEvaluator`. The normal MedRaC generation path
still uses two chat requests plus one query-embedding request, so the complete
one-sample run normally uses six chat requests and one embedding request.

The evaluator explanations are public structured output fields. They are not
provider-hidden chain-of-thought and should not be expanded into or described
as hidden model reasoning. LLM judgments can be inconsistent and are not a
substitute for clinical expert review.

## Phase 5 mini experiment

The deterministic selection is `config/mini_experiment.json`: ten rows, seven
equation calculators, three rule calculators, and ten distinct calculator IDs.
The same ordered rows and generation model/parameters are used for both
`Plain("cot")` and `MedRaC(use_rag=True)`.

Run the offline preflight:

```bash
.venv/bin/python scripts/run_mini_experiment.py --dry-run
```

Run a one-row paired smoke without step-wise evaluation:

```bash
.venv/bin/python scripts/run_mini_experiment.py --limit 1
```

Run the default ten-row comparison:

```bash
.venv/bin/python scripts/run_mini_experiment.py
```

The default comparison uses 30 chat attempts and 10 query-embedding attempts:
one Plain generation plus two MedRaC generations and one MedRaC query embedding
per row. Requests are interleaved by row and throttled for the GitHub Models
free tier. The Phase 5 timeout default is 5 seconds; override it only if the
chosen value is recorded and justified.

Enable full F/E/C/A only after checking quota and explicitly accepting the
additional request volume:

```bash
.venv/bin/python scripts/run_mini_experiment.py \
  --llm-evaluation
```

For ten paired rows this adds 80 chat requests: four Plain evaluator calls and
four MedRaC evaluator calls per row. Do not run this immediately after the
default experiment unless sufficient quota is confirmed.

Do not use that command to evaluate the already completed 10-row artifact: it
would also repeat 30 generation requests. Use the evaluation-only workflow
below instead.

### Post-hoc F/E/C/A evaluation of stored outputs

First validate the entire 20-unit plan without a token or API request:

```bash
.venv/bin/python scripts/evaluate_experiment_artifact.py \
  --source-artifact artifacts/experiments/20260713T060240915862Z \
  --dry-run
```

Expected plan: 20 method/sample evaluation units, four judge calls each, 80
chat requests total, and `api_requests_made: 0`.

Use one stable derived output path and split the calls into two quota windows.
Batch 1:

```bash
.venv/bin/python scripts/evaluate_experiment_artifact.py \
  --source-artifact artifacts/experiments/20260713T060240915862Z \
  --output-dir artifacts/experiments/20260713T060240915862Z-stepwise \
  --batch-indices 0 40 20 120 60
```

Batch 2, in a later quota window:

```bash
.venv/bin/python scripts/evaluate_experiment_artifact.py \
  --source-artifact artifacts/experiments/20260713T060240915862Z \
  --output-dir artifacts/experiments/20260713T060240915862Z-stepwise \
  --batch-indices 200 80 100 160 280
```

The first batch leaves the derived manifest `in_progress`; the second reaches
`completed`. Re-running either command resumes safely and skips units whose
step-wise status is already `success`. The runner checkpoints after every
method/sample and stops on the first failure so quota is not consumed by an
unbounded retry loop.

The derived artifact contains the same four files as the parent. Its metrics
include F/E/C/A accuracy, strict accuracy, Conditional Correctness,
paper-faithful FE counts/rates, and masked reasoning failures. Do not present
aggregate values as final while the manifest is `in_progress`.

Each run writes:

```text
artifacts/experiments/<run-id>/
├── manifest.json
├── samples.json
├── metrics.json
└── report.md
```

`manifest.json` records Git SHA, command, rows/IDs, models, embedding model,
temperatures, seed, dependency versions, timestamps, components, timeout, index
provenance, and output paths. `metrics.json` records method-specific and paired
denominators, rule accuracy, optional F/E/C/A metrics, strict accuracy,
Conditional Correctness, First Error Attribution, tokens, requests, and
failed/skipped counts. Missing step-wise metrics use `value: null`, never a
fabricated zero.

### Locally reproduced Phase 5 result

Command:

```bash
.venv/bin/python scripts/run_mini_experiment.py
```

Artifact:

```text
artifacts/experiments/20260713T060240915862Z/
```

| Result | Plain CoT | MedRaC |
|---|---:|---:|
| Method-specific rule accuracy | 9/10 | 10/10 |
| Completed / failed | 10 / 0 | 10 / 0 |
| Paired successful-row accuracy | 9/10 | 10/10 |
| Chat attempts | 10 | 20 |
| Embedding attempts | 0 | 10 |

DataFrame index 20 / Calculator ID 3 originally exceeded the run's 2-second
process timeout. Its stored code was re-executed locally with a 5-second
timeout and no API request, producing a rule-correct answer. The original
timeout remains in post-run provenance. MedRaC was uniquely correct on index
40 / Calculator ID 4; the other nine paired rows were correct for both methods.

This run did not enable `--llm-evaluation`; F/E/C/A and derived strict,
conditional, and first-error metrics are unavailable. The output is a local
seminar-scale comparison, not a paper-result reproduction and not evidence of
statistical significance.

## Import and compile checks

```bash
.venv/bin/python -m compileall -q -f run.py method model evaluator schema utils
.venv/bin/python - <<'PY'
from model import APIModel
from method.plain import Plain
from method.medRaC import MedRaC
from evaluator import RegEvaluator, LLM_Evaluator
from method.rag import RAG
print("API demo imports: OK")
PY
```

## Dataset and formula validation

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import pandas as pd

rows = pd.read_csv("data/test_data.csv")
formulas = json.loads(Path("data/formula_new.json").read_text(encoding="utf-8"))
corpus = Path("data/web_formula.txt").read_text(encoding="utf-8")

assert len(rows) == 1048
assert rows["Calculator ID"].nunique() == 55
assert len(formulas) == 55
assert corpus.count("<<FORMULA START>>") == 785
assert corpus.count("<<FORMULA END>>") == 785
print("dataset rows=1048 calculator_ids=55 formulas=55 rag_blocks=785")
PY
```

## Controlled checked-in FAISS validation

This repository's `RAG` loader calls LangChain with
`allow_dangerous_deserialization=True`. Run this only for the two checked-in
files, inside `.venv`, after confirming their hashes:

```bash
test "$(shasum -a 256 data/one_shot_finalized_explanation_formulas_embeddings/index.faiss | awk '{print $1}')" = "ef8b80ab8566e28b73a78536ec373f767a9bbbb8cde4465c310ad8c080328a9a"
test "$(shasum -a 256 data/one_shot_finalized_explanation_formulas_embeddings/index.pkl | awk '{print $1}')" = "fd4de9d5ba416f2004d6c35a6408ed3c12abdb0903c7dbf36b6f44469179dbd8"
```

Loading the vector store itself is local. A normal `RAG.retrieve()` embeds its
query remotely, so it is **not** the no-API smoke command. The verified
retrieval-only check reconstructs the stored embedding for the first indexed
document and performs local FAISS search:

```bash
OPENAI_API_KEY=not-used .venv/bin/python - <<'PY'
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

path = "data/one_shot_finalized_explanation_formulas_embeddings"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key="not-used")
store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
assert store.index.ntotal == len(store.index_to_docstore_id) == 776
vector = store.index.reconstruct(0)
hits = store.similarity_search_by_vector(vector, k=1)
assert hits and hits[0].page_content
print(f"faiss vectors={store.index.ntotal} local_top1_chars={len(hits[0].page_content)}")
PY
```

Do not point this command at external files and do not rebuild the index after a
failure. Record the exact error and treat RAG as blocked pending a documented
compatibility decision.

## Dry-run boundary

Phase 4 provides a non-API dry run. The fixed default pair is equivalent to
`--row-indices 0,40`:

```bash
.venv/bin/python scripts/demo_pipeline.py --dry-run --row-indices 0,40 --seed 42
```

Calculator-ID selection deterministically chooses the first retained row for
each requested ID:

```bash
.venv/bin/python scripts/demo_pipeline.py --dry-run --calculator-ids 2,4 --seed 42
```

Inspect available flags with:

```bash
.venv/bin/python scripts/demo_pipeline.py --help
```

With the default GitHub Models provider, dry-run validates the 1,048 -> 940
exclusion policy, selected rows, 55-formula source, and the new index manifest
and hashes when present. It writes runtime stages as `not_run` and does not
initialize or call provider clients. Use `--provider openai` only to validate
the legacy checked-in index path.

Demo outputs are written to `artifacts/demo/<run-id>/manifest.json`,
`samples.json`, and `report.md`. The directory is git-ignored. Future experiment
outputs remain planned at `artifacts/experiments/<run-id>/`.

## Legacy direct-OpenAI command boundary

The preserved direct-OpenAI/checked-in-index path uses:

```bash
.venv/bin/python scripts/demo_pipeline.py \
  --provider openai \
  --row-indices 0 \
  --trust-checked-in-index \
  --generator-model OpenAI/gpt-4o-mini \
  --evaluator-model OpenAI/gpt-4o-mini \
  --embedding-model text-embedding-ada-002 \
  --seed 42
```

Add `--llm-evaluation` only when the extra four evaluator requests per sample
are authorized. This legacy command requires `OPENAI_API_KEY` and is not the
free-tier recommendation. Both live providers execute generated code only
through `safe_ast_child_process`.

The current safe executor supports arithmetic, comparisons, conditionals,
selected builtins, and approved `math` functions. It rejects imports and does
not yet support date calculators. Rejections are recorded; never fall back to
`MedRaC.execute_code`.

## Cleanup

```bash
rm -rf .venv
rm -rf __pycache__ method/__pycache__ model/__pycache__ evaluator/__pycache__ schema/__pycache__ utils/__pycache__
```

Do not delete checked-in FAISS files, benchmark data, formulas, or documentation.

## Known issues

- The original full dependency file includes vLLM and is not the API-demo
  installation path.
- A demo provider failure after run-directory creation but before manifest
  writing can leave an empty `artifacts/demo/<run-id>/` directory. When locating
  a completed demo, search for `manifest.json` rather than assuming the newest
  directory is valid. Phase 5 metrics never scan demo directories.
- The checked-in index requires trusted pickle deserialization.
- Normal text-query retrieval calls the OpenAI embeddings API; the local smoke
  check above intentionally uses a stored vector instead.
- The source corpus has 785 delimited blocks, but the checked-in index and
  docstore contain 776 entries covering block indices 0–775. The final nine
  corpus blocks are not represented; the index was not rebuilt.
- The original generated-code execution path is unsafe and is not exercised by
  this runbook.

## Phase 3 verification record

Executed on 2026-07-13 from commit
`900c54123ce03ca467ebdf0b2f28afb7dcaabc5e`:

| Check | Command summary | Exit/result |
|---|---|---|
| Create environment | `/opt/homebrew/bin/python3.11 -m venv .venv` | 0, success |
| Install | upgrade pip; `pip install -r requirements-demo.txt` | 0, success |
| Dependency consistency | `.venv/bin/python -m pip check` | 0, no broken requirements |
| Compile | `.venv/bin/python -m compileall -q -f run.py method model evaluator schema utils` | 0, success |
| Required imports | import `APIModel`, `Plain`, `MedRaC`, both evaluators, and `RAG` | 0, success |
| Model initialization | initialize OpenAI and DeepSeek wrappers with non-secret placeholders; assert placeholders absent from stdout | 0, success; no network call |
| Data validation | pandas/JSON/text assertions and formula lookup for IDs 2 and 4 | 0; 1,048 rows, 55 IDs, 55 formulas, 785 blocks |
| Hash gate | compare both checked-in artifacts to audited SHA-256 values | 0, both match |
| First index assertion | expected 785 vectors after successful load | 1; assertion exposed a 776/785 mismatch |
| Controlled index inspection | load trusted index and inspect sizes | 0; 776 vectors, mappings, and documents; dimension 1,536 |
| Retrieval-only smoke | reconstruct stored vector 0 and call `similarity_search_by_vector(k=3)` | 0; three local hits, top hit block index 0 |
| Coverage inspection | compare docstore `block_index` values with 0–784 | 0; index covers 0–775, missing 776–784, no duplicates |

Direct dependency versions:

```text
faiss-cpu==1.11.0
langchain==0.3.26
langchain-community==0.3.26
langchain-core==0.3.86
langchain-openai==0.3.27
nest-asyncio==1.6.0
numpy==1.26.2
openai==1.93.0
pandas==2.3.0
pydantic==2.11.7
python-dotenv==1.1.1
tiktoken==0.9.0
tqdm==4.67.1
```

No OpenAI/DeepSeek/embedding request was made. No generated code ran, no
benchmark/formula changed, and the index was not rebuilt. RAG loading and local
vector search are operational; text-query retrieval remains unverified because
it would require an embedding API call. The 776/785 corpus-index mismatch is a
non-blocking Phase 4 provenance risk.

### Phase 4 verification

```text
python -m compileall -q -f scripts seminar_demo tests       PASS
python -m unittest discover -s tests -v                    PASS (13 tests)
python scripts/demo_pipeline.py --help                     PASS
python scripts/demo_pipeline.py --dry-run --row-indices 0,40 --seed 42
                                                             PASS
python scripts/demo_pipeline.py --dry-run --calculator-ids 2,4 --seed 42
                                                             PASS
```

The dry-run artifacts confirmed 1,048 source rows, 108 exclusions, 940 retained
rows, selected IDs 2 and 4, matching index hashes, `inferred but unverified`
provenance, and no fabricated stage results. No API or generated-code execution
occurred.

## GitHub Models compatibility verification — 2026-07-13

The compatibility implementation was verified on the same Python 3.11
environment:

| Check | Result |
|---|---|
| GitHub catalog preflight | HTTP 200; 37 catalog entries; both `openai/gpt-4.1` and `openai/text-embedding-3-small` present |
| Canonical source validation | 55 unique Calculator IDs; estimated 12,725 embedding input tokens; largest document 1,169 tokens |
| Index build | Success in one embedding request; 55 vectors; dimension 1,536 |
| New index format | `index.faiss`, `documents.json`, `manifest.json`; no pickle |
| Original index integrity | Both audited SHA-256 hashes unchanged |
| Compile | PASS for `scripts`, `seminar_demo`, and `tests` |
| Unit tests | PASS, 17 tests including mocked GitHub chat/embedding compatibility |
| GitHub-default dry-run | PASS; provider/model/index metadata recorded; no provider call |
| Legacy OpenAI dry-run | PASS with explicit legacy model and index arguments |
| One-sample live smoke | PASS for DataFrame index 0 / Calculator ID 2 |

Live smoke artifact:

```text
artifacts/demo/20260713T051602329268Z/
```

Observed stages were retrieval success (cosine score 0.8930958), extraction
success, code generation success, safe child-process execution success, and
rule evaluation `Correct`; final answer was `25.238095238095237`. Recorded chat
usage was 1,620 + 263 tokens for extraction and 563 + 258 for code generation.
LLM step-wise evaluation was intentionally not enabled, so no F/E/C/A evaluator
result is claimed from this smoke run.
