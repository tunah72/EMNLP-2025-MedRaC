# Seminar Demo Final Report

## Completion status

Phases 1–6 of the seminar-demo plan are complete as of 2026-07-13. Phase 5 is
closed with a deterministic ten-row Plain CoT versus MedRaC comparison evaluated
by the repository rule evaluator. The optional post-hoc F/E/C/A evaluation is
deferred to later GitHub Models quota windows; no CC, FE, strict, or step-level
local result is claimed yet.

The locally reproduced rule-based result is:

| Method | Correct | Denominator | Accuracy |
|---|---:|---:|---:|
| Plain CoT | 9 | 10 | 0.900 |
| MedRaC with RAG | 10 | 10 | 1.000 |

This is a small seminar-scale engineering comparison. It is not a reproduction
of the paper's reported results, a clinical validation, or evidence of
statistical significance.

## Delivered changes

- Repository-to-paper audit and persistent research, experiment, decision, and
  runbook documentation under `docs/`.
- Minimal API-demo environment in `requirements-demo.txt` and `.env.example`.
- Deterministic retained-row and paper-exclusion manifests under `config/`.
- Transparent demo, canonical-index builder, mini-experiment runner, and
  evaluation-only post-hoc runner under `scripts/`.
- GitHub Models compatibility, separate 55-formula canonical index support,
  safe AST/child-process execution, experiment aggregation, CC/FE metrics, and
  resumable derived-artifact handling under `seminar_demo/`.
- Focused tests under `tests/` for deterministic selection, formula/index
  lookup, structured response parsing, code-fence removal, safe execution and
  rejection, manifest generation, metric aggregation, and post-hoc resume.
- A single-page `streamlit_app.py` seminar interface plus
  `seminar_demo/ui_adapter.py`, `.streamlit/config.toml`, and focused adapter
  tests. It presents MedRaC RAG replay and one-row deterministic live execution,
  keeps Patient Note and Question separate, and exposes structured F/E/C/A
  output without displaying hidden chain-of-thought.
- Small compatibility edits in `.gitignore`, `model/__init__.py`, and
  `model/gpt.py`; the original `requirements.txt`, benchmark CSV, medical
  formulas, and checked-in FAISS index were not replaced.

Changed project files are grouped below:

```text
AGENTS.md
.env.example
.gitignore
requirements-demo.txt
.streamlit/config.toml
streamlit_app.py
model/__init__.py
model/gpt.py
config/mini_experiment.json
config/paper_exclusions.json
docs/DECISIONS.md
docs/EXPERIMENT_PLAN.md
docs/FINAL_REPORT.md
docs/PAPER_CONTEXT.md
docs/REPOSITORY_AUDIT.md
docs/RUNBOOK.md
scripts/build_canonical_index.py
scripts/demo_pipeline.py
scripts/evaluate_experiment_artifact.py
scripts/run_mini_experiment.py
seminar_demo/*.py
tests/test_streamlit_ui_adapter.py
tests/test_*.py
```

## Verification record

Final offline verification was run on macOS 26.5.2 with Python 3.11.15 from Git
SHA `7b7db2c0312596b3fd2ed8fd0ea30f64ff945428`.

```text
.venv/bin/python -m compileall -q -f seminar_demo scripts tests
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python scripts/demo_pipeline.py --provider github-models \
  --row-indices 0 --dry-run --output-dir <temporary-dir>/demo
.venv/bin/python scripts/run_mini_experiment.py \
  --dry-run --output-dir <temporary-dir>/experiment
.venv/bin/python scripts/evaluate_experiment_artifact.py \
  --source-artifact artifacts/experiments/20260713T060240915862Z \
  --dry-run
.venv/bin/pip check
git diff --check
```

All commands above passed. The unit suite reported 28 tests.

The post-hoc dry-run found 20 pending method/sample units and estimated 80 chat
requests while reporting `api_requests_made: 0`. SHA-256 hashes of the Phase 5
parent `manifest.json` and `samples.json` were identical before and after the
check. No API-dependent command was executed during final verification.

Previously completed API-dependent checks:

- One-sample transparent demo smoke: completed successfully with retrieval,
  extraction, code generation, safe execution, and rule evaluation.
- Ten-row paired experiment: completed successfully after the documented local,
  zero-API timeout repair at index 20.

No final check failed. Temporary final dry-run artifacts were written outside
the repository under `/tmp` and are not project deliverables.

### Streamlit integration review

After the user confirmed the application works, the Streamlit integration was
reviewed again on 2026-07-13. The review covered actual source paths, all 39
unit tests, forced compilation, dependency consistency, patch whitespace,
Streamlit AppTest in Replay and credential-free Dataset live modes, a headless
server smoke, and a browser DOM/visual inspection. All final checks passed; no
API, evaluation, or embedding request was made.

Verified Streamlit data flow:

```text
manifest-selected dataset row
  -> Calculator ID + Patient Note + Question
  -> Question-only formula retrieval
  -> Patient Note + Question + retrieved formula extraction
  -> retrieved formula + extracted values + Question code generation
  -> safe AST validation and child-process execution
  -> final answer, rule evaluation, and optional F/E/C/A evaluation
```

Replay is schema-tolerant for `Patient Note`, `Clinical Note`, and normalized
internal names while preserving field provenance. Artifact discovery accepts
directories with readable manifests and list-shaped samples, surfaces their
recorded status/provenance, and skips missing or malformed directories. Dataset
live mode remains limited to the identity-checked rows in
`config/mini_experiment.json`; custom free-text input remains deferred.

## Artifact paths

- Live demo: `artifacts/demo/20260713T051602329268Z/`
- Final Phase 5 experiment: `artifacts/experiments/20260713T060240915862Z/`
- Future derived step-wise artifact:
  `artifacts/experiments/20260713T060240915862Z-stepwise/`

The `artifacts/` directory and `.env` are ignored by Git. The parent experiment
must remain unchanged. When quota becomes available, use the two resumable
batches in `docs/RUNBOOK.md` and report step-wise metrics only after the derived
manifest reaches `status: "completed"`.

## Unresolved risks and deferred work

- F/E/C/A, strict correctness, CC, FE, and masked-reasoning metrics are pending
  the deferred 80-request post-hoc evaluation.
- The ten-row sample is too small for statistical inference.
- GitHub Models catalog availability, access, and free-tier quota can change;
  paid usage must remain disabled and there is no provider fallback.
- Formula, Extraction, and Calculation verdicts depend on an LLM judge; the
  released Answer stage is hybrid LLM parsing plus rule evaluation.
- The original checked-in FAISS index requires trusted pickle deserialization
  and covers 776 of 785 parsed web-formula blocks. The seminar compatibility
  path instead uses the separate 55-formula JSON-backed index.
- The seminar executor is a minimal risk-reduction layer, not a hardened secure
  sandbox and not suitable for arbitrary untrusted code.
- The index-20 result includes a disclosed offline timeout repair; both the
  original timeout and repaired result remain in artifact provenance.
- At narrower browser widths, long Streamlit metric-card values may be elided.
  This is a presentation-only limitation: complete values remain available in
  the sidebar selectors and dedicated input/stage fields.
