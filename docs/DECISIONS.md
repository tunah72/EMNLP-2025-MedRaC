# Project Decisions

This is an append-only decision record for the seminar-demo work. Changes to a
decision should be added as a dated replacement rather than silently rewriting
the scientific history.

## 2026-07-13 — Scope and evidence

- This project is a transparent seminar demo and simplified reproduction, not
  a full reproduction of every reported paper number.
- Every reported statement must be labelled **paper-reported**,
  **repository-reported**, **locally reproduced**, or
  **inferred but unverified**.
- The released CSV has 1,048 rows and 55 calculator IDs. The paper describes a
  cleaned cohort of 940 cases after excluding 108. Until that exact cohort is
  represented by a verified manifest, local results must not be compared as if
  they used the paper cohort.
- Benchmark rows and medical formulas will not be modified without explicit
  approval. Filtering will be represented as selection metadata, not in-place
  edits.
- `test=True` is prohibited for reported comparisons because it performs
  unseeded sampling. Baseline and MedRaC will use the same explicit row indices.
- No full-dataset experiment may run without explicit approval. The planned
  mini experiment will use approximately 10–20 fixed samples.

## 2026-07-13 — Architecture and safety

- The original `run.py` remains a fidelity reference and will not be broadly
  refactored. Seminar orchestration will be implemented later in separate
  scripts.
- The current `MedRaC.execute_code` implementation is unrestricted `exec`; it
  must not be called or described as a secure sandbox. Phase 4 will add a
  separate AST-validated, timeout-limited child-process execution mode.
- The checked-in FAISS files are trusted repository artifacts only at their
  audited paths and hashes. Arbitrary external indexes must never be loaded
  through dangerous deserialization.
- The checked-in index may be loaded in the isolated Phase 3 environment for a
  retrieval-only test. It must not be rebuilt. Rebuilding first requires a
  written estimate of embedding requests, token volume, cost, and expected
  retrieval drift.
- Controlled loading found 776 vectors/documents in the checked-in index while
  `data/web_formula.txt` currently contains 785 blocks. The index covers block
  indices 0–775 and omits 776–784. It remains usable for Phase 4 retrieval, but
  the nine-block provenance gap must be disclosed and the index must not be
  described as a complete embedding of the current corpus.

## 2026-07-13 — Models, dependencies, and deviations

- The initial runnable path is API-only: no vLLM server, CUDA, GPU, or local
  model weights are required. The original `requirements.txt` is preserved;
  `requirements-demo.txt` defines the smaller seminar environment.
- Python 3.11 is the supported environment for this verified setup. The host's
  default Python 3.13.5 is not used because the released pins target an older
  dependency combination.
- `model/__init__.py` now treats the vLLM stack as optional. Attempting to
  instantiate `vllmModels` without `torch`, `transformers`, and `vllm` raises a
  targeted error instead of breaking API-only imports.
- `model/gpt.py` no longer imports Hugging Face Transformers solely for the
  DeepSeek tokenizer. It uses `tiktoken`'s `cl100k_base` for local estimates;
  provider-returned usage remains authoritative. This is a tokenizer
  substitution, not a generation-model or evaluator substitution.
- Candidate low-cost defaults for the future experiment are
  `OpenAI/gpt-4o-mini` for generation and evaluation and
  `text-embedding-ada-002` for compatibility with the checked-in index. No API
  model was called in Phase 2/3.
- The paper's DeepSeek-chat evaluator and DeepSeek-reasoner error analyzer are
  optional fidelity configurations, not Phase 3 verification requirements.
- Generation temperature, evaluator temperature, reduced sample size, disabled
  error analysis, or any future model substitution must be recorded here and
  in each run manifest.

## 2026-07-13 — Phase 4 seminar pipeline

- Bad-data handling is non-destructive. `config/paper_exclusions.json`
  reproduces the repository helper's 108 exclusions and asserts a 940-row
  retained cohort; the source CSV remains unchanged.
- The default transparent demo uses zero-based DataFrame indices 0 and 40,
  corresponding to CSV Row Numbers 1 and 41 and Calculator IDs 2
  (equation-based Cockcroft-Gault) and 4 (rule-based CHA2DS2-VASc).
- Dry-run artifacts are labelled `inferred but unverified`; every generation,
  retrieval, execution, and evaluation stage remains `not_run`. Dry-run output
  must never be reported as a reproduced result.
- Live RAG requires explicit `--trust-checked-in-index` acknowledgement. Normal
  text retrieval makes an embedding API call; dry-run only validates hashes.
- The seminar code-generation prompt preserves formula/value-to-Python
  decomposition but narrows the permitted language to the safe executor's
  arithmetic and conditional subset. This is a safety deviation from the
  released prompt, which advertised broad packages including `os` and `sys`.
- The safe executor validates AST syntax, supplies a minimal builtin/math
  allowlist, runs in a spawned child process, and terminates after a strict
  timeout. It is a seminar-specific risk reduction layer, not a hardened or
  multi-tenant secure sandbox. It deliberately rejects imports, loops, comprehensions, arbitrary
  attributes, filesystem/network/shell access, reflection, and user input.
- Date calculators and generated code needing libraries outside the current
  numeric allowlist are not supported in the initial demo selection. They must
  be marked rejected/unsupported rather than routed to the original unsafe
  executor.
- The original `MedRaC.execute_code` remains unchanged for fidelity reference
  and is not called by the seminar pipeline.

## 2026-07-13 — GitHub Models compatibility phase

- The default seminar provider is now the GitHub Models free tier. Generation
  and optional step-wise evaluation use the same OpenAI-published catalog model,
  initially `openai/gpt-4.1`; embeddings use
  `openai/text-embedding-3-small`.
- This is a provider and model substitution from the paper and from the earlier
  checked-in-index compatibility candidate. Results from this path must be
  labelled as local seminar-demo results, not paper reproduction results.
- RAG uses exactly the 55 canonical entries in `data/formula_new.json`. Each
  embedding document contains Calculator ID, canonical question, and canonical
  formula. The expanded 785-block `data/web_formula.txt` corpus is not used by
  this compatibility path.
- The new index lives in `data/github_models_canonical_formulas_index/`. It is
  separate from and never overwrites the original
  `data/one_shot_finalized_explanation_formulas_embeddings/` directory.
- The new serialization is `index.faiss` plus JSON documents and a hash-bearing
  JSON manifest. It does not load a pickle. The generated index is ignored by
  Git because it can be recreated from the public canonical formulas.
- The compatibility code has no provider fallback. GitHub has no per-request
  “free only” flag, so paid usage must remain disabled in GitHub billing
  settings; with paid usage disabled, exhausted free quota blocks requests.
- The client uses the fixed GitHub Models endpoint and API version, requires a
  fine-grained `GITHUB_TOKEN` with `models:read`, enforces the `openai/`
  publisher, sends generation sequentially, and caps output at 2,048 tokens.
- The existing direct OpenAI and checked-in-index path remains available with
  `--provider openai` for fidelity reference, but is no longer the CLI default.

## 2026-07-13 — Step-wise evaluation semantics

- LLM step-wise evaluation is opt-in through `--llm-evaluation`. The default
  `status: "not_run"` and `evaluation: null` represent a disabled component,
  not an incorrect sample.
- The local artifact uses the released evaluator semantics: Formula,
  Extraction, and Calculation are LLM judgments; Answer uses an LLM parsing
  request followed by the rule-based evaluator. This hybrid A stage is a
  repository behavior that differs from a simplistic description of four
  independent LLM judges.
- Formula judging includes the generated calculation/code text when it differs
  from the declared formula. Calculation judging receives no gold calculation
  reference and assesses arithmetic validity from the candidate alone. These
  limitations must accompany any locally reported F and C accuracy.
- `first_failed_step` follows F -> E -> C -> A. A null value is interpreted as
  “all stages correct” only when `stepwise_evaluation.status` is `success` and
  all four stage results are present; otherwise it is unavailable.
- Enabling step-wise evaluation adds four chat requests per sample. The
  GitHub Models compatibility path therefore normally uses six chat requests
  and one query-embedding request for one fully evaluated MedRaC sample.

## 2026-07-13 — Phase 5 mini experiment

- The committed comparison uses the same ten fixed rows for `Plain("cot")` and
  `MedRaC(use_rag=True)`: seven equation calculators and three rule calculators.
  The runner interleaves both methods per row so a provider/quota failure does
  not leave all baseline results separated from all MedRaC results.
- Both methods share the same `openai/gpt-4.1` generator instance, temperature
  `0.0`, seed `42`, and output-token cap. The evaluator model identifier is the
  same, but step-wise evaluation was disabled for the actual 10-row run.
- Full evaluation would add eight chat requests per paired row—four for Plain
  and four for MedRaC—or 80 extra chat requests for ten rows. The free-tier run
  therefore reports rule-based final-answer accuracy only and leaves F/E/C/A,
  strict, Conditional Correctness, and First Error Attribution unavailable.
- Metrics report both method-specific denominators and a paired denominator.
  Samples with a method failure are never silently counted as an incorrect
  answer or included in a paired accuracy denominator.
- The actual run used a 2-second safe-execution timeout at DataFrame index 20.
  Inspection showed finite arithmetic-only code. At the user's request, only
  that stored code was re-executed locally with a 5-second timeout; no model or
  embedding API was called. The repaired result and original timeout are both
  preserved in artifact provenance. No generation was repeated.

## 2026-07-13 — Paper-faithful post-hoc CC/FE evaluation

- Step-wise evaluation is performed from stored Plain CoT and MedRaC outputs
  through `scripts/evaluate_experiment_artifact.py`; it never regenerates
  answers, reruns RAG, rebuilds embeddings, or executes generated code.
- The source experiment remains immutable. A separate derived artifact records
  parent manifest/sample hashes, evaluator model and temperature, seed, prompt
  version/hash, commands, batches, per-unit attempts, and progress.
- First Error Attribution Rate follows the paper denominator `not kappa`: only
  complete strict-failure cases are eligible. All-correct cases are excluded;
  zero strict failures produce null FE rates.
- `masked_reasoning_failures` is a disclosed local diagnostic, not a named
  paper metric. It counts A-correct cases that fail strict F/E/C/A and reports
  their rate among A-correct cases.
- Evaluation checkpoints after every method/sample and stops on the first
  evaluator failure. Reusing the same output directory resumes and skips
  successful units. A failed unit may be retried only by an explicit later
  command, with every attempt retained.
- Ten paired rows require 80 evaluator chat requests. The recommended free-tier
  schedule is two five-row batches of 40 requests each in separate quota
  windows, using the same evaluator model and settings.

## 2026-07-13 — Close Phase 5 and defer API-dependent step-wise evaluation

- Phase 5 is considered complete for the locally reproduced, rule-evaluated
  ten-row comparison stored in `artifacts/experiments/20260713T060240915862Z/`.
- The post-hoc F/E/C/A evaluation is deliberately deferred because the current
  GitHub Models free-tier request budget is insufficient. It is future follow-up
  work, not a failed or partially fabricated Phase 5 result.
- Until the derived artifact reaches `status: "completed"`, local F/E/C/A,
  strict correctness, Conditional Correctness, First Error Attribution, and
  masked-reasoning metrics remain unavailable and must not be reported.
- Closing the current plan must not make generation, evaluation, or embedding
  requests. The saved source artifact remains immutable; a later run must use
  the documented evaluation-only runner and a separate output directory.

## 2026-07-13 — Streamlit MedRaC seminar interface

- The Streamlit application presents MedRaC RAG only. It does not run or render
  Plain CoT and is not a baseline-comparison interface.
- Replay is the default and works without credentials. It discovers demo and
  experiment directories that contain readable `manifest.json` and
  list-shaped `samples.json` files, displays each artifact's recorded status and
  provenance, and exposes only the `medrac_rag` branch of experiment artifacts.
  Missing, malformed, or non-list artifact directories are skipped with a
  warning; discovery does not silently relabel a non-completed run as completed.
- Dataset live mode is limited to one explicit row from
  `config/mini_experiment.json`. RAG is always enabled and the only execution
  mode is `safe_ast_child_process`.
- Retrieval uses the Question alone. Patient Note, Question, and the retrieved
  formula are all passed to value extraction; Question is also preserved in the
  code-generation input. The UI must not imply that Patient Note is a retrieval
  query.
- Step-wise evaluation is opt-in because it adds four chat requests. A complete
  MedRaC run with F/E/C/A normally uses six chat requests and one embedding
  request. Missing step-wise output is unavailable, not incorrect.
- Custom input is deferred. The released pipeline cannot reliably validate an
  arbitrary Question against a selected Calculator ID and custom samples lack
  the benchmark ground truth required by the evaluators.
- Live Input Sample selection is therefore manifest-controlled rather than
  editable free text. The sidebar may select any row declared in
  `config/mini_experiment.json`; Patient Note and Question are then loaded
  verbatim from the identity-checked `data/test_data.csv` row. Adding a new
  seminar sample requires updating the manifest's DataFrame index, row number,
  Calculator ID, family, and `expected_sample_count`, without modifying the
  benchmark CSV.
- Streamlit live results are locally reproduced seminar artifacts, never
  paper-reported results or clinical outputs. API clients and credentials are
  not cached, displayed, or written to artifacts.

## 2026-07-13 — Final Streamlit review boundary

- The final review found no blocking correctness, safety, or seminar-usability
  issue. The three-field input contract, Question-only retrieval query,
  note-plus-question extraction, safe-executor-only live path, explicit API
  action, replay-without-credentials path, and MedRaC-only downloads are covered
  by code inspection, unit tests, Streamlit AppTest, and a local browser smoke.
- The review made no generation, evaluation, or embedding request and did not
  alter benchmark data, formulas, indexes, or existing generated artifacts.
- Long metric values can be visually truncated by Streamlit cards at narrower
  viewports. This is accepted as a non-blocking presentation limitation because
  the complete Calculator ID/name and clinical text remain available in the
  selectors and dedicated detail fields.
