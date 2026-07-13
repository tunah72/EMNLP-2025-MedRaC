You are acting as an AI research engineer helping me prepare a code demonstration for a Deep Learning seminar.

Repository:
https://github.com/benluwang/EMNLP-2025-MedRaC

Paper:
“From Scores to Steps: Diagnosing and Improving LLM Performance in Evidence-Based Medical Calculations”
https://arxiv.org/abs/2509.16584

Read `AGENTS.md` first and treat it as binding project guidance.

## Research context

The paper argues that final-answer accuracy alone is insufficient for evaluating LLMs on medical calculations. A model may obtain a numerically acceptable answer despite using an incorrect medical formula, extracting the wrong patient values, or making invalid intermediate steps.

The evaluation pipeline separates four stages:

1. Formula selection, F
2. Value extraction, E
3. Mathematical calculation, C
4. Final answer, A

Strict case correctness is:

κ = V(F) AND V(E) AND V(C) AND V(A)

The paper additionally studies Conditional Correctness, First Error Attribution, and structured error categories such as formula error, extraction error, clinical misinterpretation, missing variables, demographic adjustment error, unit conversion error, arithmetic error, and rounding error.

MedRaC is a training-free pipeline combining:

1. Retrieval of a medical formula from a formula knowledge base
2. Extraction of patient values required by that formula
3. LLM generation of Python calculation code
4. Execution of that code
5. Numerical and step-wise evaluation

The purpose of this project is not to reproduce every number in the paper. The immediate objective is to create:

1. A transparent pipeline demo suitable for a seminar
2. A small, reproducible comparison between a baseline and MedRaC
3. Documentation explaining how the released code maps to the paper

## Required working behavior

Do not begin with broad refactoring.

First inspect the repository and verify the actual implementation. Do not assume that the README, paper, and code are perfectly consistent.

Before editing code, inspect at least:

* `README.md`
* `run.py`
* `requirements.txt`
* `method/method.py`
* `method/plain.py`
* `method/medRaC.py`
* `method/rag.py`
* `model/model.py`
* `model/gpt.py`
* `evaluator/regEvaluator.py`
* `evaluator/llmEvaluator.py`
* `utils/error_type.py`
* `schema/schemas.py`
* the schema and representative rows of `data/test_data.csv`
* `data/formula_new.json`
* `data/web_formula.txt`

Also inspect the checked-in FAISS index and determine whether it can be loaded with the current dependency versions.

## Phase 1: Repository audit

Produce a concise audit containing:

1. The real end-to-end call graph for:

   * Plain CoT
   * MedRaC with RAG
   * rule-based evaluation
   * LLM step-wise evaluation
   * error-type analysis
2. Input and output schemas at every important stage
3. All API calls made per sample
4. Files and directories written during a run
5. Current sources of nondeterminism
6. Platform and dependency risks
7. Bugs, inconsistencies, or misleading comments that could affect the demo
8. Differences between the paper description and released implementation
9. An estimate of API calls and cost-driving operations for:

   * one sample,
   * four samples,
   * the proposed mini experiment

For every important claim, cite the relevant repository path and symbol or line range.

Classify findings as:

* blocking,
* important but non-blocking,
* cosmetic.

Do not fix unrelated cosmetic issues during this phase.

## Phase 2: Persistent project documentation

Create or update:

* `docs/PAPER_CONTEXT.md`
* `docs/EXPERIMENT_PLAN.md`
* `docs/DECISIONS.md`
* `docs/RUNBOOK.md`
* `.env.example`

`PAPER_CONTEXT.md` must explain:

* the research problem,
* Formula–Extraction–Calculation–Answer decomposition,
* strict overall correctness,
* Conditional Correctness,
* First Error Attribution,
* error taxonomy,
* MedRaC architecture,
* equation-based versus rule-based calculators,
* important limitations.

`DECISIONS.md` must record all deviations from the original paper, including model substitutions, reduced sample size, evaluator substitutions, temperature changes, or disabled components.

Never include actual API keys.

## Phase 3: Environment setup

Determine the current operating system and Python version.

Do not blindly install every dependency.

The repository pins `vllm`, but the initial demo should use API-based models and should not require a local vLLM server.

Create a minimal and reproducible API-demo dependency path, preferably one of:

* `requirements-demo.txt`, or
* a documented set of optional dependency groups.

Do not replace the original `requirements.txt`.

Verify:

1. Imports required for the API demo
2. FAISS CPU installation
3. Loading of the checked-in index
4. Dataset loading
5. Formula lookup
6. Model initialization without printing secrets

Clearly report anything that cannot run on the current platform.

## Phase 4: Build a transparent pipeline demo

Prefer adding `scripts/demo_pipeline.py` instead of rewriting the original `run.py`.

The script should expose a small CLI and support:

* fixed row indices,
* fixed calculator IDs where practical,
* generator model selection,
* evaluator model selection,
* RAG enabled or disabled,
* LLM evaluation enabled or disabled,
* output directory,
* random seed,
* dry-run mode.

The demo must use fixed, verified dataset rows rather than random sampling.

Select a very small representative set only after inspecting the dataset:

* at least one equation-based calculator,
* optionally one rule-based calculator,
* no invented calculator IDs.

For each sample, save or display these stages:

1. Patient note and question
2. Retrieved formula and retrieval score
3. Extracted values
4. Generated calculation code
5. Execution result
6. Final answer
7. Rule-based evaluation
8. Step-wise evaluation, when enabled
9. First failed step, if derivable from the evaluator output

Generate machine-readable JSON and a concise Markdown report.

Do not expose hidden model reasoning. Store only the structured reasoning fields already required by the repository’s public output schema.

## Code-execution safety

Inspect the current use of `exec`.

The seminar demo must not claim that the current implementation is a secure sandbox.

Add a minimal seminar-specific safety layer without silently changing the scientific method. Prefer:

* AST validation permitting only the arithmetic and library operations needed by the calculators,
* execution in a child process,
* a strict timeout,
* no filesystem, network, shell, dynamic import, or user-input access.

Keep the original implementation available for fidelity comparison and document which execution mode was used.

Do not run arbitrary code from untrusted sources.

## Phase 5: Mini experiment

Create `scripts/run_mini_experiment.py`.

Default experiment:

* Baseline: `Plain("cot")`
* Proposed method: `MedRaC(use_rag=True)`
* Same fixed samples for both methods
* Same generation model
* Same evaluation model
* Same generation parameters
* Approximately 10–20 samples, subject to API budget
* Prefer a mixture of equation-based tasks and a small number of rule-based tasks
* Deterministic row list committed in a config or manifest

Use the repository-compatible low-cost API model for the first run unless an exact paper model is available.

For reproducibility, record:

* Git SHA
* selected row indices
* calculator IDs
* model identifiers
* embedding model
* temperatures
* seed
* start and end timestamps
* dependency versions
* enabled pipeline components
* commands
* output paths

At minimum report:

* final-answer accuracy from the rule-based evaluator,
* Formula accuracy,
* Extraction accuracy,
* Calculation accuracy,
* Answer accuracy from step-wise evaluation when enabled,
* strict all-steps-correct accuracy,
* token usage,
* number of failed or skipped samples.

Compute Conditional Correctness and First Error Attribution only when the available outputs are sufficient. Do not fabricate them from missing information.

Clearly separate:

* paper-reported results,
* repository-provided results,
* locally reproduced results.

Do not claim statistical significance for this small experiment.

## Phase 6: Tests and verification

Add focused tests using the standard library where practical.

Tests should cover at least:

* deterministic dataset selection,
* formula lookup by calculator ID,
* parsing structured model output,
* code-fence removal,
* safe arithmetic execution,
* timeout or rejected unsafe code,
* experiment manifest generation,
* aggregation of step-wise correctness.

Use mocked model outputs for tests that do not require an API.

Run:

* syntax or compile checks,
* focused unit tests,
* a dry run,
* one-sample smoke test when credentials are available,
* the mini experiment only after the smoke test succeeds.

Do not claim API-dependent verification if credentials or network access are unavailable.

## Expected artifacts

Produce:

* `AGENTS.md`, preserving existing valid instructions
* `docs/PAPER_CONTEXT.md`
* `docs/EXPERIMENT_PLAN.md`
* `docs/DECISIONS.md`
* `docs/RUNBOOK.md`
* `.env.example`
* minimal demo dependency specification
* `scripts/demo_pipeline.py`
* `scripts/run_mini_experiment.py`
* focused tests
* a deterministic experiment configuration or manifest
* `artifacts/demo/<run-id>/...`
* `artifacts/experiments/<run-id>/...`

Do not commit generated artifacts containing sensitive information or excessively large raw outputs unless explicitly required.

## Definition of done

This task is complete only when:

1. The released implementation has been audited and mapped to the paper.
2. Setup instructions work on the current environment or blockers are documented.
3. The demo runs on at least one fixed sample, when API credentials are available.
4. The output visibly demonstrates Formula → Extraction → Calculation → Answer.
5. Baseline and MedRaC can run on the same fixed sample list.
6. Metrics are generated from actual outputs.
7. Tests and verification commands are recorded.
8. All deviations from the paper are disclosed.
9. The final report lists files changed, commands executed, successful checks, failed checks, artifact paths, and unresolved risks.

Proceed in phases.

Start now with Phase 1. After the audit, continue with the minimal setup and implementation unless you encounter one of these decision points:

* modifying benchmark data or medical formulas,
* running the full dataset,
* incurring substantial API cost,
* replacing a central evaluator or model in a way that changes the scientific question,
* loading an untrusted serialized index,
* committing secrets.

For routine engineering choices, make the most conservative, well-grounded decision and document it instead of stopping unnecessarily.
