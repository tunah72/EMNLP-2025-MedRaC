# Simplified Experiment Plan

## Research question

On a small, fixed mixture of equation-based and rule-based medical calculators,
does the released MedRaC decomposition improve final-answer and strict
step-wise correctness relative to zero-shot Chain-of-Thought under otherwise
matched generation and evaluation settings?

This is a seminar-scale engineering comparison. It is not powered to establish
statistical significance.

## Methods and controls

- Baseline: `Plain("cot")`.
- Proposed method: `MedRaC(use_rag=True)`.
- Both methods receive identical ordered row indices, generator model,
  evaluator model, generation parameters, and evaluator configuration.
- Compatibility default: GitHub Models free tier with `openai/gpt-4.1` for both
  generation and evaluation. Catalog availability must be confirmed before an
  API run; the paper-faithful DeepSeek evaluator remains optional.
- Compatibility embedding model: `openai/text-embedding-3-small`, using a new
  separate index over the 55 canonical formulas. The original checked-in index
  remains unchanged and is never queried with this embedding model.

## Sample-selection policy

The Phase 5 manifest is committed at `config/mini_experiment.json`. It contains
10 retained rows in deterministic order: DataFrame indices
`[0, 40, 20, 120, 60, 200, 80, 100, 160, 280]`. The set contains seven
equation-based and three rule-based calculators across ten distinct calculator
IDs. Each entry records CSV Row Number, calculator ID, family, and inclusion
rationale. The runner verifies these identities against the source CSV and the
940-row exclusion policy before any API call.

`--limit N` selects only the first N committed entries for smoke testing; it
does not resample or reorder them. Do not use `test=True`, random sampling,
excluded rows, invented IDs, or modify the CSV.

## Metrics

- Rule-based evaluator final-answer accuracy.
- Formula, Extraction, Calculation, and Answer accuracy from complete step-wise
  evaluation.
- Strict accuracy: proportion with F, E, C, and A all correct.
- Conditional Correctness for each step, including numerator and denominator.
- First Error Attribution counts and rates in F -> E -> C -> A order,
  conditioned only on strict failures (`not kappa`) as defined in the paper.
- Masked reasoning failures: cases where A is correct but strict F/E/C/A is
  false, including the rate among A-correct cases.
- Input/output token usage, provider-request counts, retrieval-request counts,
  elapsed time, failed samples, and skipped samples.

Conditional and first-error metrics are reported only when all required fields
are parseable. Missing metrics are `null` with an explanation, never fabricated
as zero.

### Step-wise availability contract

- Runs without `--llm-evaluation` must store
  `stepwise_evaluation.status = "not_run"`, `evaluation = null`, and must not
  contribute to Formula, Extraction, Calculation, strict correctness,
  Conditional Correctness, or First Error Attribution metrics.
- Runs with complete evaluation store one `Correct`/`Incorrect` object for each
  of `formula`, `extracted_values`, `calculation`, and `answer`.
- `first_failed_step` is the earliest incorrect stage in F -> E -> C -> A order.
  A null value counts as “all evaluated stages correct” only when the step-wise
  status is `success` and all four stage objects are present and parseable.
- The released Answer stage is hybrid: one LLM request parses the answer and
  `RegEvaluator` assigns the correctness flag. It must not be described as an
  independent LLM answer verdict.
- Enabling full evaluation adds four chat requests per method/sample. Cost and
  free-tier quota estimates must include these calls before choosing the final
  10–20 sample manifest.
- FE uses `strict_failures = complete_evaluations - all_correct` as its
  denominator. If that value is zero, FE rates are null rather than zero.

## Cost control

Run compile/unit checks and dry-run validation first, then one API smoke sample,
then request approval before the full mini experiment. Disable error-taxonomy
analysis by default because it adds five to seven judge calls per method/sample.
Stop on unexpected retries, model substitutions, index rebuilding, or material
cost expansion. No full-dataset run is authorized.

## Failure and skip policy

- Preserve every attempted sample and stage status.
- Mark provider, parse, retrieval, execution, timeout, and evaluator failures
  separately; do not convert them silently to incorrect answers.
- A method-level failure is counted in failed/skipped totals and excluded from
  a metric only when that metric's denominator is explicitly reported.
- Do not retry indefinitely. Record retry count and terminal error.
- Use only the future safe execution mode for generated code; retain the
  original mode solely as an unexecuted fidelity reference.

## Claim boundary

A run may be called a **locally reproduced simplified comparison** only if both
methods used the same committed sample manifest and settings, artifacts came
from actual provider outputs, all enabled metrics came from those artifacts,
and deviations are recorded in `docs/DECISIONS.md`. It must not be called a
paper-result reproduction, clinical validation, or statistically significant
result.

## Phase 5 local run — 2026-07-13

The authorized free-tier run used `openai/gpt-4.1`, temperature `0.0`, seed
`42`, `openai/text-embedding-3-small`, the canonical 55-formula index, and the
safe child-process executor. Step-wise LLM evaluation was disabled to keep the
10-sample comparison within the free-tier request budget.

- Plain CoT: 9/10 rule-correct, no failed samples.
- MedRaC: 10/10 rule-correct, no remaining failed samples.
- Paired successful rows: Plain CoT 9/10 and MedRaC 10/10.
- Paired outcomes: nine both correct, one MedRaC-only correct, zero
  Plain-only correct, zero both incorrect.
- MedRaC row index 20 / Calculator ID 3 originally hit the configured 2-second
  child-process timeout. Its already stored generated code was re-executed
  locally with a 5-second timeout, without an API call, and returned
  `7.451164826774139` / `Correct`. The original timeout remains preserved in
  the artifact's post-run repair provenance.
- F/E/C/A, strict correctness, Conditional Correctness, and First Error
  Attribution are unavailable for this run because `--llm-evaluation` was not
  enabled. Their metric values remain null with zero denominators.

These results are locally reproduced seminar outputs only. The sample is too
small and incomplete for a statistical-significance claim.

Step-wise evaluation must be added as a derived, evaluation-only artifact. It
must reuse these stored generations, checkpoint after every method/sample, and
must not overwrite the parent artifact.
