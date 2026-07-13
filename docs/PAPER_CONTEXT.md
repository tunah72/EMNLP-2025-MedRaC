# Paper Context

## Research problem

Medical-calculation benchmarks often judge only the final value. A numerically
acceptable answer can still be clinically unreliable if the model chose the
wrong formula, extracted the wrong patient facts, or used invalid arithmetic.
This project therefore treats the released code as a demonstration of
process-level evaluation, not as a clinical decision system.

## Four-stage evaluation

Each case is decomposed into:

1. **Formula selection (F):** the applicable equation or scoring rule.
2. **Value extraction (E):** the required numerical and categorical patient
   values.
3. **Calculation (C):** the mathematical or rule-based computation.
4. **Final answer (A):** the reported result in an acceptable form.

Strict correctness requires every stage to pass:

`κ = V(F) AND V(E) AND V(C) AND V(A)`

Conditional Correctness for a stage is its correctness rate among cases where
all preceding stages were correct. First Error Attribution assigns a failed
case to the earliest incorrect stage in F -> E -> C -> A order. Both metrics
require complete, consistently parsed step-level outputs; they must not be
inferred when those outputs are absent.

Paper-faithful First Error Attribution Rate is conditioned on strict failure
(`not kappa`): its denominator is the number of complete evaluations with at
least one incorrect F/E/C/A stage, not all evaluated cases. Consequently the
four FE rates sum to one when strict failures exist and remain null when every
case is strict-correct. The seminar metrics also report the derived diagnostic
`masked_reasoning_failures`: A is correct while strict F/E/C/A is false.

### Mapping to the released evaluator

In the seminar artifact, these stages appear under `stepwise_evaluation` only
when `--llm-evaluation` is enabled. `status: "not_run"` with `evaluation: null`
means no step judge was requested; it does not mean the sample failed.

The released implementation is partly hybrid rather than four independent LLM
verdicts. Formula, Extraction, and Calculation are LLM judgments. For Answer,
an LLM first parses the generated answer and the repository's `RegEvaluator`
then determines correctness. Formula judging may include generated calculation
text, while Calculation judging intentionally receives no gold calculation
reference. These implementation details must be disclosed when interpreting
local F/E/C/A results.

`first_failed_step` is derivable only from a completed, parseable step-wise
evaluation and follows F -> E -> C -> A order. A null value is ambiguous unless
the accompanying status is checked: it can mean either all enabled stages were
correct or step-wise evaluation was not run.

## Error taxonomy

The paper analyzes formula misselection or hallucination, incorrect variable
extraction, rule-based clinical misinterpretation, missing variables,
demographic or adjustment-coefficient errors, equation-based unit conversion,
arithmetic errors, and rounding or precision errors. Labels may overlap and are
LLM-aided judgments rather than clinical ground truth.

## Calculator families

Equation-based calculators use explicit mathematical transformations and cover
the repository categories `lab test`, `physical`, `date`, and
`dosage conversion`. Rule-based calculators assign points or categories from
clinical criteria and cover `diagnosis`, `risk`, and `severity`. The distinction
matters because unit/rounding errors primarily apply to equations, while
clinical interpretation primarily applies to rules.

## MedRaC architecture

MedRaC is training-free. It retrieves a formula from a FAISS-backed knowledge
base, asks an LLM to extract required values, asks an LLM to generate Python,
executes that Python, and evaluates the result numerically and step by step.
Formula retrieval targets formula errors; executable calculation targets
arithmetic errors. The Plain zero-shot CoT method is the intended baseline.

## Limitations

- The paper reports a manually cleaned 940-case cohort, but the released CSV
  contains 1,048 rows and the runnable path does not reproduce that cohort.
- LLM judges can be inconsistent and do not replace expert medical review.
- The released rule evaluator and step-wise answer evaluator use different
  numeric tolerances.
- The checked-in FAISS docstore requires trusted pickle deserialization.
- The released `exec` path is not a secure sandbox and must not execute
  untrusted generated code in the seminar workflow.
- Small local experiments cannot reproduce every paper result or establish
  statistical significance.

Implementation evidence and discrepancies are recorded in
`docs/REPOSITORY_AUDIT.md`. Paper-reported, repository-reported, locally
reproduced, and inferred-but-unverified statements must remain visibly
separate.
