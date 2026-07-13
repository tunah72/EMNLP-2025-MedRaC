# Phase 1 Repository Audit

Audit target: commit `900c54123ce03ca467ebdf0b2f28afb7dcaabc5e` on branch
`seminar/demo-pipeline`.

Audit environment: macOS 26.5.2, Apple ARM64, Python 3.13.5. This document
describes the checked-in implementation. It does not claim that the paper's
reported experiments were reproduced locally.

## Executive summary

The repository contains the major components described in the paper: Plain
prompting, formula retrieval, structured value extraction, LLM-generated
Python, rule-based final-answer evaluation, four-field LLM evaluation, and an
eight-label error analysis. The released code is nevertheless not a turnkey
reproduction of the paper.

The most important findings are:

1. **Blocking — generated code is executed with unrestricted `exec`.** The
   method calls this a sandbox, but it runs model-produced Python in the main
   process with builtins automatically available and no timeout. It must not be
   used as the seminar safety boundary (`method/medRaC.py:365-390`).
2. **Blocking — the default demo cannot run in the audited environment.** The
   installed Python lacks `tiktoken`, `langchain`, `langchain_community`,
   `langchain_openai`, and `faiss`. The original requirements also include
   `vllm`, although the API path does not need a local vLLM server
   (`requirements.txt:1-13`, `model/gpt.py:5-16`).
3. **Blocking — the checked-in RAG metadata is deserialized with pickle.**
   `FAISS.load_local(..., allow_dangerous_deserialization=True)` loads
   `index.pkl`; this is code-execution-sensitive and is safe only under an
   explicit trust decision (`method/rag.py:48-60`). Static inspection found
   LangChain `InMemoryDocstore` and `Document` constructors. The files were not
   deserialized during this audit.
4. **Important — paper filtering is not part of the released run path.** The
   checked-in CSV has 1,048 rows, while the paper reports removing 108 and
   evaluating 940. `run.py` calls the normal loaders directly, and the cleaning
   helper operates only on generated JSON and would overwrite its inputs
   (`run.py:57-66`, `method/method.py:59-125`,
   `utils/remove_bad_data.py:12-71,79-138`).
5. **Important — evaluation and error-analysis costs are much larger than one
   apparent batch call.** `APIModel.generate` creates one asynchronous API task
   for every prompt, so a list of prompts is still one network request per
   sample and per judged field (`model/gpt.py:253-288`).
6. **Important — the current MedRaC output is not fully transparent.** RAG
   returns `(formula, score)`, but MedRaC stores only the formula text; code-stage
   tokens are accumulated globally and are not written per sample
   (`method/rag.py:123-147`, `method/medRaC.py:74-83,251-261`).

The implementation is therefore suitable as source material for a seminar
demo, but the released `run.py` should not be presented as a secure,
deterministic, or paper-exact reproduction.

## Real call graphs

### Plain zero-shot CoT

```text
run.py
  -> APIModel(OpenAI/gpt-4o-mini) and APIModel(DeepSeek/deepseek-chat)
  -> Plain("cot", [generator], [RegEvaluator, LLM_Evaluator])
  -> Plain.generate_raw(test=True)
       -> Method.load_data_test()
            -> pandas.read_csv()
            -> DataFrame.sample(n=2) when no fixed rows/calculator are supplied
       -> Method.cot() -> list[(system_message, user_message)]
       -> prompt_style_to_schema("cot") -> CoTOutput
       -> APIModel.generate() -> one chat completion per row
       -> Method._build_records()
       -> Method._dump_json()
  -> Plain.evaluate(raw_path)
       -> RegEvaluator.check_correctness()
       -> LLM_Evaluator.check_correctness()
       -> Method._dump_json()
  -> RegEvaluator.compute_overall_accuracy_new()
  -> error_type_pipeline()
```

Evidence: `run.py:28-44,54-67,100-101`, `method/plain.py:64-139,144-223`,
`method/method.py:90-125,192-222,366-441`, and
`schema/schemas.py:9-31`.

`test=True` does **not** mean a stable test fixture. In the default construction,
`load_data_test()` reaches `df.sample(n=2)` without a `random_state`, so each run
can select different rows (`method/method.py:105-125`).

### MedRaC with RAG

```text
MedRaC(..., model=code_generator, use_rag=True)
  -> RAG()
       -> OpenAIEmbeddings(text-embedding-ada-002)
       -> FAISS.load_local(index.faiss + index.pkl)
  -> MedRaC.generate_raw()
       -> load dataset
       -> for each question:
            RAG.retrieve(question, k=1)
              -> embedding API query
              -> FAISS similarity search
            keep formula text; discard retrieval score
       -> MedRaC._gen_extracted_values()
       -> generator.generate(..., schema=Values) [one chat request/sample]
       -> MedRaC.generate_code()
            -> _gen_code_prompt()
            -> code model.generate() [one chat request/sample]
            -> _run_responses()
                 -> ThreadPoolExecutor
                 -> execute_code() -> unrestricted exec
       -> build {formula, extracted_values, calculation/code, answer}
       -> write raw JSON
  -> MedRaC.evaluate()
       -> same evaluator loop as Plain, with canonical formulas supplied
       -> write eval JSON
```

Evidence: `method/medRaC.py:29-50,51-130,215-261,265-355,365-421,452-478`,
`method/rag.py:27-60,123-147`, and `schema/schemas.py:90-103`.

For the RAG path, the `formula` field in the model output is retrieved text, not
an LLM-selected formula. The extraction response contains reason fields in the
schema, but `self.responses` retains only `extracted_values`; the extraction
reason is dropped (`method/medRaC.py:74-83,110-113`,
`schema/schemas.py:96-98`).

### Rule-based final-answer evaluation

```text
RegEvaluator.check_correctness()
  -> _extract_answer()
       -> JSON answer field, regex fallback, calculator-specific parsing
       -> expression eval for several numeric forms
  -> calculator-ID-specific comparison
       -> exact date / exact weeks-days
       -> rounded equality for integer scores
       -> checked-in lower/upper range for decimal calculators
  -> "Result": Correct | Incorrect
```

Evidence: `evaluator/regEvaluator.py:23-170,173-234`.

This is the evaluator used by `Plain.evaluate` and `MedRaC.evaluate`. It is not
the paper's stricter decimal-precision rule: the main checker uses the CSV's
lower and upper limits (`evaluator/regEvaluator.py:223-226`). A separate
`check_correctness_parsed` implements response-precision tolerance for the LLM
evaluator's answer stage (`evaluator/regEvaluator.py:240-326`).

### LLM step-wise evaluation

```text
LLM_Evaluator.check_correctness()
  -> normalize every response to formula/extracted_values/calculation/answer
  -> judge formula       [one chat request/sample]
  -> judge extraction    [one chat request/sample]
  -> judge calculation   [one chat request/sample]
  -> parse final answer  [one chat request/sample]
  -> RegEvaluator.check_correctness_parsed(parsed answers)
  -> return one structured F/E/C/A object per sample
```

Evidence: `evaluator/llmEvaluator.py:23-140,144-199,325-354` and
`schema/schemas.py:59-82`.

Formula judging concatenates the declared formula and calculation/code when
they differ. Calculation judging intentionally receives no reference and asks
the judge only whether the arithmetic is internally valid
(`evaluator/llmEvaluator.py:64-87,144-162`). The final-answer stage is not an
LLM correctness judgment: the LLM only extracts a normalized answer, which is
then checked by `check_correctness_parsed`
(`evaluator/llmEvaluator.py:100-126`).

### Strict, conditional, first-error, and error taxonomy analysis

`compute_overall_accuracy_new` reads the four evaluation fields in F, E, C, A
order. It computes strict correctness with `all(bools)`, attributes the first
false field, and counts failures conditioned on all prior fields being correct
(`evaluator/baseEvaluator.py:100-167`). It reports **conditional error rates**,
not a clearly named Conditional Correctness metric, and does not expose a
standalone first-error distribution in the output even though it collects the
counts (`evaluator/baseEvaluator.py:191-210,282-301`).

The error taxonomy path is:

```text
error_type_pipeline(eval JSON)
  -> overwrite requested model_name with the generator model from the JSON
  -> build formula, extraction, missing-variable, and arithmetic prompts
  -> add clinical-misinterpretation prompts only for rule-based categories
  -> add unit, adjustment, and rounding prompts only for equation categories
  -> concatenate prompts
  -> APIModel.generate(all_prompts) [still one API request per prompt]
  -> split replies by remembered offsets
  -> add eight error/error-explanation column pairs
  -> write <generator_model>_error_eval.json
```

Evidence: `utils/error_type.py:141-173,175-265,267-328` and
`model/gpt.py:253-288`.

## Schemas and data flow

| Stage | Input | Output |
|---|---|---|
| Dataset | `data/test_data.csv` | 1,048 rows; 14 columns: row/calculator metadata, note/question, relevant entities, answer/range, explanation |
| Formula library | `data/formula_new.json` | List of 55 `{Calculator ID, Question, Formula}` objects |
| RAG corpus | `data/web_formula.txt` | 785 blocks delimited by `<<FORMULA START>>` / `<<FORMULA END>>` |
| Plain CoT | calculator ID, note, question | `{step_by_step_thinking: str, answer: str}` (`schema/schemas.py:27-31`) |
| RAG extraction | note, question, retrieved formula | `{extracted_values_reason: str, extracted_values: [KeyValue]}` (`schema/schemas.py:5-7,96-98`) |
| Code generation | formula, extracted values, question | Free text expected to be fenced Python assigning `result` (`method/medRaC.py:265-355`) |
| MedRaC sample | retrieved/generated fields and execution result | `{formula, extracted_values, calculation, answer}` (`method/medRaC.py:110-113`) |
| Raw record | dataset row + model output/tokens | Fields assembled by `Method._build_records` (`method/method.py:380-434`) |
| Rule evaluation | response, truth, calculator ID, limits | `Result: Correct | Incorrect` (`evaluator/regEvaluator.py:173-234`) |
| Step evaluation | response plus canonical references | `LLM Evaluation` with four `{result, explanation}` objects (`evaluator/llmEvaluator.py:128-140`) |
| Error analysis | evaluated records | Eight label columns plus eight explanation columns (`utils/error_type.py:280-327`) |

The CSV contains 55 calculator IDs. Representative checked rows include an
equation calculator at zero-based index 0 / Row Number 1 / Calculator ID 2
(Cockcroft-Gault) and a rule calculator at index 40 / Row Number 41 /
Calculator ID 4 (CHA2DS2-VASc). Relevant entities are serialized Python-dict
strings and are converted using JSON first, then `ast.literal_eval`
(`method/method.py:29-55,73-78`).

## API calls and cost-driving operations

`APIModel.generate` schedules `_generate_single` once for every prompt
(`model/gpt.py:253-288`). Therefore “one generate” is not one provider request.
The following counts assume no retries or structured-output fallback:

| Pipeline | Equation sample | Rule sample | Notes |
|---|---:|---:|---|
| Plain CoT generation only | 1 chat | 1 chat | One structured completion |
| Plain + full F/E/C/A evaluation | 5 chat | 5 chat | Generation + 3 step judges + answer parser |
| MedRaC RAG generation only | 2 chat + 1 embedding | 2 chat + 1 embedding | Extraction + code generation + query embedding |
| MedRaC + full F/E/C/A evaluation | 6 chat + 1 embedding | 6 chat + 1 embedding | MedRaC generation plus four evaluation requests |
| Error taxonomy on an existing output | 7 chat | 5 chat | Equation: formula, extraction, missing, unit, adjustment, arithmetic, rounding; rule substitutes clinical interpretation and omits unit/adjustment/rounding |

For four samples, multiply by the category mix. Four equation samples cost 20
chat requests for Plain with evaluation, or 24 chat plus four embedding queries
for MedRaC with evaluation. Four rule samples have the same generation and
step-evaluation counts; error analysis would be 20 rather than 28 requests.

For the proposed 12-sample mini experiment (eight equation, four rule), with
both Plain and MedRaC and F/E/C/A evaluation enabled:

- Plain: `12 x 5 = 60` chat requests.
- MedRaC: `12 x 6 = 72` chat requests and 12 embedding queries.
- Total: **132 chat requests plus 12 embedding queries** before retries.
- Error taxonomy for both methods would add
  `2 x (8 x 7 + 4 x 5) = 152` chat requests.

Structured-output failures may add another provider call because the OpenAI
wrapper retries with an unstructured chat completion
(`model/gpt.py:194-228`). DeepSeek failures retry up to three times
(`model/gpt.py:167-192`). Token cost is driven primarily by long patient notes,
ground-truth explanations, formula text, code prompts, and repeated judge/error
prompts. The code records generation tokens per row but not embedding usage and
does not persist MedRaC's code-generation tokens per row
(`method/method.py:380-385`, `method/medRaC.py:255-258`).

## Files written by the released workflow

| Caller | Default output |
|---|---|
| `Plain.generate_raw` | `raw_output/prompts.json`; `raw_output/<model>_<style>_raw.json` (`method/plain.py:92-127`) |
| `Plain.evaluate` | Writes `<raw-file-prefix>_eval.json` next to the raw file; it creates but otherwise ignores `eval_json_dir` (`method/plain.py:209-220`) |
| `MedRaC.generate_raw` | `raw_output/code/<model>_<style>_raw.json` (`method/medRaC.py:51-54,115-122`) |
| `MedRaC.evaluate` | `eval_output/code/<model>_<style>_eval.json` (`method/medRaC.py:203-212`) |
| Overall statistics | `stats/results_<eval-file>.json` (`evaluator/baseEvaluator.py:304-313`) |
| Error analysis | `ErrorTypes/<generator-model>_error_eval.json` (`utils/error_type.py:320-328`) |
| RAG rebuild fallback | Overwrites/creates `index.faiss` and `index.pkl` under the embeddings directory (`method/rag.py:107-118`) |

The checked-in workflow has no run ID, manifest, command capture, Git SHA,
dependency snapshot, or artifact-level separation between paper, repository,
and local results.

## Nondeterminism

- Default test selection uses unseeded `DataFrame.sample`.
- OpenAI generation in `run.py` uses temperature 1.0; no seed is passed through
  `APIModel` (`run.py:36-41`, `model/gpt.py:19-33,154-163`).
- Error analysis uses temperature 0.1 (`utils/error_type.py:167-173`).
- Provider model revisions, structured-output parsing, retries, and asynchronous
  request completion can change results.
- RAG depends on the remote query embedding model and the exact serialized
  index/docstore versions.
- Threaded code execution completes out of order but is reassembled by index;
  generated programs can still contain their own nondeterminism
  (`method/medRaC.py:392-421`).
- Output filenames are reused, so later runs can overwrite prior results.

## Findings by severity

### Blocking for the seminar demo

1. **Unsafe execution boundary.** Unrestricted model-generated code can import,
   read/write files, access the network, spawn processes, or hang. Empty globals
   do not remove Python builtins (`method/medRaC.py:365-390`).
2. **Untrusted-pickle-sensitive index loading.** The code explicitly opts into
   dangerous deserialization (`method/rag.py:48-60`). The audited files have
   SHA-256 hashes:
   - `index.faiss`: `ef8b80ab8566e28b73a78536ec373f767a9bbbb8cde4465c310ad8c080328a9a`
   - `index.pkl`: `fd4de9d5ba416f2004d6c35a6408ed3c12abdb0903c7dbf36b6f44469179dbd8`
3. **Current environment is incomplete.** API/RAG imports cannot succeed with
   the currently installed modules. The index therefore has not yet been
   runtime-loaded against a compatible dependency set.
4. **`run.py` eagerly initializes a DeepSeek evaluator.** Despite the comment
   saying GPT is the accessible default, the active evaluator is DeepSeek and
   its constructor loads a Hugging Face tokenizer, which can require credentials
   and network/cache availability (`run.py:28-44`, `model/gpt.py:73-80`).

### Important but non-blocking

1. The paper's cleaned 940-row cohort is not checked in or automatically
   selected; current runs use 1,048 rows unless callers pre-filter them.
2. `error_type_pipeline(model_name=...)` discards its argument and replaces it
   with the generator model from the input, so the requested error judge can be
   silently ignored (`utils/error_type.py:141,156-173`).
3. RAG scores are discarded by MedRaC, preventing the requested transparent
   retrieval report (`method/medRaC.py:74-76`).
4. `RAG.retrieve` correctly describes higher relevance scores, but reproducible
   validation requires the same embedding model used to build the index
   (`method/rag.py:123-147`). That provenance is not stored in a manifest.
5. `APIModel` does not pass its inherited `seed`, `top_p`, or max-token controls
   from a run configuration to provider requests (`model/model.py:14-42`,
   `model/gpt.py:154-163`).
6. The OpenAI generic exception path refers to `_RawWrapper`, which is declared
   only inside the preceding `LengthFinishReasonError` handler. A non-length
   parse failure can therefore raise `UnboundLocalError` instead of returning
   raw text (`model/gpt.py:206-228`).
7. The main rule evaluator uses multiple `eval` calls on parsed response or CSV
   strings (`evaluator/regEvaluator.py:111-153,207-226`). Its inputs are more
   constrained than generated code, but it remains an avoidable execution risk.
8. The rule evaluator and the LLM answer-stage evaluator use different numeric
   policies: checked-in ±5% ranges versus response-precision tolerance. A report
   must label them distinctly (`evaluator/regEvaluator.py:223-226,294-318`).
9. Conditional bookkeeping reports conditional **error** percentages, while the
   paper defines Conditional Correctness. First-error counts are computed but
   not emitted as a dedicated metric (`evaluator/baseEvaluator.py:152-167,191-210`).
10. README says the repository can reproduce every experiment, but it provides
    only `python run.py`, no deterministic row list, cleaned cohort, environment
    lock, run manifest, or published result artifacts (`README.md:2-3,21-56`).
11. `Plain.evaluate` creates `eval_json_dir` but writes beside the raw input,
    making output routing misleading (`method/plain.py:209-220`).
12. MedRaC's `prompt_style` concatenates the model name without a separator and
    is later included in filenames; this is confusing provenance rather than a
    functional method identifier (`method/medRaC.py:39`).

### Cosmetic

1. README references `method/codeGenerator.py`, but the repository contains
   `codeGeneratorNoCode.py` (`README.md:13`).
2. The README clone URL points to `Super-Billy` while the requested repository
   is `benluwang` (`README.md:26`).
3. `run.py` labels the commented TwoAgent example as “RAG Method Example” even
   though the imported standalone `RAG` class is not used there
   (`run.py:73-82`).
4. Several docstrings/comments call unsafe `exec` a sandbox or say concatenated
   prompts are sent “once”; both are misleading wording
   (`method/medRaC.py:365-368`, `utils/error_type.py:141-147,264-265`).

No cosmetic issue above was changed during Phase 1.

## Paper versus released implementation

| Paper description | Released implementation | Audit verdict |
|---|---|---|
| Four independently evaluated F/E/C/A stages | Three LLM judgments plus LLM answer extraction followed by a rule checker | Partially supported |
| Strict case correctness is conjunction of all stages | `all(bools)` is implemented in statistics aggregation | Supported, but only after LLM evaluation |
| Conditional Correctness and First Error Attribution | Conditional failure counters and first-failure counters exist; reporting/naming is incomplete | Partially supported |
| Cleaned 940-case dataset after removing 108 | Checked-in CSV and default loader contain 1,048 rows | Not supported by the runnable release path |
| Formula RAG injects retrieved formula | Checked-in FAISS retrieval feeds extraction and code generation | Supported, subject to index trust/dependencies |
| Python execution reduces arithmetic error | Generated code is executed and its `result` becomes the answer | Supported functionally; unsafe operationally |
| Equation and rule calculators are reported separately | Statistics aggregate categories into equation/rule buckets | Supported (`evaluator/baseEvaluator.py:213-266`) |
| DeepSeek-chat evaluates steps; DeepSeek-reasoner labels errors | Default `run.py` uses DeepSeek-chat for steps but calls error analysis with a GPT name that is then overwritten by the generator model | Not reliably supported |
| Complete reproducibility | No deterministic cohort, environment lock, run manifest, or checked-in results | Not supported |

The paper states that formula, extraction, calculation, and answer are evaluated
sequentially, defines strict/conditional/first-error metrics, and reports a
940-case filtered cohort. See Sections 3.1 and 4 and Appendix A of
<https://arxiv.org/abs/2509.16584>. These statements are used here only as the
comparison target; no paper result is presented as locally reproduced.

## Phase 1 verification performed

- Inspected every file required by `AGENTS.md`, including representative CSV
  rows, all 55 canonical formulas, the 785-block RAG corpus, and both FAISS
  artifacts.
- Confirmed CSV size (1,048 rows), 55 calculator IDs, formula library size
  (55), and corpus block count (785).
- Performed static pickle opcode inspection only; no deserialization occurred.
- Probed the active Python environment and recorded missing imports.
- Ran `python3 -m compileall -q -f run.py method model evaluator schema utils`;
  it completed successfully.
- Made no API calls, installed no packages, loaded no serialized index, and ran
  no generated code.

## Phase 1 conclusion

The released implementation substantially supports the paper's architectural
story, but the claims of complete reproducibility and safe executable-code use
are not supported by the audited release. Phase 2/3 should first establish
persistent decisions and a minimal API environment; Phase 4 must add a separate
safe execution path and deterministic, score-preserving demo orchestration
instead of broad refactoring of `run.py`.
