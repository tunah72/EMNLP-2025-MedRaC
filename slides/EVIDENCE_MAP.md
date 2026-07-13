# MedRaC Scientific Seminar Evidence Map

Paper source: Wang et al., *From Scores to Steps: Diagnosing and Improving LLM Performance in Evidence-Based Medical Calculations*, arXiv:2509.16584v2, 31 January 2026. Audited PDF: 25 pages; SHA-256 `757b98d0881c661d0e365b12998550751b289d610bea0339281dc7609f5a0107`.

## Main deck

| Slide | Topic | Primary evidence |
|---:|---|---|
| 1 | Title, paper authors, venue, seminar and group information | Paper title page; arXiv metadata; user-provided seminar metadata |
| 2 | Presentation outline | Presentation structure |
| 3 | Evidence-based medical calculations | Paper Sections 1-2, PDF pp. 1-3; released 55-calculator formula library for examples |
| 4 | Final-answer accuracy is insufficient | Appendix G, PDF p. 17 |
| 5 | Research gap and questions | Sections 1-3, PDF pp. 1-5 |
| 6 | Main contributions | Section 1, PDF pp. 1-2 |
| 7 | F-E-C-A process | Figure 1 and Section 3.1, PDF pp. 3-4 |
| 8 | Formula selection and value extraction | Section 3.1, PDF p. 3 |
| 9 | Calculation versus final answer and precision-aware tolerance | Section 3.1 and Appendix G, PDF pp. 3 and 17 |
| 10 | Strict correctness | Section 3.1 equation, PDF p. 3 |
| 11 | Conditional Correctness | Section 3.1 equation, PDF p. 3 |
| 12 | First Error Attribution | Section 3.1 equation, PDF p. 3 |
| 13 | Structured error taxonomy and multi-label Jaccard agreement | Figure 2 and Sections 3.2 and 4.3, PDF pp. 4 and 7 |
| 14 | Equation-based versus rule-based tasks | Sections 2, 3.2, and 4.1, PDF pp. 2, 4, and 6 |
| 15 | MedRaC overview | Figure 3 and Section 3.3, PDF p. 5 |
| 16 | Formula retrieval | Section 3.3 and Table 7, PDF pp. 5 and 13 |
| 17 | Value extraction and code generation | Section 3.3, PDF p. 5; local input-contract verification |
| 18 | Python-assisted calculation | Section 3.3, PDF p. 5; released output contract |
| 19 | Component-to-error mapping | Sections 3.3, 4.3, and 5, PDF pp. 5, 7, and 8 |
| 20 | Dataset and benchmark cleanup | Section 4 and Appendix A, PDF pp. 5 and 13 |
| 21 | Experimental setup | Section 4 and Appendix B, PDF pp. 5 and 13 |
| 22 | Main results | Table 1, PDF p. 6 |
| 23 | Equation-based and rule-based performance | Table 1 and Section 4.1, PDF p. 6 |
| 24 | Error reduction | Figure 4 and Section 4.3, PDF p. 7 |
| 25 | Retrieval and code-execution ablations | Tables 5 and 6, PDF p. 8 |
| 26 | LLM Judge validation | Table 2 and Appendix C, PDF pp. 6 and 13-14 |
| 27 | Local demo objective and pipeline | `seminar_demo/pipeline.py`, `seminar_demo/safe_execution.py`, `streamlit_app.py` |
| 28 | Streamlit representative example | Actual replay screenshots under `slides/assets/local/` |
| 29 | Local mini-experiment | `artifacts/experiments/20260713T060240915862Z/manifest.json` and `metrics.json` |
| 30 | Interpretation and limitations | Paper Sections 6-8, PDF p. 9; local manifest and metrics |
| 31 | Conclusions | Synthesis of Sections 3-8, PDF pp. 3-9 |
| 32 | Questions and discussion | Research implications and limitations in Sections 6-8, PDF p. 9 |

## Appendix

| Slide | Topic | Primary evidence |
|---:|---|---|
| 33 | Complete main-results table | Paper Table 1, PDF p. 6 |
| 34 | Complete metric definitions | Section 3.1, PDF p. 3 |
| 35 | Paper discrepancies | Sections 2, 4, and 5; Appendix A, PDF pp. 2, 5, 8, and 13 |
| 36 | Formula-bank scaling | Table 7 and Section 5, PDF pp. 8 and 13 |
| 37 | Local experiment details | `artifacts/experiments/20260713T060240915862Z/manifest.json` and `metrics.json` |
| 38 | Safe-execution boundary | `seminar_demo/safe_execution.py` and `docs/RUNBOOK.md` |
| 39 | References | Paper bibliographic metadata |

## Quantitative ledger

- Appendix G: generated answer `135.432`; gold answer `137.248`; accepted interval `[130.39, 144.11]`; F incorrect, E correct, C correct, A incorrect.
- Table 1 selected One-shot/MedRaC rule and calculation accuracy: Phi-4-mini `12.09/35.10` and `16.47/68.39`; LLaMA3.1-8B `25.07/31.27` and `20.97/70.22`; Qwen3-14B `60.77/50.44` and `67.05/78.37`; GPT-4o `62.24/51.03` and `54.24/64.39`.
- Figure 4 reductions: formula `-587 (-77.5%)`; arithmetic `-352 (-82.6%)`; demographic adjustment `-105 (-70.9%)`.
- Table 5: accuracy `64.68/25.64`; Formula FE `20.78/71.96`; Formula CC `92.66/46.49`.
- Table 6: accuracy `64.68/53.09`; Calculation FE `3.23/31.88`; Calculation CC `97.82/76.52`.
- Table 2 LLM-Expert agreement: Formula `90.2%`; Extraction `78.3%`; Calculation `88.1%`; Answer `97.8%`; 46 validated samples from five calculators.
- Error-type agreement uses multi-label Jaccard similarity, `|A intersection B| / |A union B|`.
- Formula-bank scaling from 55 to 785 documents: Top-1 retrieval `100%` for ada-002, `96.36%` for text-embedding-3-large, and `98.18%` for text-embedding-3-small; Top-2 retrieval `100%` for all three embedding models.
- Local run: Plain CoT `9/10`; MedRaC RAG `10/10`; zero failed/skipped; step-wise evaluation disabled, so local F/E/C/A, strict, CC, and FE metrics are unavailable.

## Disclosed paper discrepancies

- Section 2 reports `1,047` vignettes, while Section 4 and Appendix A report `1,048`. The experimental cohort is `940` after `108` exclusions.
- Section 5 prose reports no-RAG Formula CC as `7.34%`; Table 5 prints `46.49%`. The deck uses the printed table value and discloses both.
