
> **Official Repository for EMNLP 2025 Oral Paper: From Scores to Steps: Diagnosing and Improving LLM Performance in Evidence-Based Medical Calculations**  
> This repo contains the complete inference pipeline and *LLM evaluation* code used in the paper, allowing reviewers and researchers to reproduce every experiment.

---

## Repository at a Glance

| Path             | Purpose                                             | Key Files / Notes                                                                                                                                          |
| ---------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`data/`**      | Test set, one-shot examples, formula knowledge base | `test_data.csv`, `one_shot_finalized_explanation.json`, `formula_new.json`, `web_formula.txt`                                                              |
| **`evaluator/`** | Evaluation utilities                                | `regEvaluator.py` – original MedCalc-Bench rubric (\[NeurIPS 2024 Dataset & Benchmark Track])<br>`llmEvaluator.py` – our *LLM Step-wise Evaluation* (§3.1) |
| **`method/`**    | All baselines and our proposed **MedRaC**, as well as the base class         | `plain.py`, `method.py`, `codeGenerator.py`, `selfRefine.py`, …                                                                                |
| **`model/`**     | Model wrappers                                      | `gpt.py` (OpenAI API), `vllmModels.py` (vLLM server), `model.py` (base class)                                                                          |
| **`schema/`**    | JSON schema definitions                             | `schemas.py`                                                                                                                                               |
| **`utils/`**     | Helper scripts                                      | `error_type.py` – error taxonomy (§3.2)<br>`remove_bad_data.py` – remove erroneous samples (Appendix A)                                                    |
| **Root**         | Entry point & metadata                              | `run.py` (demo workflow), `requirements.txt`, `.env` (user-supplied), `README.md`                                                                          |

---

## Quick Start

### 1 · Install dependencies

```bash
git clone https://github.com/Super-Billy/EMNLP-2025-MedRaC.git
cd EMNLP-2025-MedRaC

pip install -r requirements.txt

# For RAG baselines
pip install faiss-gpu      # or: pip install faiss-cpu
```

### 2 · Add API keys

Create an **`.env`** file in the repo root:

```env
OPENAI_API_KEY=sk-********************************
DEEPSEEK_API_KEY=dsk-*****************************
```

*`Our paper` originally used DeepSeek models for evaluation by default; We understand it might not be available for all reviewers, we also support evaluation using GPT series*

### 3 · Run the demo / reproduce results

```bash
python run.py
```

`run.py` loads the test set, runs selected methods, do LLM evaluations and save results. Inline comments show how to change models, batch sizes, or subset the data for quick checks.

### 4 · How to evaluate with our framework

`method.py` serves as the base class for all our methods. By implementing the required functions defined in this abstract class, you can evaluate a new method using the same pipeline without any additional modifications.

---


## License & Citation

Released **for review and non-commercial research only**.
A full license and BibTeX entry will be added after acceptance.

---