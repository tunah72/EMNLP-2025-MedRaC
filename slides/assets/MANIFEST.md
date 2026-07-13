# Slide Asset Manifest

All paths are repository-relative. Paper assets preserve the original labels, legends, colors, and relative proportions. Cropping removes only page context or transparency padding.

| Asset | Source | PDF page | Extraction method |
|---|---|---:|---|
| `assets/paper/paper_fig_1_stepwise_evaluation.png` | Wang et al., Figure 1 | 4 | Embedded image extraction with original alpha mask |
| `assets/paper/paper_fig_2_error_taxonomy.png` | Wang et al., Figure 2 | 4 | Embedded image extraction with original alpha mask |
| `assets/paper/paper_fig_3_medrac_pipeline.png` | Wang et al., Figure 3 | 5 | Embedded image extraction with original alpha mask |
| `assets/paper/paper_fig_4_llama_error_counts.png` | Wang et al., Figure 4 | 7 | Embedded image extraction with original alpha mask |
| `assets/paper/paper_table_1_main_results.pdf` | Wang et al., Table 1 | 6 | Vector crop from original PDF page |
| `assets/paper/paper_table_2_judge_validation.png` | Wang et al., Table 2 | 6 | High-resolution crop preserving the original table |
| `assets/paper/paper_table_5_rag_ablation.png` | Wang et al., Table 5 | 8 | High-resolution crop preserving the original table |
| `assets/paper/paper_table_6_code_ablation.png` | Wang et al., Table 6 | 8 | High-resolution crop preserving the original table |
| `assets/paper/paper_appendix_g_case_top.pdf` | Wang et al., Appendix G | 17 | Vector crop: case, gold formula, and broad-tolerance verdict |
| `assets/paper/paper_appendix_g_case_bottom.pdf` | Wang et al., Appendix G | 17 | Vector crop: strict check, F/E/C/A table, and clinical significance |
| `assets/local/streamlit_replay_inputs.png` | Running local `streamlit_app.py` | n/a | Browser screenshot; Replay mode |
| `assets/local/streamlit_replay_question.png` | Running local `streamlit_app.py` | n/a | Browser screenshot; Calculator ID, Patient Note, Question, and pipeline stages |
| `assets/local/streamlit_replay_contract.png` | Running local `streamlit_app.py` | n/a | Browser screenshot; retrieval query, score, rank, and retrieved formula |
| `assets/local/streamlit_stepwise_evaluation.png` | Running local `streamlit_app.py` | n/a | Browser screenshot; Step-wise Evaluation tab |
| `assets/local/streamlit_live_configuration.png` | Running local `streamlit_app.py` | n/a | Browser screenshot; Dataset live demo configuration, no API call |
| `assets/local/fit_hcmus_logo.png` | `Diffusion LLM-SampleSlides.pdf`, title page | 1 | Original embedded raster image combined with its alpha mask |

Paper PDF SHA-256: `757b98d0881c661d0e365b12998550751b289d610bea0339281dc7609f5a0107`.
