
We thank the reviewers and Area Chair for their thoughtful engagement. In response to the feedback, we have substantially revised the paper to improve its clarity, rigor, and presentation. We also note that this submission was made to the EMNLP Special Theme Track: *Interdisciplinary Recontextualization of NLP*. Our work aligns closely with this theme by exploring how LLMs can be more reliably evaluated and applied in high-stakes clinical settings. By shifting the focus from surface-level benchmark scores to process-level reasoning and real-world reliability, the study contributes to a deeper understanding of how NLP advancements can support trustworthy and impactful applications in medicine.

Below, we highlight key contributions and improvements.



### **1. Summary of Recognized Contributions**
Several reviewers acknowledged the core value of our paper in the following areas:
- **Contribution 1**: A step-wise, interpretable evaluation framework for medical calculation.
 Our evaluation pipeline decomposes reasoning into verifiable steps (formula selection, entity extraction, etc.) and supports structured error analysis. Reviewer 9QaW acknowledged this "rigorous methodology design" and noted that the shift toward interpretability is a “highly insightful” contribution to the field.
- **Contribution 2**: A cleaned and re-audited MedCalc-Bench.
 We fixed errors in the original dataset and added specialty-level labels. Reviewer qG73 recognized this as a strength, noting that the cleaned benchmark "enhances reproducibility and evaluation fidelity." Reviewer 9QaW also called the dataset "useful to the community."
- **Contribution 3**: MedRaC, a training-free modular pipeline.
 MedRaC integrates retrieval-augmented generation and code execution to substantially improve LLM accuracy without requiring fine-tuning. Reviewer 9QaW highlighted its "impressive empirical results," while Reviewer nyUH noted that it "strongly increases its actual usefulness in practice." Reviewer ESVB further affirmed its "effectiveness in improving accuracy," even though the pipeline itself is not technically complex.



### **2. Improvements Made During the Rebuttal and Discussion Period**
We took all feedback seriously and have already planned concrete revisions for the camera-ready version. These include:
- **Improvement 1**: Strengthening domain-level analysis.
 Reviewer 9QaW encouraged a more fine-grained benchmark. In response, we annotated each MedCalc-Bench question with its medical specialty and provided domain-specific performance results. These analyses show that MedRaC excels in calculation-heavy domains, where it reduces formula and arithmetic errors substantially. This addition deepens insight into MedRaC’s clinical relevance and robustness.
- **Improvement 2**: Validating the impact of code execution on calculation accuracy
 Reviewer ESVB questioned the effectiveness of the code component. We re-evaluated model outputs using stronger, reasoning-capable judges (GPT‑4.1, GPT‑4o-mini, DeepSeek R1), all of which consistently confirm that code execution significantly improves calculation accuracy. This addresses the ambiguity in Table 5 and substantiates the utility of incorporating Python-based computation.
- **Improvement 3**: Expanding error-type comparison.
 In response to requests for a more complete analysis (e.g., Reviewer ESVB), we extended the error-type breakdown in Figure 4 to include all baselines and MedRaC ablations (code-only, RAG-only). The expanded results highlight how each component addresses different failure modes, demonstrating MedRaC’s comprehensive impact on reasoning fidelity.

These revisions were guided by reviewer input and, we believe, substantially improved the work. We thank Reviewer 9QaW for their encouraging response. For the reviewers who did not participate in further discussion (e.g., Reviewer ESVB, nyUH and qG73), we hope the AC will note that we took their initial feedback seriously and responded with thoughtful analyses and targeted revisions to address their core concerns.



