# MedRaC Scientific Seminar Blueprint

## Communication objective

By the end of the seminar, the audience should understand why final-answer accuracy can conceal invalid medical reasoning, how the F--E--C--A framework diagnoses it, and why MedRaC improves tool-addressable failures.

## Structure

The deck has 21 presentation slides plus Slide 22 Q&A. The spoken portion remains within the 20-minute seminar budget. Slide 2 and the footer use the same six section names:

1. Motivation and Research Problem
2. Evaluation Framework
3. MedRaC Methodology
4. Paper Experiments and Results
5. Local Demo and Mini-Experiment
6. Discussion and Conclusion

Slides 1 and 2 leave the footer context blank. From Slide 3 onward, the footer displays the current section title exactly as listed above.

## Evidence and content boundaries

- The paper remains the primary evidence source; Slide 1 contains a single generic acknowledgement.
- Evaluation Framework shows the complete Strict Correctness, Conditional Correctness, and First Error Attribution equations.
- Paper evidence uses Figures 1--4, the full paper main-results table, and compact tables for setup, focused gains, and LLM--expert agreement.
- Ablation, Streamlit screenshots, appendix material, references frame, repository details, and local implementation citations are excluded.
- The local section is limited to one live-demo pipeline cue and one fixed 10-sample result table.
- Discussion and conclusions concern the paper only.

## Layout system

- Navy title bars, white canvas, equal margins, and a maximum of two visual regions per slide.
- Titles remain on one line; reduce wording before font size.
- Bold is reserved for claims, key metrics, and numerical gains; blue labels introduce content groups.
- Tables align numerical columns and use balanced column widths. Equations and figures are centered in their visual region.
- Every slide must be rendered and inspected for clipping, wrapping, visual imbalance, and footer consistency.
