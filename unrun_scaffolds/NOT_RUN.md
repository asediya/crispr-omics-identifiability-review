# Unrun scaffolds — NOT used in the paper

These two scripts are **runnable scaffolds for future work** and produced **no result reported in the manuscript**:

- `study_K_perturbseq_metabenchmark.py` — a uniform multi-dataset Perturb-seq pipeline. For this paper it was executed **on the Norman 2019 dataset only**; the other five datasets were not downloaded. There is **no** "whole-corpus" result. The Norman 2019 finding the paper does report comes from Studies B and F, not from this script.
- `study_L_foundation_audit.py` — a scaffold to compare single-cell foundation models (scGPT/Geneformer) against an identification-aware baseline. The **foundation-model arm was never run** (it needs a GPU and an unimplemented `foundation_predict()`). The paper makes **no** claim that the author tested foundation models; all foundation-model statements cite the published benchmark literature.

Their machine-written result JSONs are intentionally **not deposited** because their summary fields do not reflect a completed analysis. No conclusion in the manuscript depends on either script.
