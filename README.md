# Reproducible computational vignettes — CRISPR-perturbation multi-omics identifiability

[![DOI](https://zenodo.org/badge/1285047797.svg)](https://zenodo.org/badge/latestdoi/1285047797)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Code and result files that regenerate the **illustrative computational analyses** in the review:

> Asediya, V. **When can CRISPR-perturbation multi-omics support a causal claim? A method-selection framework for livestock functional genomics.** (Under review, *Briefings in Bioinformatics*.)

All analyses run on **synthetic or public data; no new experimental data were generated.**

---

## Contents
- [What this is](#what-this-is)
- [Study → figure map](#study--figure-map)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Reproduce the figures](#reproduce-the-figures)
- [Data](#data)
- [Scope and integrity](#scope-and-integrity)
- [Citation](#citation)
- [Licence](#licence)
- [Contact](#contact)

## What this is
This repository accompanies a **methods review**, not a primary benchmarking study. The computations here are small, clearly-labelled **illustrative reproductions** that calibrate and visualise the review's framework; each reproduces or makes concrete a result already established in the cited literature. The substantive evidence in the paper is the published benchmark record, which the review synthesises.

## Study → figure map
| Study | Script | Backs in the paper | Status |
|---|---|---|---|
| A | [`study_A_identifiability.py`](study_A_identifiability.py) | Main **Fig 2**; Supp **Fig S1a–b** | load-bearing |
| B | [`study_B_norman_benchmark.py`](study_B_norman_benchmark.py) | Main **Fig 9a**; Supp **Fig S2a** | load-bearing |
| C | [`study_C_grand_benchmark.py`](study_C_grand_benchmark.py) | Supp **Fig S1c** (contract matrix) | load-bearing |
| D | [`study_D_obstruction_hierarchy.py`](study_D_obstruction_hierarchy.py) | Supp **Fig S1d** | load-bearing |
| F | [`study_F_synergy.py`](study_F_synergy.py) | Main **Fig 9b**; Supp **Fig S2b** | load-bearing |
| G | [`study_G_esm.py`](study_G_esm.py) | ESM-2 constraint ↔ conservation (support for **Figs 11–12**) | load-bearing |
| H, H2, I, J | `study_H*.py`, `study_I*.py`, `study_J*.py` | Supp **Fig S1e–f** + editability material | **exploratory** |
| K, L | [`unrun_scaffolds/`](unrun_scaffolds/) | — | **not run** (see [`unrun_scaffolds/NOT_RUN.md`](unrun_scaffolds/NOT_RUN.md)) |

[`study_figures.py`](study_figures.py) regenerates `study_validation_causal.png` and `study_perturbation_ml.png` from the computed result files.

## Repository layout
| Path | Purpose |
|---|---|
| `study_*.py` | One analysis ("study") each; see the map above |
| `study_*_results*.json` | Machine-written numeric results for each completed study (figures are built from these) |
| `study_figures.py` | Builds the publication figures from the result JSONs |
| `prepare_norman_cache.py` | One-time cache of the Norman 2019 pseudobulk effects → `norman_cache.npz` |
| `study_validation_causal.png`, `study_perturbation_ml.png` | Rendered figure outputs |
| `unrun_scaffolds/` | Runnable scaffolds **not executed** for this paper (no conclusions depend on them) |
| `requirements.txt` | Python dependencies |
| `.zenodo.json`, `CITATION.cff` | Archive + citation metadata |
| `LICENSE` | MIT licence |

## Installation
Python **3.9+** is recommended.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the figures
```bash
# 1. (once) fetch the public Norman 2019 dataset (see Data) and build its cache
python prepare_norman_cache.py

# 2. regenerate the figure outputs from the computed results
python study_figures.py
```
The synthetic-network studies (A, C, D) generate their own data in-script from fixed random seeds, so they are fully self-contained.

## Data
The Norman et al. (2019) combinatorial Perturb-seq dataset is **public and large**, so it is **not redistributed here**. Fetch it from its primary source (GEO **GSE133344** / the authors' release) and build the analysis cache with `prepare_norman_cache.py`. The ESMFold / ESM-2 analyses (Studies G, J) use the public ESM Atlas / `fair-esm` weights. No new biological data underlie any result here.

## Scope and integrity
- **Illustrative, not primary research.** The vignettes calibrate the framework; they are not a comprehensive method benchmark.
- The Norman 2019 reproduction is run **on the Norman 2019 dataset only** — it is not a multi-dataset or "whole-corpus" benchmark.
- **Exploratory studies (H, H2, I, J)** back the exploratory supplementary panels and are offered as illustration, not validated methods, exactly as stated in the paper.
- **Not run for this paper (`unrun_scaffolds/`).** `study_K_*` was executed on the Norman 2019 dataset only; `study_L_*`'s foundation-model arm was never run (it needs a GPU and an unimplemented `foundation_predict()`). Their machine-written result JSONs are intentionally **omitted** so they cannot misrepresent scope. **No conclusion in the manuscript depends on Study K or L.**

## Citation
If you use this code, please cite the article and the software archive (a `CITATION.cff` is included, so GitHub's *"Cite this repository"* button works):

> Asediya, V. *When can CRISPR-perturbation multi-omics support a causal claim? A method-selection framework for livestock functional genomics.* (Under review, *Briefings in Bioinformatics*.)

The software archive carries a Zenodo DOI (badge above); the article DOI will be added on publication.

## Licence
Code is released under the **MIT Licence** (see [`LICENSE`](LICENSE)). The manuscript text and figures are © the author.

## Contact
Varunkumar Asediya — Institute of Veterinary Medicine, Nicolaus Copernicus University in Toruń, Poland — varun@doktorant.umk.pl — ORCID [0000-0002-9402-0349](https://orcid.org/0000-0002-9402-0349)
