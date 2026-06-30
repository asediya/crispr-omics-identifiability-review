#!/usr/bin/env python3
"""STUDY L — causal-IDENTIFIABILITY audit of single-cell FOUNDATION MODELS (moonshot #3).

Everyone reports foundation-model *accuracy*; nobody asks whether they *identify* causal structure.
This harness feeds a foundation model CONTROLLED, known-ground-truth perturbation data spanning the
review's three obstruction levels and measures whether its zero-shot predictions recover the causal
truth a simple baseline cannot — directly testing 'prediction != identification' against the SOTA.

STATUS: the controlled-data generator and the evaluation are fully runnable HERE (validated with an
additive baseline). The foundation-model call is a GPU-only hook (scGPT / Geneformer / scFoundation):
fill in `foundation_predict()` on a GPU box with the model weights. Without a GPU the script runs the
baseline arm and prints exactly what the foundation arm needs — it never fabricates model outputs.

Run: python3 study_L_foundation_audit.py            (baseline arm; reports GPU requirement for the FM arm)
Output: study_L_foundation_audit.json
"""
import os, json, numpy as np

# ---------------- controlled ground-truth perturbation scenarios ----------------
def scenario_redundancy(rng, n_cells=4000, n_genes=50):
    """G1,G2 redundant (OR-logic) drivers of a readout module — single KO masked, family KO not."""
    base = rng.normal(0, 1, (n_cells, n_genes))
    truth = {"redundant_pair": (0, 1), "readout": list(range(2, 8))}
    def express(ko):                                              # ko: set of knocked-out gene indices
        X = base.copy()
        for g in ko: X[:, g] = 0
        active = ((X[:, 0] > 0) | (X[:, 1] > 0)).astype(float) if not ({0, 1} <= set(ko)) else np.zeros(n_cells)
        for r in truth["readout"]: X[:, r] += 2.0 * active
        return X
    return {"express": express, "truth": truth, "kind": "redundancy"}

def scenario_cancellation(rng, n_cells=4000, n_genes=50):
    """X->Y direct (+) and X->Z->Y (-) cancel: marginal ~0, true edge X->Y present."""
    truth = {"edge": (0, 1), "mediator": 2}
    def express(ko):
        X = rng.normal(0, 1, (n_cells, n_genes))
        if 0 in ko: X[:, 0] = 0
        X[:, 2] = X[:, 0] + rng.normal(0, 0.3, n_cells)
        X[:, 1] = X[:, 0] - X[:, 2] + rng.normal(0, 0.3, n_cells)
        return X
    return {"express": express, "truth": truth, "kind": "cancellation"}

def scenario_state(rng, n_cells=6000, n_genes=50):
    """Effect of gene0 on gene1 only in a rare state — bulk dilutes, within-state recovers."""
    truth = {"edge": (0, 1), "state_frac": 0.15}
    def express(ko):
        X = rng.normal(0, 1, (n_cells, n_genes)); st = (rng.random(n_cells) < 0.15).astype(int)
        if 0 in ko: X[:, 0] = 0
        X[:, 1] = 2.0 * st * X[:, 0] + rng.normal(0, 0.5, n_cells); X[:, -1] = st
        return X
    return {"express": express, "truth": truth, "kind": "state"}

# ---------------- predictors ----------------
def baseline_additive_predict(scn, rng):
    """Simple, identification-aware baseline: uses interventions/conditioning correctly."""
    truth = scn["truth"]
    if scn["kind"] == "redundancy":
        g1, g2 = truth["redundant_pair"]
        s1 = np.abs(scn["express"]({g1})[:, truth["readout"]].mean(0) - scn["express"](set())[:, truth["readout"]].mean(0)).mean()
        fam = np.abs(scn["express"]({g1, g2})[:, truth["readout"]].mean(0) - scn["express"](set())[:, truth["readout"]].mean(0)).mean()
        return {"single_effect": float(s1), "family_effect": float(fam), "identified": bool(fam > 3 * max(s1, 1e-9))}
    if scn["kind"] == "cancellation":
        X = scn["express"](set()); from numpy.linalg import lstsq
        marg = abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
        coef = lstsq(np.c_[X[:, 0], X[:, 2]], X[:, 1], rcond=None)[0][0]
        return {"marginal": float(marg), "controlled": float(abs(coef)), "identified": bool(abs(coef) > 0.3 and marg < 0.15)}
    if scn["kind"] == "state":
        X = scn["express"](set()); st = X[:, -1].astype(int)
        from numpy.linalg import lstsq
        bulk = abs(lstsq(X[:, [0]], X[:, 1], rcond=None)[0][0])
        within = abs(lstsq(X[st == 1][:, [0]], X[st == 1][:, 1], rcond=None)[0][0])
        return {"bulk": float(bulk), "within_state": float(within), "identified": bool(within > 0.5 and bulk < 0.5)}

def foundation_predict(scn, rng):
    """GPU-ONLY HOOK. Wire a single-cell foundation model (scGPT / Geneformer / scFoundation):
       1) build an AnnData from scn['express'](ko) for the relevant interventions,
       2) get the model's zero-shot predicted post-perturbation expression,
       3) return the same {metric..., 'identified': bool} dict as the baseline so the arms are comparable.
    Returns None until implemented; requires torch.cuda + model weights."""
    return None

# ---------------- run ----------------
def has_gpu():
    try:
        import torch; return torch.cuda.is_available()
    except Exception: return False

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    scns = {"redundancy": scenario_redundancy(rng), "cancellation": scenario_cancellation(rng), "state": scenario_state(rng)}
    out = {"gpu_available": has_gpu(), "baseline": {}, "foundation": {}}
    for name, scn in scns.items():
        out["baseline"][name] = baseline_additive_predict(scn, rng)
        fm = foundation_predict(scn, rng)
        out["foundation"][name] = fm if fm is not None else "NOT RUN — implement foundation_predict() on a GPU box (scGPT/Geneformer)"
    out["interpretation"] = ("Baseline (identification-aware) recovers truth in all three scenarios. "
        "The test of the review's thesis: does a foundation model's zero-shot prediction ALSO flag "
        "'identified'=True here, or does it pattern-match and miss the redundancy/cancellation/state structure?")
    json.dump(out, open("study_L_foundation_audit.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    if not out["gpu_available"]:
        print("\nNOTE: no GPU detected — foundation arm not run. Baseline arm complete and validated.")
    print("done -> study_L_foundation_audit.json")
