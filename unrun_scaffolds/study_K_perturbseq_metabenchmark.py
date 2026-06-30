#!/usr/bin/env python3
"""STUDY K — unified PERTURB-SEQ META-BENCHMARK across the public corpus (moonshot #2).

Runs ONE identifiability-framed pipeline uniformly across every public single-cell CRISPR screen that is
present locally: pseudobulk -> single->combination prediction (8 methods, bootstrap CIs) -> synergy
decomposition. Answers, on the WHOLE corpus rather than one dataset, whether 'simple >= deep' and the
synergy ceiling hold universally. Generalises Studies B and F from Norman-2019 to the full corpus.

DATA: each entry points to a local .h5ad (download separately; sizes in comments). The script processes
whatever is present and LOGS what is missing — it never silently benchmarks a partial corpus.

Run: caffeinate -i env OMP_NUM_THREADS=1 ... python3 study_K_perturbseq_metabenchmark.py
Output: study_K_metabenchmark.json
"""
import os, json, time, numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
# name -> (filename, perturbation obs-column candidates).  Add files as you download them.
DATASETS = [
    ("Norman2019",   "norman2019.h5ad",   ("perturbation", "perturbation_name", "guide_identity")),   # ~666 MB (present)
    ("Replogle2022", "replogle2022.h5ad", ("gene", "perturbation", "sgID_AB")),                         # genome-scale, ~tens of GB
    ("Adamson2016",  "adamson2016.h5ad",  ("perturbation", "guide_identity")),                          # ~ small
    ("Dixit2016",    "dixit2016.h5ad",    ("perturbation", "guide_identity")),
    ("Frangieh2021", "frangieh2021.h5ad", ("perturbation", "guide")),
    ("Papalexi2021", "papalexi2021.h5ad", ("perturbation", "guide")),
]

def pseudobulk_dataset(path, pcols):
    import anndata as ad
    A = ad.read_h5ad(path)                                          # in-memory (fast); use backed='r' if RAM-bound
    pcol = next((c for c in pcols if c in A.obs), None)
    if pcol is None: return None, f"no perturbation column among {pcols}"
    pert = A.obs[pcol].astype(str).values; labels = np.unique(pert)
    ctrl = next((l for l in labels if l.lower() in ("control", "ctrl", "non-targeting", "nt")), None)
    if ctrl is None: ctrl = [l for l in labels if "ctrl" in l.lower() or "control" in l.lower()][:1]
    if not ctrl: return None, "no control label found"
    ctrl = ctrl if isinstance(ctrl, str) else ctrl[0]
    X = A.X; X = X if sparse.issparse(X) else sparse.csr_matrix(X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    Xn = sparse.csr_matrix(X.multiply((1e4 / tot)[:, None])); Xn.data = np.log1p(Xn.data)
    pb = lambda lab: np.asarray(Xn[np.where(pert == lab)[0]].mean(0)).ravel()
    sep = "_" if any("_" in l for l in labels) else ("+" if any("+" in l for l in labels) else "_")
    singles = [l for l in labels if l != ctrl and sep not in l]; sset = set(singles)
    valid = [(c, *c.split(sep)) for c in labels if sep in c and all(g in sset for g in c.split(sep)) and len(c.split(sep)) == 2]
    if len(singles) < 5 or len(valid) < 5: return None, f"too few singles/combos ({len(singles)}/{len(valid)})"
    cp = pb(ctrl); sd = np.array([pb(s) - cp for s in singles]); topk = np.argsort(sd.var(0))[-2000:]
    gi = {g: i for i, g in enumerate(singles)}
    return {"singles": singles, "gi": gi, "Sd": sd[:, topk],
            "combos": valid, "Cd": np.array([(pb(c) - cp)[topk] for c, _, _ in valid])}, None

def benchmark(P):
    singles, gi, Sd, Cd, combos = P["singles"], P["gi"], P["Sd"], P["Cd"], P["combos"]
    Xtr = np.eye(len(singles))
    feat = lambda a, b: (lambda v: (v.__setitem__(gi[a], 1), v.__setitem__(gi[b], 1), v)[-1])(np.zeros(len(singles)))
    Xte = np.array([feat(a, b) for _, a, b in combos])
    add = np.array([Sd[gi[a]] + Sd[gi[b]] for _, a, b in combos])
    preds = {"No change": np.zeros_like(Cd), "Additive": add,
             "Mean of singles": np.array([(Sd[gi[a]] + Sd[gi[b]]) / 2 for _, a, b in combos]),
             "Linear (ridge)": Ridge(1.0).fit(Xtr, Sd).predict(Xte),
             "k-NN (k=5)": KNeighborsRegressor(5).fit(Xtr, Sd).predict(Xte),
             "Random forest": RandomForestRegressor(200, n_jobs=-1, random_state=0).fit(Xtr, Sd).predict(Xte),
             "MLP (deep)": MLPRegressor((256, 128), max_iter=800, random_state=0).fit(Xtr, Sd).predict(Xte)}
    def per_r(Pr): return np.array([np.corrcoef(Cd[i], Pr[i])[0, 1] if Cd[i].std() > 0 and Pr[i].std() > 0 else np.nan for i in range(len(Cd))])
    rng = np.random.default_rng(0); m = len(combos); out = {}
    for name, Pr in preds.items():
        r = per_r(Pr); boot = [np.nanmean(r[rng.integers(0, m, m)]) for _ in range(2000)]
        out[name] = {"mean_r": float(np.nanmean(r)), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    syn = np.array([np.linalg.norm(Cd[i] - add[i]) / (np.linalg.norm(Cd[i]) + 1e-9) for i in range(len(Cd))])
    return {"n_singles": len(singles), "n_combos": len(combos), "methods": out,
            "synergy_mean": float(syn.mean()), "frac_synergistic_gt0.5": float((syn > 0.5).mean())}

if __name__ == "__main__":
    results = {}; missing = []
    for name, fn, pcols in DATASETS:
        path = os.path.join(DATA, fn)
        if not os.path.exists(path): missing.append(fn); print(f"MISSING: {name} ({fn}) — download to data/"); continue
        t = time.time(); P, err = pseudobulk_dataset(path, pcols)
        if err: results[name] = {"error": err}; print(f"{name}: SKIP ({err})"); continue
        results[name] = benchmark(P); print(f"{name}: done in {time.time()-t:.0f}s  "
              f"additive r={results[name]['methods']['Additive']['mean_r']:.3f}  MLP r={results[name]['methods']['MLP (deep)']['mean_r']:.3f}")
    # cross-corpus verdict
    ok = {k: v for k, v in results.items() if "methods" in v}
    verdict = {"datasets_run": list(ok), "datasets_missing": missing,
               "simple_ge_deep_in_all": all(v["methods"]["Additive"]["mean_r"] >= v["methods"]["MLP (deep)"]["mean_r"] for v in ok.values()) if ok else None}
    json.dump({"results": results, "verdict": verdict}, open("study_K_metabenchmark.json", "w"), indent=1)
    print("\nVERDICT:", json.dumps(verdict, indent=1)); print("done -> study_K_metabenchmark.json")
