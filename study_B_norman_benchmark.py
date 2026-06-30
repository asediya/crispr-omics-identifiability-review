#!/usr/bin/env python3
"""STUDY B — extended, statistically-quantified perturbation-prediction benchmark (Norman 2019).

Strengthens the cautionary Figure-4 result by (i) adding method classes spanning the simple→complex
spectrum, (ii) reporting two metrics (mean Pearson r and RMSE), and (iii) giving 95% bootstrap
confidence intervals over the held-out two-gene combinations — so "simple baselines match or beat a
deep model" is shown with uncertainty, not as point estimates. Real public data; all numbers computed.

Task: predict the held-out two-gene combination effect (Δ vs control, top-2000 variable genes) from
single-gene effects. Train signal = single-gene perturbations; test = combinations of seen singles.

Output: study_B_results.json
"""
import anndata as ad, numpy as np, os, json, time
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
H5 = os.path.join(HERE, "data", "norman2019.h5ad")
t0 = time.time()
A = ad.read_h5ad(H5, backed="r")
pcol = next((c for c in ("perturbation", "perturbation_name", "guide_identity", "gene") if c in A.obs), "perturbation")
pert = A.obs[pcol].astype(str).values
labels = np.unique(pert)
ctrl = "control" if "control" in labels else [l for l in labels if "ctrl" in l.lower()][0]

def pseudobulk(label):
    idx = np.where(pert == label)[0]
    X = A[idx].to_memory().X
    X = X if sparse.issparse(X) else sparse.csr_matrix(X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    Xn = sparse.csr_matrix(X.multiply((1e4 / tot)[:, None])); Xn.data = np.log1p(Xn.data)
    return np.asarray(Xn.mean(0)).ravel()

singles = [l for l in labels if l != ctrl and "_" not in l]; sset = set(singles)
valid = [(c, *c.split("_")) for c in labels if "_" in c and all(g in sset for g in c.split("_"))]
print(f"control={ctrl}  singles={len(singles)}  held-out combos={len(valid)}  ({time.time()-t0:.0f}s)")
need = set([ctrl]) | set(singles) | set(c for c, _, _ in valid)
pb = {l: pseudobulk(l) for l in need}; cp = pb[ctrl]
deltas = np.array([pb[s] - cp for s in singles])
topk = np.argsort(deltas.var(0))[-2000:]
d = lambda l: (pb[l] - cp)[topk]
gi = {g: i for i, g in enumerate(singles)}
Xtr = np.eye(len(singles)); Ytr = np.array([d(s) for s in singles])
def feat(a, b):
    v = np.zeros(len(singles)); v[gi[a]] = 1; v[gi[b]] = 1; return v
Xte = np.array([feat(a, b) for _, a, b in valid]); Yte = np.array([d(c) for c, _, _ in valid])
print(f"fitting methods ({time.time()-t0:.0f}s) ...")

preds = {
    "No change (control)": np.zeros_like(Yte),
    "Best single":         np.array([d(a) if np.abs(d(a)).sum() >= np.abs(d(b)).sum() else d(b) for _, a, b in valid]),
    "Mean of singles":     np.array([(d(a) + d(b)) / 2 for _, a, b in valid]),
    "Additive":            np.array([d(a) + d(b) for _, a, b in valid]),
    "Linear (ridge)":      Ridge(alpha=1.0).fit(Xtr, Ytr).predict(Xte),
    "k-NN (k=5)":          KNeighborsRegressor(n_neighbors=5).fit(Xtr, Ytr).predict(Xte),
    "Random forest":       RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0).fit(Xtr, Ytr).predict(Xte),
    "MLP (deep)":          MLPRegressor((256, 128), max_iter=800, random_state=0).fit(Xtr, Ytr).predict(Xte),
}

def per_combo_r(P):
    return np.array([np.corrcoef(Yte[i], P[i])[0, 1] if Yte[i].std() > 0 and P[i].std() > 0 else np.nan
                     for i in range(len(Yte))])
def per_combo_rmse(P):
    return np.sqrt(((Yte - P) ** 2).mean(1))

rng = np.random.default_rng(0); B = 2000; m = len(valid)
res = {}
for name, P in preds.items():
    r = per_combo_r(P); rmse = per_combo_rmse(P)
    rv = r[~np.isnan(r)]
    boot = [np.nanmean(r[rng.integers(0, m, m)]) for _ in range(B)]
    res[name] = {"mean_r": float(np.nanmean(r)),
                 "ci95_r": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                 "mean_rmse": float(np.mean(rmse)), "n_combos_scored": int(len(rv))}
print("\nmethod                     mean r   95% CI            RMSE")
for k, v in sorted(res.items(), key=lambda kv: -kv[1]["mean_r"]):
    print(f"  {k:24s} {v['mean_r']:.3f}  [{v['ci95_r'][0]:.3f},{v['ci95_r'][1]:.3f}]  {v['mean_rmse']:.3f}")
json.dump({"n_singles": len(singles), "n_combos": len(valid), "n_genes": int(len(topk)),
           "bootstrap": B, "results": res}, open("study_B_results.json", "w"), indent=1)
print(f"\ndone in {time.time()-t0:.0f}s -> study_B_results.json")
