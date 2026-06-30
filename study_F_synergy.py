#!/usr/bin/env python3
"""STUDY F — synergy decomposition: where (and why) perturbation prediction breaks (Norman 2019).
Loads the cached pseudobulk effects (norman_cache.npz). Decomposes each two-gene combination into the
additive part (sum of singles) and the residual (genetic interaction / synergy), then shows EVERY model
— simple or deep — degrades on exactly the high-synergy combinations: the unpredictable part is the
genuine interaction, not model capacity. Operationalises the review's 'interaction' causal question and
the limit that prediction != identification. Real public data; all numbers computed."""
import numpy as np, json
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

c = np.load("norman_cache.npz", allow_pickle=True)
singles = list(c["singles"]); gi = {g: i for i, g in enumerate(singles)}
Sd, Yte = c["single_delta"], c["combo_delta"]; ca, cb = c["combo_a"], c["combo_b"]
Xtr = np.eye(len(singles)); Ytr = Sd
def feat(a, b):
    v = np.zeros(len(singles)); v[gi[a]] = 1; v[gi[b]] = 1; return v
Xte = np.array([feat(a, b) for a, b in zip(ca, cb)])
add = np.array([Sd[gi[a]] + Sd[gi[b]] for a, b in zip(ca, cb)])
# synergy = residual of the true effect beyond additive, normalised by true-effect magnitude
syn = np.array([np.linalg.norm(Yte[i] - add[i]) / (np.linalg.norm(Yte[i]) + 1e-9) for i in range(len(Yte))])
preds = {"Additive": add,
         "Linear (ridge)": Ridge(alpha=1.0).fit(Xtr, Ytr).predict(Xte),
         "Random forest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0).fit(Xtr, Ytr).predict(Xte),
         "MLP (deep)": MLPRegressor((256, 128), max_iter=800, random_state=0).fit(Xtr, Ytr).predict(Xte)}
def per_r(P):
    return np.array([np.corrcoef(Yte[i], P[i])[0, 1] if Yte[i].std() > 0 and P[i].std() > 0 else np.nan
                     for i in range(len(Yte))])
order = np.argsort(syn); terc = np.array_split(order, 3); tn = ["low-synergy", "mid-synergy", "high-synergy"]
res = {"n_combos": int(len(syn)), "synergy_mean": float(syn.mean()), "synergy_median": float(np.median(syn)),
       "frac_additive_explained(<0.25)": float((syn < 0.25).mean()), "frac_synergistic(>0.5)": float((syn > 0.5).mean()),
       "per_method_r_by_synergy_tercile": {}, "corr_synergy_vs_predErr": {}}
for name, P in preds.items():
    r = per_r(P)
    res["per_method_r_by_synergy_tercile"][name] = {tn[k]: float(np.nanmean(r[terc[k]])) for k in range(3)}
    res["per_method_r_by_synergy_tercile"][name]["overall"] = float(np.nanmean(r))
    m = ~np.isnan(r); res["corr_synergy_vs_predErr"][name] = float(np.corrcoef(syn[m], (1 - r)[m])[0, 1])
print(json.dumps(res, indent=1)); json.dump(res, open("study_F_results.json", "w"), indent=1)
print("done -> study_F_results.json")
