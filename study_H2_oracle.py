#!/usr/bin/env python3
"""STUDY H2 — the METHOD-SELECTION ORACLE (moonshot #1, second stage).

Reads the identifiability atlas (study_H_atlas.json) and asks: can the review's contract matrix be
LEARNED? From only cheaply-computable dataset descriptors (n, sample/feature ratio, varsortability,
non-Gaussianity, mean|corr|), train a model to predict (a) which algorithm will best recover the
structure, and (b) each algorithm's expected F1. Honest, deposited, group-aware cross-validation
(splitting on the data-generating REGIME so the oracle is tested on unseen regimes, not memorised cells).

This converts the qualitative identifiability-contract matrix into a quantitative, validated, deployable
recommender — the central novelty: a method-selection oracle grounded in identifiability theory.

Run: analysis_env/bin/python study_H2_oracle.py   (or base python if sklearn present)
Output: study_H2_oracle.json
"""
import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

A = json.load(open("study_H_atlas.json"))["records"]
recs = [r for r in A if "f1" in r and "algo" in r]                      # drop error rows
FEATS = ["n", "deg", "ns", "sample_feature_ratio", "varsortability", "non_gaussianity", "mean_abs_corr"]
# group cells by their full configuration; within a cell pick the best algorithm (max F1, tie-break min SHD)
cells = defaultdict(dict)
for r in recs:
    key = (r["n"], r["deg"], r["ns"], r["noise"], r["form"], r["conf"], r["seed"])
    cells[key][r["algo"]] = r
rows = []
for key, algos in cells.items():
    if not algos: continue
    best = max(algos.values(), key=lambda r: (r["f1"], -r["shd"]))
    feat = [best[f] for f in FEATS]
    rows.append({"key": key, "regime": (best["noise"], best["form"]), "feat": feat,
                 "best_algo": best["algo"], "per_algo_f1": {a: algos[a]["f1"] for a in algos}})
print(f"atlas: {len(recs)} records -> {len(rows)} cells; algorithms: {sorted({r['algo'] for r in recs})}")

X = np.array([r["feat"] for r in rows], float)
y = np.array([r["best_algo"] for r in rows])
groups = np.array([str(r["regime"]) for r in rows])                     # CV split on regime (unseen-regime test)
classes = sorted(set(y))
out = {"n_cells": len(rows), "features": FEATS, "classes": classes,
       "best_algo_distribution": {c: int((y == c).sum()) for c in classes}}

# ----- (a) classifier: predict the best algorithm from descriptors, regime-grouped CV -----
ngroups = len(set(groups))
if ngroups >= 2 and len(rows) >= 20:
    gkf = GroupKFold(n_splits=min(5, ngroups)); accs = []; baselines = []
    maj = max(classes, key=lambda c: (y == c).sum())
    for tr, te in gkf.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
        baselines.append(float((y[te] == maj).mean()))                 # always-pick-majority baseline
    out["oracle_accuracy_cv"] = [float(np.mean(accs)), float(np.std(accs))]
    out["majority_baseline_cv"] = [float(np.mean(baselines)), float(np.std(baselines))]
    full = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0).fit(X, y)
    out["feature_importance"] = {f: float(round(v, 4)) for f, v in zip(FEATS, full.feature_importances_)}
else:
    out["note"] = f"insufficient regimes/cells for grouped CV (groups={ngroups}, cells={len(rows)}); run study_H with --full"

# ----- (b) per-algorithm F1 regressors: expected F1 from descriptors -----
algos = sorted({a for r in rows for a in r["per_algo_f1"]})
out["per_algo_f1_regression_r2"] = {}
for a in algos:
    idx = [i for i, r in enumerate(rows) if a in r["per_algo_f1"]]
    if len(idx) < 30: continue
    Xa = X[idx]; ya = np.array([rows[i]["per_algo_f1"][a] for i in idx]); ga = groups[idx]
    if len(set(ga)) < 2: continue
    gkf = GroupKFold(n_splits=min(5, len(set(ga)))); r2s = []
    for tr, te in gkf.split(Xa, ya, ga):
        rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(Xa[tr], ya[tr])
        p = rf.predict(Xa[te]); ss = ((ya[te] - p) ** 2).sum(); tot = ((ya[te] - ya[te].mean()) ** 2).sum()
        r2s.append(float(1 - ss / tot) if tot > 0 else np.nan)
    out["per_algo_f1_regression_r2"][a] = float(np.nanmean(r2s))

json.dump(out, open("study_H2_oracle.json", "w"), indent=1)
print(json.dumps(out, indent=1)); print("done -> study_H2_oracle.json")
