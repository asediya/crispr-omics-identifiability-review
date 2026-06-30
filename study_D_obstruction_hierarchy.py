#!/usr/bin/env python3
"""STUDY D — empirical validation of the identifiability-obstruction hierarchy (Table 3).

For each obstruction level, simulate the exact failure mode and show (i) the predicted method fails,
and (ii) the 'minimal remedying design' recovers the truth — turning the hierarchy from a claim into a
demonstrated result. Original, reproducible, assumption-stated; all numbers computed.

  L1 Coverage / redundancy  — a redundant gene family: single knockout is buffered (no necessity
       signal), family-level (combinatorial) knockout reveals necessity.  [the ANP32A vignette]
  L2 Faithfulness violation — a cancelling path (direct + opposing indirect ~ net zero marginal
       association): observational discovery misses the true edge; controlling/perturbing the
       mediator (multi-context) reveals it.
  L3 State-conditionality   — a rare-state-specific effect: bulk/population average reads ~null;
       state-stratified (single-cell) analysis recovers the strong within-state effect.

Output: study_D_results.json
"""
import numpy as np, json, sys
from sklearn.linear_model import LinearRegression

FULL = "--full" in sys.argv
SEEDS = list(range(2000)) if FULL else list(range(50))

def L1_redundancy(rng):
    """Necessity of a 2-member redundant family for a phenotype P = step(G1+G2 >= 1)."""
    n = 4000
    U = rng.normal(0, 1, n)                       # shared upstream driver
    G1 = (U + rng.normal(0, 0.4, n) > 0).astype(float)
    G2 = (U + rng.normal(0, 0.4, n) > 0).astype(float)
    pheno = lambda g1, g2: ((g1 + g2) >= 1).astype(float).mean()   # OR-logic redundancy
    base = pheno(G1, G2)
    d_single = base - pheno(np.zeros(n), G2)                       # knock out G1 only
    d_double = base - pheno(np.zeros(n), np.zeros(n))              # knock out the family
    return {"single_KO_effect": float(d_single), "family_KO_effect": float(d_double)}

def L2_faithfulness(rng):
    """X->Y direct (+a) and X->Z->Y indirect (b*c) tuned to cancel -> ~0 marginal cov(X,Y)."""
    n = 5000; a = 1.0; b = 1.0; c = -1.0          # a + b*c = 0  => net marginal ~ 0
    X = rng.normal(0, 1, n)
    Z = b * X + rng.normal(0, 0.3, n)
    Y = a * X + c * Z + rng.normal(0, 0.3, n)
    marg = abs(np.corrcoef(X, Y)[0, 1])                            # observational X-Y association (~0)
    # remedy: control the mediator Z (regress Y on X and Z) -> partial X effect recovered
    coef = LinearRegression().fit(np.c_[X, Z], Y).coef_[0]
    return {"obs_XY_assoc": float(marg), "edge_detected_obs": int(marg > 0.1),
            "partial_X_effect_controlling_Z": float(abs(coef)), "edge_detected_remedy": int(abs(coef) > 0.3)}

def L3_state_conditional(rng):
    """Effect strong in a rare state (fraction p), null otherwise; bulk dilutes it."""
    n = 6000; p = 0.15; beta_state = 2.0
    state = (rng.random(n) < p).astype(int)
    X = rng.normal(0, 1, n)
    Y = beta_state * state * X + rng.normal(0, 0.5, n)             # effect only in the rare state
    bulk = LinearRegression().fit(X.reshape(-1, 1), Y).coef_[0]    # population-average effect (~p*beta)
    within = LinearRegression().fit(X[state == 1].reshape(-1, 1), Y[state == 1]).coef_[0]
    return {"bulk_effect": float(abs(bulk)), "within_state_effect": float(abs(within)),
            "detected_bulk": int(abs(bulk) > 0.5), "detected_within": int(abs(within) > 0.5)}

if __name__ == "__main__":
    out = {"L1": [], "L2": [], "L3": []}
    for s in SEEDS:
        rng = np.random.default_rng(s)
        out["L1"].append(L1_redundancy(rng))
        out["L2"].append(L2_faithfulness(np.random.default_rng(s + 10**6)))
        out["L3"].append(L3_state_conditional(np.random.default_rng(s + 2 * 10**6)))
    def agg(level, key):
        v = np.array([t[key] for t in out[level]]); return [float(v.mean()), float(v.std())]
    summary = {
        "L1_redundancy": {"single_KO_effect": agg("L1", "single_KO_effect"),
                          "family_KO_effect": agg("L1", "family_KO_effect")},
        "L2_faithfulness": {"obs_XY_assoc": agg("L2", "obs_XY_assoc"),
                            "edge_detected_obs_rate": agg("L2", "edge_detected_obs"),
                            "partial_X_effect_controlling_Z": agg("L2", "partial_X_effect_controlling_Z"),
                            "edge_detected_remedy_rate": agg("L2", "edge_detected_remedy")},
        "L3_state": {"bulk_effect": agg("L3", "bulk_effect"), "within_state_effect": agg("L3", "within_state_effect"),
                     "detected_bulk_rate": agg("L3", "detected_bulk"), "detected_within_rate": agg("L3", "detected_within")},
        "n_seeds": len(SEEDS)}
    json.dump(summary, open("study_D_results.json", "w"), indent=1)
    print("STUDY D — obstruction-hierarchy validation (mean, sd):")
    print(json.dumps(summary, indent=1))
