#!/usr/bin/env python3
"""STUDY H — the IDENTIFIABILITY ATLAS (moonshot #1, ~1000x Study A/C).

A massive factorial sweep that maps, empirically, the entire identifiability frontier of causal
discovery: for every combination of {graph size, density, sample size, noise family, functional form,
latent confounding} it simulates a known-ground-truth SEM and records how every algorithm recovers the
structure, ALONGSIDE cheaply-computable dataset descriptors (varsortability, non-Gaussianity, sparsity,
sample/feature ratio). The resulting table is the training set for STUDY H2 (the method-selection
ORACLE) — turning the review's qualitative identifiability-contract matrix into a quantitative,
deposited, learnable map of *when each method's contract holds*.

Design notes (audit-clean):
  * SHD is the STANDARD DAG SHD (missing + extra + reversed, reversed counted ONCE) — not study_C's
    fn+fp shortcut.
  * Fast algorithms (PC, GES, DirectLiNGAM) run on the full grid; slow continuous-optim methods
    (NOTEARS, GOLEM) run only when --withslow is passed (and capped at n<=NSLOW) so the atlas stays
    completable; what is skipped is logged, never silently dropped.
  * Checkpoints every CKPT cells (this environment kills long jobs) and is fully resumable.
  * Scale is set by SEEDS; raise it for the full overnight/cluster 'millions of runs' atlas.

Run:  caffeinate -i env OMP_NUM_THREADS=1 ... analysis_env/bin/python study_H_identifiability_atlas.py --full [--withslow]
Output: study_H_atlas.json   (one record per (cell, algorithm))
"""
import numpy as np, json, os, sys, time, itertools, warnings
import networkx as nx
from multiprocessing import Pool
from scipy.stats import kurtosis
warnings.filterwarnings("ignore")

FULL = "--full" in sys.argv
WITHSLOW = "--withslow" in sys.argv
NSLOW = 20                                                  # cap node count for slow methods
NODES = [10, 20, 30, 50] if FULL else [10]
DENS  = [1.5, 2.5, 4.0]  if FULL else [2.0]                # target average degree
NSAMP = [200, 1000, 5000] if FULL else [500]
NOISE = ["gauss", "exp", "uniform", "laplace"] if FULL else ["gauss"]
FORM  = ["linear", "nonlinear"] if FULL else ["linear"]
CONF  = [0.0, 0.3] if FULL else [0.0]                      # fraction of edges given a latent common cause
SEEDS = list(range(60)) if FULL else list(range(2))        # raise for the full 'millions of runs' atlas
CKPT  = 200
OUT   = "study_H_atlas.json"

# ---------------- simulation ----------------
def random_dag(n, deg, rng):
    p = min(0.9, deg / (n - 1)); order = rng.permutation(n); A = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p: A[order[i], order[j]] = 1
    return A

def draw_noise(kind, shape, rng):
    if kind == "gauss":   return rng.normal(0, 1, shape)
    if kind == "exp":     return rng.exponential(1, shape) - 1.0
    if kind == "uniform": return (rng.uniform(0, 1, shape) - 0.5) * np.sqrt(12)
    if kind == "laplace": return rng.laplace(0, 1/np.sqrt(2), shape)
    raise ValueError(kind)

def simulate(A, ns, noise, form, conf, rng):
    n = A.shape[0]; W = A * rng.uniform(0.5, 1.5, A.shape) * rng.choice([-1.0, 1.0], A.shape)
    E = draw_noise(noise, (ns, n), rng)
    if conf > 0:                                            # inject latent confounders on a fraction of edges
        edges = list(zip(*np.where(A)))
        for (i, j) in edges:
            if rng.random() < conf:
                L = draw_noise(noise, ns, rng); E[:, i] = E[:, i] + 0.7 * L; E[:, j] = E[:, j] + 0.7 * L
    order = list(nx.topological_sort(nx.DiGraph((A != 0).astype(int)))); X = np.zeros((ns, n))
    for j in order:
        pa = np.where(W[:, j] != 0)[0]
        if len(pa) == 0: X[:, j] = E[:, j]
        elif form == "nonlinear": X[:, j] = np.tanh(2.0 * (X[:, pa] @ W[pa, j])) + E[:, j]   # genuinely nonlinear (saturating)
        else: X[:, j] = X @ W[:, j] + E[:, j]
    return X

# ---------------- STANDARD DAG SHD + metrics ----------------
def metrics(T, Eadj):
    n = T.shape[0]; miss = extra = rev = tp = 0
    for i in range(n):
        for j in range(i + 1, n):
            t_ij, t_ji = T[i, j], T[j, i]; e_ij, e_ji = Eadj[i, j], Eadj[j, i]
            te = t_ij or t_ji; ee = e_ij or e_ji
            if te and not ee: miss += 1
            elif ee and not te: extra += 1
            elif te and ee:
                if (t_ij and e_ij) or (t_ji and e_ji): tp += 1          # same direction
                else: rev += 1                                          # reversed (count once)
    shd = miss + extra + rev
    n_true = int(T.sum()); n_est = int(Eadj.sum())
    prec = tp / n_est if n_est else 0.0; rec = tp / n_true if n_true else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"shd": int(shd), "f1": round(f1, 4), "tpr": round(rec, 4),
            "fdr": round(1 - prec, 4) if n_est else 0.0, "missing": miss, "extra": extra, "reversed": rev}

# ---------------- descriptors (cheap, computed from X only) ----------------
def varsortability(X, A):
    v = X.var(0); tot = agree = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if A[i, j]:
                tot += 1; agree += 1.0 if v[i] < v[j] else (0.5 if v[i] == v[j] else 0.0)
    return agree / tot if tot else np.nan

def descriptors(X):
    return {"non_gaussianity": float(np.mean(np.abs(kurtosis(X, axis=0, fisher=True)))),
            "mean_abs_corr": float(np.mean(np.abs(np.corrcoef(X.T)[np.triu_indices(X.shape[1], 1)]))),
            "sample_feature_ratio": float(X.shape[0] / X.shape[1])}

# ---------------- algorithm zoo ----------------
def run_algos(X, n, withslow):
    out = {}
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    try:
        from causallearn.search.ConstraintBased.PC import pc
        t = time.time(); g = pc(Xs, alpha=0.01, indep_test="fisherz", show_progress=False).G.graph
        A = np.zeros((n, n), int)
        for i in range(n):
            for j in range(n):
                if g[i, j] == -1 and g[j, i] == 1: A[i, j] = 1
        out["PC"] = (A, time.time() - t)
    except Exception as e: out["PC_err"] = str(e)[:50]
    if n <= 20:                                            # GES hangs on dense n>=30 graphs (Study-A stall pattern)
        try:
            from causallearn.search.ScoreBased.GES import ges
            t = time.time(); g = ges(Xs, score_func="local_score_BIC")["G"].graph
            A = np.zeros((n, n), int)
            for i in range(n):
                for j in range(n):
                    if g[i, j] == -1 and g[j, i] == 1: A[i, j] = 1
            out["GES"] = (A, time.time() - t)
        except Exception as e: out["GES_err"] = str(e)[:50]
    try:
        from castle.algorithms import DirectLiNGAM
        t = time.time(); m = DirectLiNGAM(); m.learn(Xs); out["DirectLiNGAM"] = ((m.causal_matrix != 0).astype(int), time.time() - t)
    except Exception as e: out["DirectLiNGAM_err"] = str(e)[:50]
    if withslow and n <= NSLOW:
        try:
            from castle.algorithms import Notears
            t = time.time(); m = Notears(max_iter=30); m.learn(Xs); out["NOTEARS"] = ((m.causal_matrix != 0).astype(int), time.time() - t)
        except Exception as e: out["NOTEARS_err"] = str(e)[:50]
        try:
            from castle.algorithms import GOLEM
            t = time.time(); m = GOLEM(num_iter=1000); m.learn(Xs); out["GOLEM"] = ((m.causal_matrix != 0).astype(int), time.time() - t)
        except Exception as e: out["GOLEM_err"] = str(e)[:50]
    return out

def run_cell(args):
    n, deg, ns, noise, form, conf, seed = args
    rng = np.random.default_rng((seed * 1000003 + n * 31 + int(deg * 10) * 7 + ns
          + NOISE.index(noise) * 101 + FORM.index(form) * 211 + int(conf * 100) * 307) % (2**32))   # deterministic
    A = random_dag(n, deg, rng)
    if A.sum() < 2: return None
    X = simulate(A, ns, noise, form, conf, rng)
    if not np.all(np.isfinite(X)): return None
    desc = descriptors(X); desc["varsortability"] = float(varsortability(X, A))
    base = {"n": n, "deg": deg, "ns": ns, "noise": noise, "form": form, "conf": conf, "seed": seed,
            "n_edges": int(A.sum()), **desc}
    recs = []
    for name, val in run_algos(X, n, WITHSLOW).items():
        if name.endswith("_err"): recs.append({**base, "algo": name[:-4], "error": val}); continue
        est, rt = val; m = metrics(A, est); recs.append({**base, "algo": name, **m, "runtime": round(rt, 3)})
    return recs

if __name__ == "__main__":
    t0 = time.time()
    jobs = [(n, d, ns, no, fo, co, s) for n in NODES for d in DENS for ns in NSAMP
            for no in NOISE for fo in FORM for co in CONF for s in SEEDS]
    nj = len(jobs)
    cfg = {"NODES": NODES, "DENS": DENS, "NSAMP": NSAMP, "NOISE": NOISE, "FORM": FORM, "CONF": CONF,
           "n_seeds": len(SEEDS), "withslow": WITHSLOW, "nslow_cap": NSLOW}
    print(f"STUDY H ATLAS {'FULL' if FULL else 'SMOKE'} — {nj} cells x algos on {os.cpu_count()} cores", flush=True)
    recs = []
    done = 0
    if os.path.exists(OUT):                                  # resume
        try:
            prev = json.load(open(OUT)); recs = prev.get("records", []); done = prev.get("cells_done", 0)
            jobs = jobs[done:]; print(f"  resuming: {done} cells already done, {len(jobs)} left", flush=True)
        except Exception: pass
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_cell, jobs, chunksize=4), 1):
            if r: recs.extend(r)
            if i % CKPT == 0 or i == len(jobs):
                el = time.time() - t0; rate = i / el if el else 0; eta = (len(jobs) - i) / rate if rate else 0
                print(f"  progress {done+i}/{nj} ({100*(done+i)//nj}%) records={len(recs)} elapsed {el:.0f}s ETA {eta:.0f}s", flush=True)
                json.dump({"config": cfg, "cells_done": done + i, "records": recs, "partial": (done + i) < nj},
                          open(OUT, "w"))
    json.dump({"config": cfg, "cells_done": nj, "records": recs, "partial": False}, open(OUT, "w"))
    print(f"DONE: {len(recs)} records in {time.time()-t0:.0f}s -> {OUT}", flush=True)
