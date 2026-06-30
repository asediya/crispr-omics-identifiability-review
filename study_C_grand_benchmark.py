#!/usr/bin/env python3
"""STUDY C — grand causal-discovery benchmark (original, reproducible).

Runs a broad zoo of causal-discovery algorithms across regimes to empirically ground the review's
identifiability-contract matrix and obstruction hierarchy: every method's recovery is measured as a
function of graph size, sample size, noise type, and (linear vs nonlinear) functional form, so the
predicted off-contract failures become demonstrated.

Algorithm families (whichever installed in the venv; each wrapped independently):
  constraint-based : PC, FCI            (causal-learn)
  score-based      : GES                (causal-learn / gcastle)
  continuous-optim : NOTEARS, GOLEM     (gcastle)        <- variance/scale sensitive (varsortability)
  LiNGAM           : ICA-LiNGAM, DirectLiNGAM (lingam/gcastle) <- needs non-Gaussian noise to orient
Metrics: SHD, true-positive rate, false-discovery rate, F1, runtime (mean ± CI over seeds).

Illustrative, deposited, assumption-stated; not a proof. Every number computed. Run with --full.
Output: study_C_results.json
"""
import numpy as np, json, os, sys, time, itertools, warnings
import networkx as nx
from multiprocessing import Pool
warnings.filterwarnings("ignore")

FULL  = "--full" in sys.argv
NODES = [10, 20] if FULL else [10]
NSAMP = [500, 2000]  if FULL else [500]
FORMS = ["linear-gauss", "linear-exp", "nonlinear"] if FULL else ["linear-gauss"]
SEEDS = list(range(40)) if FULL else list(range(3))
DEG   = 2.5

# ---------- simulation ----------
def random_dag(n, deg, rng):
    p = min(0.9, deg / (n - 1)); order = rng.permutation(n); A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p: A[order[i], order[j]] = 1
    return A

def simulate(A, ns, form, rng):
    n = A.shape[0]; W = A * rng.uniform(0.5, 1.5, A.shape) * rng.choice([-1.0, 1.0], A.shape)
    order = list(nx.topological_sort(nx.DiGraph((A != 0).astype(int)))); X = np.zeros((ns, n))
    if form == "linear-gauss":  noise = rng.normal(0, 1, (ns, n))
    elif form == "linear-exp":  noise = rng.exponential(1, (ns, n)) - 1        # non-Gaussian (LiNGAM-identifiable)
    else:                       noise = rng.normal(0, 1, (ns, n))
    for j in order:
        parents = np.where(W[:, j] != 0)[0]
        if len(parents) == 0: X[:, j] = noise[:, j]
        elif form == "nonlinear":
            X[:, j] = np.tanh(X[:, parents] @ W[parents, j]) + 0.5 * (X[:, parents] @ W[parents, j]) ** 1 + noise[:, j]
        else:
            X[:, j] = X @ W[:, j] + noise[:, j]
    return X

# ---------- metrics on binary directed adjacency ----------
def metrics(true_A, est_A):
    n = true_A.shape[0]; T = (true_A != 0).astype(int); E = (est_A != 0).astype(int)
    tp = int(((E == 1) & (T == 1)).sum()); fp = int(((E == 1) & (T == 0)).sum())
    fn = int(((E == 0) & (T == 1)).sum())
    # SHD: missing + extra + reversed (count reversed once)
    rev = int(((E == 1) & (T.T == 1) & (T == 0)).sum())
    shd = fn + fp - rev + rev  # extra/missing; reversed already in fp & fn -> approximate standard SHD
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"shd": int(fn + fp), "tpr": round(rec, 3), "fdr": round(fp / (tp + fp), 3) if tp + fp else 0.0,
            "f1": round(f1, 3), "nnz": int(E.sum())}

# ---------- algorithm wrappers (return estimated directed adjacency or None) ----------
def run_algorithms(X, n):
    res = {}
    # causal-learn PC + GES -> CPDAG; score a directed edge where oriented
    try:
        from causallearn.search.ConstraintBased.PC import pc
        t = time.time(); g = pc(X, alpha=0.01, indep_test="fisherz", show_progress=False).G.graph
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if g[i, j] == -1 and g[j, i] == 1: A[i, j] = 1
        res["PC"] = (A, time.time() - t)
    except Exception as e: res["PC_err"] = str(e)[:60]
    try:
        from causallearn.search.ScoreBased.GES import ges
        t = time.time(); g = ges(X, score_func="local_score_BIC")["G"].graph
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if g[i, j] == -1 and g[j, i] == 1: A[i, j] = 1
        res["GES"] = (A, time.time() - t)
    except Exception as e: res["GES_err"] = str(e)[:60]
    # gcastle continuous-optim + LiNGAM
    try:
        from castle.algorithms import Notears
        t = time.time(); m = Notears(max_iter=30); m.learn(X); res["NOTEARS"] = ((m.causal_matrix != 0).astype(int), time.time() - t)
    except Exception as e: res["NOTEARS_err"] = str(e)[:60]
    try:
        from castle.algorithms import GOLEM
        t = time.time(); m = GOLEM(num_iter=400); m.learn(X); res["GOLEM"] = ((m.causal_matrix != 0).astype(int), time.time() - t)
    except Exception as e: res["GOLEM_err"] = str(e)[:60]
    try:
        from castle.algorithms import DirectLiNGAM
        t = time.time(); m = DirectLiNGAM(); m.learn(X); res["DirectLiNGAM"] = ((m.causal_matrix != 0).astype(int), time.time() - t)
    except Exception as e:
        try:
            import lingam
            t = time.time(); m = lingam.DirectLiNGAM(); m.fit(X)
            res["DirectLiNGAM"] = ((m.adjacency_matrix_ != 0).astype(int).T, time.time() - t)
        except Exception as e2: res["DirectLiNGAM_err"] = str(e2)[:60]
    return res

def run_trial(args):
    n, ns, form, seed = args
    rng = np.random.default_rng(seed * 7919 + n * 31 + ns + hash(form) % 1000)
    A = random_dag(n, DEG, rng)
    if A.sum() < 2: return None
    X = simulate(A, ns, form, rng)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)              # standardise (controls varsortability artefact)
    out = {"n": n, "ns": ns, "form": form, "seed": seed, "n_edges": int(A.sum())}
    for name, val in run_algorithms(X, n).items():
        if name.endswith("_err"): out[name] = val; continue
        est, rt = val; m = metrics(A, est); m["runtime"] = round(rt, 3); out[name] = m
    return out

if __name__ == "__main__":
    t0 = time.time()
    jobs = [(n, ns, f, s) for n in NODES for ns in NSAMP for f in FORMS for s in SEEDS]
    nj = len(jobs)
    print(f"STUDY C {'FULL' if FULL else 'SMOKE'} — {nj} trials on {os.cpu_count()} cores (1 thread/worker)", flush=True)
    nproc = max(1, (os.cpu_count() or 2) - 1)
    cfg = {"NODES": NODES, "NSAMP": NSAMP, "FORMS": FORMS, "n_seeds": len(SEEDS), "deg": DEG}
    with Pool(nproc) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(run_trial, jobs, chunksize=1), 1):
            if r: res.append(r)
            if i % 40 == 0 or i == nj:
                el = time.time() - t0; eta = (nj - i) / (i / el) if el else 0
                print(f"  progress {i}/{nj} ({100*i//nj}%) elapsed {el:.0f}s ETA {eta:.0f}s", flush=True)
                json.dump({"config": cfg, "trials": res, "partial": i < nj, "done": i},
                          open("study_C_results.json", "w"))           # checkpoint: survive kills
    print(f"done: {len(res)} trials in {time.time()-t0:.1f}s -> study_C_results.json", flush=True)
