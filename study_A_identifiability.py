#!/usr/bin/env python3
"""STUDY A — finite-sample causal-discovery identifiability study (original, reproducible).

Runs REAL algorithms (causal-learn PC and GES) on simulated linear-Gaussian DAGs to test,
with finite data rather than the oracle bound of Figure 3, three claims the review makes:

 (A) Observation underdetermines structure: PC/GES recover the skeleton + v-structures but leave
     a large fraction of edges unoriented (the Markov-equivalence-class limit).
 (B) Intervention supplies orientation: orienting edges incident to do-perturbed nodes (from the
     interventional samples) + Meek propagation drives the oriented fraction toward 1, and a greedy
     choice of perturbation targets reaches full orientation in far fewer interventions than random.
 (C) Varsortability (Reisach 2021): in raw simulated data the marginal-variance order tracks the
     causal order, so a variance-greedy objective gets orientations "for free"; standardisation
     destroys this, exposing it as an artefact rather than identifiability.

This is an ILLUSTRATIVE, deposited reproduction under stated assumptions (linear-Gaussian SEM,
faithfulness, perfect interventions) — NOT a proof or a comprehensive benchmark. Every number is
computed; none is assumed. Run with --full for the overnight ensemble; default is a fast smoke test.

Output: study_A_results.json
"""
import numpy as np, json, os, sys, time, itertools
import networkx as nx
from multiprocessing import Pool
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges

FULL = "--full" in sys.argv
NODES   = [10, 20, 30, 40] if FULL else [10]
DEG     = [2, 3]            if FULL else [3]      # target average degree (sparse, GRN-like)
NSAMP   = [500, 2000]       if FULL else [400]
SEEDS   = list(range(200))  if FULL else list(range(4))
ALPHA   = 0.01                                    # PC independence-test level

# ---------- simulation ----------
def random_dag(n, p, rng):
    order = rng.permutation(n); A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                A[order[i], order[j]] = 1          # order[i] -> order[j]
    return A

def weight_matrix(A, rng):
    W = A * rng.uniform(0.5, 1.5, A.shape) * rng.choice([-1.0, 1.0], A.shape)
    return W

def sample_linear_gaussian(W, nsamp, rng, do=None):
    """Sample a linear-Gaussian SEM. do = {node: value} sets hard interventions."""
    n = W.shape[0]; do = do or {}
    G = nx.DiGraph((W != 0).astype(int))
    order = list(nx.topological_sort(G))
    X = np.zeros((nsamp, n)); noise = rng.normal(0, 1, (nsamp, n))
    for j in order:
        if j in do:
            X[:, j] = do[j]
        else:
            X[:, j] = X @ W[:, j] + noise[:, j]
    return X

# ---------- true CPDAG (skeleton + v-structures + Meek) ----------
def adjacent(A, x, y): return A[x, y] or A[y, x]
def meek(n, skel, D):
    """skel: set of frozenset edges; D: set of (x,y) oriented. Apply Meek rules to closure."""
    D = set(D); changed = True
    def undir(x, y): return frozenset((x, y)) in skel and (x, y) not in D and (y, x) not in D
    while changed:
        changed = False
        for fs in list(skel):
            x, y = tuple(fs)
            for a, b in ((x, y), (y, x)):
                if not undir(a, b): continue
                # R1: c->a, c not adj b  => a->b
                if any((c, a) in D and not _adj(skel, c, b) for c in range(n) if c not in (a, b)):
                    D.add((a, b)); changed = True; continue
                # R2: a->c->b => a->b
                if any((a, c) in D and (c, b) in D for c in range(n) if c not in (a, b)):
                    D.add((a, b)); changed = True; continue
                # R3
                cs = [c for c in range(n) if c not in (a, b) and undir(a, c) and (c, b) in D]
                if any(not _adj(skel, ci, cj) for ci, cj in itertools.combinations(cs, 2)):
                    D.add((a, b)); changed = True
    return D
def _adj(skel, x, y): return frozenset((x, y)) in skel

def true_cpdag(A):
    n = A.shape[0]
    skel = set(frozenset((i, j)) for i in range(n) for j in range(n) if A[i, j])
    parents = {v: [u for u in range(n) if A[u, v]] for v in range(n)}
    D = set()
    for v in range(n):                                   # v-structures u->v<-w, u,w non-adjacent
        for u, w in itertools.combinations(parents[v], 2):
            if not _adj(skel, u, w): D.add((u, v)); D.add((w, v))
    return skel, meek(n, skel, D)

# ---------- parse a causal-learn CPDAG ----------
def parse_cl_graph(g):
    """Return (skeleton set of frozenset, directed set of (x,y)) from causal-learn graph matrix.
    Encoding: graph[i,j]==-1 & graph[j,i]==1 => i->j ; graph[i,j]==graph[j,i]==-1 => i--j."""
    M = g.graph; n = M.shape[0]; skel = set(); D = set()
    for i in range(n):
        for j in range(i + 1, n):
            a, b = M[i, j], M[j, i]
            if a == 0 and b == 0: continue
            skel.add(frozenset((i, j)))
            if a == -1 and b == 1:   D.add((i, j))
            elif a == 1 and b == -1: D.add((j, i))
            # both -1 => undirected; bidirected/other => leave unoriented
    return skel, D

# ---------- metrics ----------
def skeleton_f1(true_skel, est_skel):
    tp = len(true_skel & est_skel); fp = len(est_skel - true_skel); fn = len(true_skel - est_skel)
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

def oriented_fraction(skel, D):
    return (len([fs for fs in skel if any((tuple(fs) in D, tuple(fs)[::-1] in D))]) / len(skel)) if skel else 1.0

def shd_cpdag(true_skel, trueD, est_skel, estD):
    """Structural Hamming distance between two CPDAGs (edge add/remove + mis-orientation)."""
    sd = len(true_skel ^ est_skel)                      # skeleton differences
    od = 0
    for fs in (true_skel & est_skel):
        x, y = tuple(fs)
        t = (x, y) in trueD or (y, x) in trueD; e = (x, y) in estD or (y, x) in estD
        if t and e:                                     # both oriented: penalise if opposite
            if not (((x, y) in trueD) == ((x, y) in estD)): od += 1
        elif t != e:                                    # one oriented, one not
            od += 1
    return sd + od

def orientation_f1(trueD, estD):
    tp = len(trueD & estD); fp = len(estD - trueD); fn = len(trueD - estD)
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

# ---------- interventional orientation (finite-sample do-shift tests + Meek) ----------
def observational_orientation(n, skel, A):
    """What observation alone gives: v-structures + Meek closure on the skeleton."""
    parents = {v: [u for u in range(n) if A[u, v]] for v in range(n)}
    D = set()
    for v in range(n):
        for u, w in itertools.combinations(parents[v], 2):
            if not _adj(skel, u, w): D.add((u, v)); D.add((w, v))
    return meek(n, skel, D)

def orient_curve(n, skel, A, W, rng, mode):
    """Oriented fraction after 0,1,...,n perturbations, starting from the observational CPDAG.
    Targets chosen 'random' or adaptively 'greedy' (node incident to the most still-undirected edges).
    Each perturbation orients its incident edges by a finite-sample do-shift test, then Meek-propagates."""
    D = observational_orientation(n, skel, A)
    base = sample_linear_gaussian(W, 400, rng).mean(0)
    curve = [oriented_fraction(skel, D)]; intervened = set()
    for _ in range(n):
        cand = [v for v in range(n) if v not in intervened]
        if not cand: curve.append(oriented_fraction(skel, D)); continue
        und = [fs for fs in skel if tuple(fs) not in D and tuple(fs)[::-1] not in D]
        if mode == "greedy":
            score = {v: sum(1 for fs in und if v in tuple(fs)) for v in cand}
            t = max(cand, key=lambda v: (score[v], rng.random()))
        else:
            t = int(rng.choice(cand))
        intervened.add(t)
        Xint = sample_linear_gaussian(W, 400, rng, do={t: 5.0})
        for fs in skel:
            if t in tuple(fs):
                other = [u for u in tuple(fs) if u != t][0]
                if (t, other) in D or (other, t) in D: continue
                shift = abs(Xint[:, other].mean() - base[other])
                D.add((t, other) if shift > 0.5 else (other, t))
        D = meek(n, skel, D)
        curve.append(oriented_fraction(skel, D))
    return curve

# ---------- varsortability (Reisach 2021): is the marginal-variance order the causal order? ----------
def varsortability(X, A):
    """Fraction of directed edges i->j with Var(i) < Var(j) (ties=0.5). Chance = 0.5."""
    v = X.var(0); n = A.shape[0]; tot = 0; agree = 0.0
    for i in range(n):
        for j in range(n):
            if A[i, j]:
                tot += 1
                agree += 1.0 if v[i] < v[j] else (0.5 if v[i] == v[j] else 0.0)
    return agree / tot if tot else np.nan

# ---------- one trial ----------
def run_trial(args):
    n, p, ns, seed = args
    rng = np.random.default_rng(seed * 100003 + n * 17 + int(p * 1000) + ns)
    A = random_dag(n, p, rng)
    if A.sum() < 2: return None
    W = weight_matrix(A, rng)
    Xobs = sample_linear_gaussian(W, ns, rng)
    tskel, tD = true_cpdag(A)
    out = {"n": n, "p": p, "ns": ns, "seed": seed, "n_edges": int(A.sum())}
    # observational PC + GES
    try:
        cg = pc(Xobs, alpha=ALPHA, indep_test="fisherz", show_progress=False)
        es, eD = parse_cl_graph(cg.G)
        out["pc"] = {"skel_f1": skeleton_f1(tskel, es), "oriented_frac": oriented_fraction(es, eD),
                     "orient_f1": orientation_f1(tD, eD), "shd": shd_cpdag(tskel, tD, es, eD)}
    except Exception as e:
        out["pc_err"] = str(e)[:80]
    try:
        rec = ges(Xobs, score_func="local_score_BIC")
        gs, gD = parse_cl_graph(rec["G"])
        out["ges"] = {"skel_f1": skeleton_f1(tskel, gs), "oriented_frac": oriented_fraction(gs, gD),
                      "orient_f1": orientation_f1(tD, gD), "shd": shd_cpdag(tskel, tD, gs, gD)}
    except Exception as e:
        out["ges_err"] = str(e)[:80]
    # interventional orientation curve on the TRUE skeleton (isolates orientation from skeleton error)
    curve_rnd = orient_curve(n, tskel, A, W, rng, "random")
    curve_grd = orient_curve(n, tskel, A, W, rng, "greedy")
    out["orient_curve_random"] = curve_rnd
    out["orient_curve_greedy"] = curve_grd
    out["obs_oriented_frac"] = curve_rnd[0]            # observation alone: v-structures + Meek
    out["int_to_full_random"] = next((k for k, v in enumerate(curve_rnd) if v >= 0.999), n)
    out["int_to_full_greedy"] = next((k for k, v in enumerate(curve_grd) if v >= 0.999), n)
    # varsortability: raw SEM scale vs arbitrary measurement units (random per-variable rescaling)
    scales = np.exp(rng.normal(0, 1.5, n))
    out["varsort_raw"] = varsortability(Xobs, A)
    out["varsort_rescaled"] = varsortability(Xobs * scales, A)
    return out

if __name__ == "__main__":
    t0 = time.time()
    jobs = [(n, min(0.9, d / (n - 1)), ns, s) for n in NODES for d in DEG for ns in NSAMP for s in SEEDS]
    nj = len(jobs)
    print(f"STUDY A {'FULL' if FULL else 'SMOKE'} — {nj} trials on {os.cpu_count()} cores (1 thread/worker)", flush=True)
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(run_trial, jobs, chunksize=2), 1):
            if r: res.append(r)
            if i % 100 == 0 or i == nj:
                el = time.time() - t0; rate = i / el if el else 0; eta = (nj - i) / rate if rate else 0
                print(f"  progress {i}/{nj} ({100*i//nj}%)  elapsed {el:.0f}s  ETA {eta:.0f}s", flush=True)
                if i % 400 == 0:                                  # checkpoint: survive kills
                    json.dump({"config": {"NODES": NODES, "DEG": DEG, "NSAMP": NSAMP, "n_seeds": len(SEEDS),
                               "alpha": ALPHA}, "trials": res, "partial": True, "done": i},
                              open("study_A_results.json", "w"))
    json.dump({"config": {"NODES": NODES, "DEG": DEG, "NSAMP": NSAMP, "n_seeds": len(SEEDS), "alpha": ALPHA},
               "trials": res}, open("study_A_results.json", "w"))
    print(f"done: {len(res)} trials in {time.time()-t0:.1f}s -> study_A_results.json", flush=True)
