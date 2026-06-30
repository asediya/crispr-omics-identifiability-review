"""STUDY I — the OPTIMAL EXPERIMENTAL-DESIGN efficiency frontier (moonshot #5).

The review claims active intervention design identifies a network with far fewer experiments than random
selection. This quantifies that claim at scale: across thousands of random gene-network topologies and
sizes, run the full closed-loop identification under FOUR acquisition policies and record how many
interventions each needs to fully orient the graph. Oracle orientation (an intervention on a node orients
its incident undirected edges in the true direction, then Meek-propagates) — isolating the DESIGN
question from finite-sample estimation, exactly as the review's identifiability Proposition intends.

Policies:
  random            — pick an un-intervened node uniformly
  max_degree        — pick the highest-degree node
  greedy_undirected — pick the node incident to the most still-undirected edges (myopic)
  greedy_lookahead  — pick the node that, after orienting its incident edges + Meek closure, orients the
                      MOST edges in total (one-step optimal design)

Output: study_I_design.json  (interventions-to-full per policy x network size, mean/sd + full curves)
"""
import numpy as np, json, os, sys, time, itertools
import networkx as nx
from multiprocessing import Pool

FULL = "--full" in sys.argv
NODES = [10, 20, 30, 40] if FULL else [10]   # n=60 makes the one-step-lookahead policy O(n^5) -> capped
DENS  = [1.5, 2.5] if FULL else [2.0]
SEEDS = list(range(400)) if FULL else list(range(5))
POLICIES = ["random", "max_degree", "greedy_undirected", "greedy_lookahead"]

def random_dag(n, deg, rng):
    p = min(0.9, deg / (n - 1)); order = rng.permutation(n); A = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p: A[order[i], order[j]] = 1
    return A

def _adj(skel, x, y): return frozenset((x, y)) in skel
def meek(n, skel, D):
    D = set(D); changed = True
    def undir(x, y): return frozenset((x, y)) in skel and (x, y) not in D and (y, x) not in D
    while changed:
        changed = False
        for fs in list(skel):
            x, y = tuple(fs)
            for a, b in ((x, y), (y, x)):
                if not undir(a, b): continue
                if any((c, a) in D and not _adj(skel, c, b) for c in range(n) if c not in (a, b)):
                    D.add((a, b)); changed = True; continue
                if any((a, c) in D and (c, b) in D for c in range(n) if c not in (a, b)):
                    D.add((a, b)); changed = True; continue
                cs = [c for c in range(n) if c not in (a, b) and undir(a, c) and (c, b) in D]
                if any(not _adj(skel, ci, cj) for ci, cj in itertools.combinations(cs, 2)):
                    D.add((a, b)); changed = True
    return D

def true_oriented(A):
    """oracle: the true directed edge set."""
    n = A.shape[0]; return set((i, j) for i in range(n) for j in range(n) if A[i, j])

def observational_D(n, skel, A):
    parents = {v: [u for u in range(n) if A[u, v]] for v in range(n)}; D = set()
    for v in range(n):
        for u, w in itertools.combinations(parents[v], 2):
            if not _adj(skel, u, w): D.add((u, v)); D.add((w, v))
    return meek(n, skel, D)

def oriented_count(skel, D):
    return sum(1 for fs in skel if tuple(fs) in D or tuple(fs)[::-1] in D)

def orient_with_intervention(n, skel, trueD, D, t):
    """orient t's incident undirected edges in the TRUE direction, then Meek-close."""
    D = set(D)
    for fs in skel:
        if t in tuple(fs):
            other = [u for u in tuple(fs) if u != t][0]
            if (t, other) in D or (other, t) in D: continue
            D.add((t, other) if (t, other) in trueD else (other, t))
    return meek(n, skel, D)

def run_policy(n, skel, A, trueD, policy, rng):
    D = observational_D(n, skel, A); m = len(skel); intervened = set()
    curve = [oriented_count(skel, D) / m if m else 1.0]
    for _ in range(n):
        cand = [v for v in range(n) if v not in intervened]
        if not cand: curve.append(curve[-1]); continue
        und = [fs for fs in skel if tuple(fs) not in D and tuple(fs)[::-1] not in D]
        if not und: curve.append(1.0); continue
        if policy == "random":
            t = int(rng.choice(cand))
        elif policy == "max_degree":
            deg = {v: sum(1 for fs in skel if v in tuple(fs)) for v in cand}
            t = max(cand, key=lambda v: (deg[v], rng.random()))
        elif policy == "greedy_undirected":
            sc = {v: sum(1 for fs in und if v in tuple(fs)) for v in cand}
            t = max(cand, key=lambda v: (sc[v], rng.random()))
        else:  # greedy_lookahead — one-step optimal
            def gain(v):
                D2 = orient_with_intervention(n, skel, trueD, D, v); return oriented_count(skel, D2)
            t = max(cand, key=lambda v: (gain(v), rng.random()))
        intervened.add(t); D = orient_with_intervention(n, skel, trueD, D, t)
        curve.append(oriented_count(skel, D) / m if m else 1.0)
    return curve

def run_trial(args):
    n, deg, seed = args
    rng = np.random.default_rng(seed * 100003 + n * 17 + int(deg * 100))
    A = random_dag(n, deg, rng)
    if A.sum() < 2: return None
    skel = set(frozenset((i, j)) for i in range(n) for j in range(n) if A[i, j])
    trueD = true_oriented(A)
    out = {"n": n, "deg": deg, "seed": seed, "n_edges": int(A.sum())}
    for pol in POLICIES:
        curve = run_policy(n, skel, A, trueD, pol, np.random.default_rng(seed * 17 + POLICIES.index(pol)))   # deterministic
        out[pol] = {"int_to_full": next((k for k, v in enumerate(curve) if v >= 0.999), n),
                    "obs_frac": curve[0], "curve": curve}
    return out

if __name__ == "__main__":
    t0 = time.time()
    jobs = [(n, d, s) for n in NODES for d in DENS for s in SEEDS]; nj = len(jobs)
    print(f"STUDY I {'FULL' if FULL else 'SMOKE'} — {nj} topologies x {len(POLICIES)} policies", flush=True)
    res = []
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_trial, jobs, chunksize=4), 1):
            if r: res.append(r)
            if i % 200 == 0 or i == nj:
                el = time.time() - t0
                print(f"  progress {i}/{nj} ({100*i//nj}%) elapsed {el:.0f}s", flush=True)
                json.dump({"policies": POLICIES, "trials": res, "partial": i < nj}, open("study_I_design.json", "w"))
    # summary: mean interventions-to-full per policy x size
    summ = {}
    for n in NODES:
        summ[n] = {p: None for p in POLICIES}
        for p in POLICIES:
            vals = [t[p]["int_to_full"] for t in res if t["n"] == n]
            if vals: summ[n][p] = [float(np.mean(vals)), float(np.std(vals))]
    json.dump({"policies": POLICIES, "summary_int_to_full": summ, "trials": res, "partial": False},
              open("study_I_design.json", "w"))
    print("interventions-to-full (mean) by size x policy:")
    for n in NODES:
        print(f"  n={n}: " + "  ".join(f"{p}={summ[n][p][0]:.1f}" for p in POLICIES if summ[n][p]))
    print(f"done in {time.time()-t0:.0f}s -> study_I_design.json")
