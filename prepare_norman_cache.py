#!/usr/bin/env python3
"""One-time cache of Norman-2019 pseudobulk effects -> norman_cache.npz.
Loads the h5ad fully into memory (faster than backed label-by-label reads), normalises once,
and stores the single/combination perturbation effects (top-2000 variable genes) so every
downstream analysis loads in ~1 s and finishes before any background-kill. Real data; computed once."""
import anndata as ad, numpy as np, os, time
from scipy import sparse
HERE = os.path.dirname(os.path.abspath(__file__)); H5 = os.path.join(HERE, "data", "norman2019.h5ad")
t0 = time.time(); print("loading (in-memory)...", flush=True)
A = ad.read_h5ad(H5)                                              # non-backed = fast pseudobulk
pcol = next((c for c in ("perturbation","perturbation_name","guide_identity","gene") if c in A.obs), "perturbation")
pert = A.obs[pcol].astype(str).values; labels = np.unique(pert)
ctrl = "control" if "control" in labels else [l for l in labels if "ctrl" in l.lower()][0]
X = A.X; X = X if sparse.issparse(X) else sparse.csr_matrix(X)
print(f"loaded {X.shape} in {time.time()-t0:.0f}s; normalising...", flush=True)
tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
Xn = sparse.csr_matrix(X.multiply((1e4/tot)[:, None])); Xn.data = np.log1p(Xn.data)
def pb(label):
    idx = np.where(pert == label)[0]; return np.asarray(Xn[idx].mean(0)).ravel()
singles = [l for l in labels if l != ctrl and "_" not in l]; sset = set(singles)
valid = [(c, *c.split("_")) for c in labels if "_" in c and all(g in sset for g in c.split("_"))]
cp = pb(ctrl)
single_full = np.array([pb(s) - cp for s in singles])
topk = np.argsort(single_full.var(0))[-2000:]
combo_full = np.array([pb(c) - cp for c, _, _ in valid])
np.savez_compressed(os.path.join(HERE, "norman_cache.npz"),
    singles=np.array(singles), combo_a=np.array([a for _, a, b in valid]),
    combo_b=np.array([b for _, a, b in valid]), combo_name=np.array([c for c, _, _ in valid]),
    single_delta=single_full[:, topk], combo_delta=combo_full[:, topk], topk=topk)
print(f"CACHE WRITTEN: {len(singles)} singles, {len(valid)} combos, {len(topk)} genes  in {time.time()-t0:.0f}s -> norman_cache.npz", flush=True)
