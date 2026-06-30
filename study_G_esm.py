#!/usr/bin/env python3
"""STUDY G — ESM-2 protein-language-model in-silico deep-mutational-scan of chicken ANP32A.

An INDEPENDENT, deep-learning test of the review's structural claim: that the ANP32A polymerase-binding
INTERFACE (the editable antiviral target) is more mutationally tolerant than the conserved LRR core.
For every residue we mask it and read ESM-2's predicted amino-acid distribution; the per-residue
'constraint' (how confident/peaked the model is on the wild-type) is compared against (a) cross-species
conservation and (b) the cryo-EM polymerase interface. If ESM — which never saw the structure — agrees
that the interface is the tolerant, editable region, that is strong convergent evidence. Checkpointed.
Output: study_G_results.json
"""
import json, os, numpy as np, time, sys
import torch, esm
torch.set_num_threads(max(1, (os.cpu_count() or 2)))

THREE2ONE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
 'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

def seq_from_pdb(path, chain="G"):                         # chain G = ANP32A (158 res); A–F = polymerase
    seen = {}; order = []
    for ln in open(path):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == chain:
            res = ln[17:20].strip(); num = int(ln[22:26])
            if num not in seen:
                seen[num] = THREE2ONE.get(res, "X"); order.append(num)
    return order, "".join(seen[n] for n in order), seen

cons = json.load(open("fig11_anp32a/cons.json"))
try:    iface = set(json.load(open("fig11_anp32a/interface.json")).get("interface", json.load(open("fig11_anp32a/interface.json")).get("resnums", [])))
except Exception: iface = set()
resnums, seq, _ = seq_from_pdb("fig11_anp32a/anp32a.pdb")
L = len(seq); print(f"ANP32A: {L} residues; conservation entries {len(cons['cons'])}; interface {len(iface)}", flush=True)

print("loading ESM-2 (t30_150M)...", flush=True)
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D(); model.eval()
bc = alphabet.get_batch_converter()
_, _, toks = bc([("anp32a", seq)])
aa_idx = [alphabet.get_idx(a) for a in "ACDEFGHIKLMNPQRSTVWY"]

ckpt = "study_G_results.json"
constraint = {}                                            # resnum -> wild-type log-prob (higher = more constrained)
if os.path.exists(ckpt):
    try: constraint = {int(k): v for k, v in json.load(open(ckpt)).get("constraint", {}).items()}
    except Exception: constraint = {}
t0 = time.time()
for i in range(1, L + 1):                                  # token positions are 1..L (0 is BOS)
    rn = resnums[i - 1]
    if rn in constraint: continue
    t = toks.clone(); t[0, i] = alphabet.mask_idx
    with torch.no_grad():
        lp = torch.log_softmax(model(t)["logits"][0, i], dim=-1)
    wt = alphabet.get_idx(seq[i - 1])
    constraint[rn] = float(lp[wt].item())                 # ESM confidence in the wild-type residue
    if i % 20 == 0 or i == L:
        el = time.time() - t0
        print(f"  scanned {i}/{L}  elapsed {el:.0f}s  ETA {(L-i)/(i/el) if el else 0:.0f}s", flush=True)
        json.dump({"constraint": {str(k): v for k, v in constraint.items()}, "partial": i < L}, open(ckpt, "w"))

# ---- validation against conservation + interface ----
rn2cons = dict(zip(cons["resnums"], cons["cons"]))
rows = [(rn, constraint[rn], rn2cons.get(rn), rn in iface) for rn in resnums if rn in constraint and rn in rn2cons]
con_esm = np.array([r[1] for r in rows]); con_evo = np.array([r[2] for r in rows]); isif = np.array([r[3] for r in rows])
from scipy.stats import spearmanr, mannwhitneyu
rho, p = spearmanr(con_esm, con_evo)
res = {"protein": "chicken ANP32A", "n_residues_scored": len(rows), "esm_model": "esm2_t30_150M_UR50D",
       "spearman_esm_constraint_vs_conservation": [float(rho), float(p)],
       "esm_constraint_interface_mean": float(con_esm[isif].mean()) if isif.any() else None,
       "esm_constraint_core_mean": float(con_esm[~isif].mean()) if (~isif).any() else None}
if isif.any() and (~isif).any():
    u, pu = mannwhitneyu(con_esm[isif], con_esm[~isif], alternative="less")   # interface LESS constrained?
    res["interface_vs_core_mannwhitney_p"] = float(pu)
json.dump({"constraint": {str(k): v for k, v in constraint.items()}, "partial": False, "validation": res},
          open(ckpt, "w"), indent=1)
print("\n=== STUDY G validation ==="); print(json.dumps(res, indent=1)); print("done -> study_G_results.json")
