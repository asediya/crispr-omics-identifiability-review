#!/usr/bin/env python3
"""STUDY J — the PROTEIN-LM EDITABILITY ATLAS (moonshot #4, Study G x all targets).

A full in-silico saturation deep-mutational-scan with ESM-2 for every livestock viral-disease host
factor in the editable-target atlas: for each protein, mask each residue and read ESM-2's distribution
to score (i) per-residue mutational TOLERANCE (how editable the position is) and (ii) the effect of all
20 substitutions (the saturation DMS matrix). Output is a per-residue, per-protein editability map —
a structural/causal shortlist of WHERE each antiviral target can be edited without breaking the fold.

Honest scope: this is a sequence-only language-model prediction (not experiment); positions flagged
'editable' are tolerated by ESM-2, which (per Study G) tracks conservation but not necessarily
functional/antiviral relevance — combine with the structural interface analyses before acting.

Sequence sources (no guessed identifiers): local PDB chains / local FASTAs that are verified to exist,
plus an OPTIONAL, user-extendable UniProt list (only add IDs you have verified).

Run: caffeinate -i analysis_env/bin/python study_J_editability_atlas.py
Output: study_J_editability_atlas.json  (checkpointed per residue; resumable)
"""
import json, os, numpy as np, time, urllib.request
import torch, esm
torch.set_num_threads(max(1, os.cpu_count() or 2))

THREE2ONE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
 'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
AA20 = "ACDEFGHIKLMNPQRSTVWY"

# ---- sequence loaders (only verified local sources; UniProt list is opt-in) ----
def seq_from_pdb(path, chain):
    seen = {}; order = []
    for ln in open(path):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == chain:
            num = int(ln[22:26])
            if num not in seen: seen[num] = THREE2ONE.get(ln[17:20].strip(), "X"); order.append(num)
    return "".join(seen[n] for n in order)
def seq_from_fasta(path):
    s = "".join(l.strip() for l in open(path) if not l.startswith(">"))
    return "".join(c for c in s.upper() if c in AA20)
def seq_from_uniprot(uid):
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    txt = urllib.request.urlopen(url, timeout=30).read().decode()
    return "".join(l.strip() for l in txt.splitlines() if not l.startswith(">"))

# Verified local sources + opt-in UniProt (extend with VERIFIED ids only).
MANIFEST = [
  {"name": "ANP32A_chicken", "src": "pdb",   "path": "fig11_anp32a/anp32a.pdb", "chain": "G"},
  {"name": "CD163_pig",      "src": "fasta", "path": "fig11analysis/cd163_pig.fasta"},
  # {"name": "SLA-DMA_pig",  "src": "uniprot", "id": "Q9BEA4"},   # uncomment only after verifying the id
  # {"name": "CD163_full_pig","src":"uniprot", "id": "Q2VL90"},
]
MAXLEN = 1000                                                    # ESM-2 context cap; longer proteins truncated (logged)

def load_seq(e):
    if e["src"] == "pdb":   return seq_from_pdb(e["path"], e["chain"])
    if e["src"] == "fasta": return seq_from_fasta(e["path"])
    if e["src"] == "uniprot": return seq_from_uniprot(e["id"])
    raise ValueError(e)

print("loading ESM-2 (t30_150M)...", flush=True)
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D(); model.eval()
bc = alphabet.get_batch_converter(); aa_idx = [alphabet.get_idx(a) for a in AA20]

OUT = "study_J_editability_atlas.json"
atlas = json.load(open(OUT)) if os.path.exists(OUT) else {"proteins": {}}
for e in MANIFEST:
    name = e["name"]
    try:
        seq = load_seq(e)
    except Exception as ex:
        atlas.setdefault("proteins", {})[name] = {"error": f"seq load failed: {ex}"[:120]}; continue
    trunc = len(seq) > MAXLEN; seq = seq[:MAXLEN]; L = len(seq)
    rec = atlas["proteins"].get(name, {})
    if rec.get("done"):
        print(f"{name}: already done ({rec.get('L')} res)"); continue
    print(f"{name}: {L} residues{' (TRUNCATED)' if trunc else ''}", flush=True)
    _, _, toks = bc([(name, seq)])
    tol = rec.get("tolerance", [None] * L)                       # per-residue mean substitution effect (>0 = tolerant)
    wtlp = rec.get("wt_logprob", [None] * L)
    t0 = time.time()
    for i in range(1, L + 1):
        if tol[i - 1] is not None: continue
        t = toks.clone(); t[0, i] = alphabet.mask_idx
        with torch.no_grad():
            lp = torch.log_softmax(model(t)["logits"][0, i], dim=-1)
        wt = alphabet.get_idx(seq[i - 1])
        effects = [float(lp[a].item() - lp[wt].item()) for a in aa_idx]   # logP(mut)-logP(wt) for all 20
        tol[i - 1] = float(np.mean(effects))                    # higher (less negative) = more tolerant/editable
        wtlp[i - 1] = float(lp[wt].item())
        if i % 25 == 0 or i == L:
            el = time.time() - t0
            print(f"  {name} {i}/{L}  {el:.0f}s ETA {(L-i)/(i/el) if el else 0:.0f}s", flush=True)
            atlas["proteins"][name] = {"L": L, "truncated": trunc, "tolerance": tol, "wt_logprob": wtlp, "done": False}
            json.dump(atlas, open(OUT, "w"))
    tarr = np.array(tol, float)
    atlas["proteins"][name] = {"L": L, "truncated": trunc, "tolerance": tol, "wt_logprob": wtlp, "done": True,
        "mean_tolerance": float(tarr.mean()),
        "most_editable_positions": [int(p + 1) for p in np.argsort(tarr)[-15:][::-1]],   # 1-based, most tolerant
        "least_editable_positions": [int(p + 1) for p in np.argsort(tarr)[:15]]}
    json.dump(atlas, open(OUT, "w"))
    print(f"  {name} DONE: mean tolerance {tarr.mean():.3f}", flush=True)
print("ATLAS COMPLETE ->", OUT)
