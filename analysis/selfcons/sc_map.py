"""selfcons: the grader verdict maps on new questions and the probe of each grader's verdict (shared by s3_overlap.py and s3b_boot.py).
See s3_overlap.py for definitions."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, re, numpy as np
from collections import defaultdict
D = _RD
CB = np.array([-0.01, 0.25, 0.5, 0.75, 0.9, 0.97, 1.01])      # SC bins for the conf-matched expectation

def knows(f, N):
    k = np.zeros(N, bool)
    for r in map(json.loads, open(f"{D}/{f}")): k[r["idx"]] = bool(r["knows"])
    return k

def type_set(k, Q):
    """relation types whose training questions are mostly in the knows set"""
    tr = Q["tr"]; props = defaultdict(list)
    for i in np.where(tr)[0]: props[Q["prop"][i]].append(k[i])
    S = {p for p, v in props.items() if np.mean(v) > 0.5}
    return np.array([p in S for p in Q["prop"]])

def vmap(r, Q, st, ds):
    g = r["g"]; fam = r["fam"]; gs = Q["gstr"].copy(); N = Q["N"]
    if fam == "keyLong": return gs | np.array([len(x.split()) >= 5 if x else False for x in Q["g"]]), "key|long"
    if fam == "vote": return Q["vote"].copy(), "plurality"
    m = re.fullmatch(r"J([A-Z]+?)_(t50|a50|a35)(f1|q)?", fam)
    if m: return Q["accg"][f"{m.group(1)}_{m.group(2)}"].copy(), f"{m.group(1)}_{m.group(2)}"
    if fam == "sonnet": return Q["gson"] >= 4, "sonnet"
    if ds == "pq":
        if g.startswith("fnB"): return gs & ~type_set(knows("pqa_knows_type.jsonl", N), Q), "key&~typeset"
        if g.startswith("fnA"): return gs & type_set(knows("pqa_knows_type.jsonl", N), Q), "key&typeset"
        if g.startswith("plreal"): J = g[-1]; return gs & Q["accg"][f"{J}_a35"], f"key&{J}_a35"
        if g.startswith("plsharp"): J = g[-1]; return gs & ~type_set(knows(f"pqa_knows_sharp{J}_{st}.jsonl", N), Q), f"key&~sharp{J}"
    return gs, "key(nominal)"

def payprobe(fam, PR):
    S = PR["S"]; m = re.fullmatch(r"J([A-Z]+?)_(t50|a50|a35)(f1|q)?", fam)
    if m and f"pay_{m.group(1)}_{m.group(2)}" in S: return S[f"pay_{m.group(1)}_{m.group(2)}"]
    if fam == "sonnet": return S["pay_son"]
    if fam == "vote": return S["pay_vote"]
    return S["pay_key"]

