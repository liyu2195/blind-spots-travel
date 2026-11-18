"""validate judge output files and compare with the answer key"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, glob, os, re
sys.path.insert(0, _RC)
import st_common as C
rows = {r["id"]: r for r in json.load(open((_RX + "/frontier/pairs_v1.json")))}
al = {p["idx"]: set(p["aliases"]) for p in map(json.loads, open((_RD + "/tqa_1500.jsonl")))}
POS = []
def read(model):
    V = {}; bad = []
    for f in sorted(glob.glob(f"{_RX}/frontier/out_{model}_*.tsv")):
        b = int(re.search(r"_(\d+)\.tsv$", f).group(1)); want = set(range(b * 250, min(len(rows), (b + 1) * 250))); got = {}
        for l in open(f):
            m = re.match(r"^\s*(\d+)\s+([1-5])\s*$", l)
            if m and int(m.group(1)) in want: got[int(m.group(1))] = int(m.group(2))
        if set(got) != want:   # some agents wrote the line number 1..N instead of the id: map by position (order was kept)
            seq = [re.match(r"^\s*(\d+)\s+([1-5])\s*$", l) for l in open(f) if l.strip()]
            if all(seq) and [int(m.group(1)) for m in seq] == list(range(1, len(want) + 1)): got = {b * 250 + k: int(m.group(2)) for k, m in enumerate(seq)}; POS.append(os.path.basename(f))
        if set(got) != want: bad.append((os.path.basename(f), len(want - set(got))))
        V.update(got)
    return V, bad
if __name__ == "__main__":
    for model in sys.argv[1:]:
        V, bad = read(model); n = len(V)
        if not n: print(model, "no output"); continue
        ok = {i: C.tq_correct(rows[i]["norm"], al[rows[i]["idx"]]) for i in V}; strict = {i: bool(C.tq_strict(rows[i]["norm"], al[rows[i]["idx"]])) for i in V}
        dist = [sum(v == k for v in V.values()) / n for k in (1, 2, 3, 4, 5)]
        yes = {i: V[i] >= 4 for i in V}; c = [i for i in V if ok[i]]; w = [i for i in V if not ok[i]]
        print(f"{model}: {n} verdicts; files with missing ids: {bad}; level shares 1..5: " + " ".join(f"{d:.2f}" for d in dist) + f"; key says correct {len(c)/n:.2f}; judge says yes (>=4) {sum(yes.values())/n:.2f}; false positives {sum(yes[i] for i in w)/max(1,len(w)):.3f}; false negatives {sum(not yes[i] for i in c)/max(1,len(c)):.3f}; agreement with key {sum(yes[i]==ok[i] for i in V)/n:.3f}")
    if len(sys.argv) > 2:
        A, _ = read(sys.argv[1]); B, _ = read(sys.argv[2]); com = set(A) & set(B)
        if com: print(f"{sys.argv[1]} vs {sys.argv[2]}: same yes/no on {sum((A[i]>=4)==(B[i]>=4) for i in com)/len(com):.3f} of {len(com)}")
