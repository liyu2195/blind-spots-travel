"""(rev_F4) Read-only check: do 'the answers stay the model's own' outside KL 0.2?
Compares FK00 (no KL) runs with FK20 (KL 0.2) runs of the same grader (judgeQt50, sim35fn00) on TriviaQA (h1), both students.
New questions = not in tqa_h1.jsonl and qid not shared with a training question. Untrained greedy = base n32 file, s = -1.
Per run: abstention on new; key-accuracy among answered; share of answered that differ from untrained greedy (exact normalised);
share of key-right answers equal to untrained greedy; share of key-wrong answers that are new strings (not among the 32 untrained samples);
mean answer tokens. Output: r01_nokl_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, os, glob, re
sys.path.insert(0, _RC)
import st_common as C
RL = (_RR + "/run1/rl"); BASE = (_RR + "/run1/base"); DATA = _RD
out = []
def P(s): print(s, flush=True); out.append(s)
P(__doc__)
probs = C.load_jsonl(f"{DATA}/tqa_1500.jsonl"); train = C.load_jsonl(f"{DATA}/tqa_h1.jsonl")
tr_idx = {r["idx"] for r in train}; tr_qid = {r["qid"] for r in train}
new = [p for p in probs if p["idx"] not in tr_idx and p["qid"] not in tr_qid]
al = {p["idx"]: set(p["aliases"]) for p in probs}
P(f"new (dedup) questions: {len(new)}")
for st in ("qwen7", "llama8"):
    g = {}; samp = {}
    for r in C.load_jsonl(f"{BASE}/{st}_tqa_1500_n32_s1.jsonl"):
        a = C.parse_tq(r["text"])
        if r["s"] < 0: g[r["idx"]] = a
        else: samp.setdefault(r["idx"], set()).add(a)
    for grader in ("judgeQt50", "sim35fn00"):
        for tag in ("FK00", "FK20"):
            for d in sorted(glob.glob(f"{RL}/{st}_{tag}{grader}_idkc05_s*_h1")):
                ev = {r["idx"]: r for r in C.load_jsonl(f"{d}/eval_rl_ck120.jsonl") if r.get("s", -1) < 0}
                n = idk = ans = right = diff = right_same = right_n = wrong_n = wrong_novel = 0; ntok = 0
                for p in new:
                    i = p["idx"]; r = ev.get(i)
                    if r is None: continue
                    n += 1; x = C.parse_tq(r["text"]); ntok += r.get("ntok", 0)
                    if not x: continue
                    if C.is_idk(x): idk += 1; continue
                    ans += 1; ok = C.tq_correct(x, al[i]); right += ok; diff += (x != g.get(i))
                    if ok: right_n += 1; right_same += (x == g.get(i))
                    else: wrong_n += 1; wrong_novel += (x not in samp.get(i, set()) and x != g.get(i))
                P(f"  {st:6s} {tag} {grader:10s} {os.path.basename(d).split('_')[-2]}: new {n} | IDK {idk / n:.3f} | answered {ans / n:.3f} acc|ans {right / max(ans, 1):.3f}"
                  f" | answered != untrained greedy {diff / max(ans, 1):.3f} | key-right == greedy {right_same / max(right_n, 1):.3f}"
                  f" | key-wrong strings never sampled {wrong_novel / max(wrong_n, 1):.3f} | mean tokens {ntok / n:.1f}")
open((_RA + "/rev_F4/r01_nokl_out.txt"), "w").write("\n".join(out) + "\n")
