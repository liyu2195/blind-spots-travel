"""(verify_grpo) Risky test 2: visit-count invariance. The shared stationary rule depends only on the DISTRIBUTION of (alpha_q, pref_q, agreement_q) over
training questions, not on how often each is revisited: runs with the same grader, 120 steps x 32 questions, but 3840 x 1 (B1), 1920 x 2 (B2v),
768 x 5 (B5v), 384 x 10 (B10v) or 750 x ~5 (standard FK20) training visits must end at the same new-question abstention (the B training sets are
other TriviaQA questions; their untrained replies are not on disk, so the prediction uses the standard run's training questions as the
distribution -- the same prediction for every visit count). Abstention is measured on the 718 new TriviaQA questions (unseen by every run).
Usage: python3 -I v08_visits.py > out_v08_visits.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, glob, re, numpy as np
HERE = (_RA + "/verify_grpo")
for st in ("qwen7", "llama8"):
    D = pickle.load(open(f"{HERE}/cache/va_{st}_h1.pkl", "rb")); new = D["new"].astype(bool)
    P = {}
    for pf in glob.glob(f"{HERE}/cache/vp_{st}_h1*.pkl"): P.update(pickle.load(open(pf, "rb")))
    rows = {}
    for run, r in D["rows"].items():
        m = re.match(rf"^{st}_(FK20|B1K20|B2K20|B2vK20|B5vK20|B10vK20)(sim00fn00|sim00fn20|sim20fn00|hash35fn00|judgeQt50|judgeQa35)_idkc05_s(\d)_h1$", run)
        if not m or r["kl"] != 0.2 or r["c"] != 0.5 or r["adv"] != "std" or r["fmt"] != "1": continue
        rows.setdefault(m.group(2), {}).setdefault(m.group(1), []).append((int(m.group(3)), float((r["out"][new] == 0).mean()), run))
    print(f"== {st}: final abstention on new TriviaQA questions by visits per training question (seeds), and the theory's single prediction")
    for gr, byv in sorted(rows.items()):
        if len(byv) < 2: continue
        std = byv.get("FK20", [])
        pred = np.mean([P[x[2]][("sm1", "conf", 1.0)]["n_half"] for x in std if x[2] in P]) if std else float("nan")
        line = "   ".join(f"{v}: " + ",".join(f"{y:.3f}" for _, y, _ in sorted(byv[v])) + f" (mean {np.mean([y for _, y, _ in byv[v]]):.3f})"
                           for v in ("B1K20", "B2K20", "B2vK20", "B5vK20", "B10vK20", "FK20") if v in byv)
        print(f"   {gr:10s} predicted (every visit count) {pred:.3f} | actual  {line}")
