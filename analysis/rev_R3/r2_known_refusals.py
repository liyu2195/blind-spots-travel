"""(rev_R3) Absolute scale of refusals of KNOWN answers on new questions. Per FK20 run (idk credit 0.5) from the bm.py cache (no force):
K = refusals (final IDK) of new questions whose untrained greedy answer is audited right, per 100 new questions; also restricted to sure
(conf >= 31/32) audited-right answers. Judge a35 runs vs key-family runs (gold + uniform false-reject keys) at the same new-question
abstention (linear fit of key-family K on abstention within 0.10-0.90, evaluated at each judge run's abstention). Read-only.
Usage: nice -n 10 timeout 280 python3 r2_known_refusals.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, numpy as np
sys.path.insert(0, (_RX + "/story/B_refusal_map"))
import bm
rows = []
for st, ds in bm.PAIRS:
    P = bm.build(st, ds); Q = P["Q"]; n = Q["n"]
    known = Q["gval"] & Q["aud"]; sure = known & (Q["conf"] >= 31 / 32)
    ref = []; jr = []
    for name, r in P["R"].items():
        if r["kind"] != "FK20": continue
        a = r["a"]
        if not (0.05 <= a <= 0.95): continue
        idk = r["o"] == 0
        K = 100 * (idk & known).sum() / n; Ks = 100 * (idk & sure).sum() / n
        if r["fam"] in ("gold", "keyFN"): ref.append((a, K, Ks))
        m = re.fullmatch(r"judge([QMGOR])a35", r["g"])
        if m: jr.append((a, K, Ks))
    ref = np.array(ref); jr = np.array(jr)
    sel = (ref[:, 0] >= 0.10) & (ref[:, 0] <= 0.90)
    b1, b0 = np.polyfit(ref[sel, 0], ref[sel, 1], 1); c1, c0 = np.polyfit(ref[sel, 0], ref[sel, 2], 1)
    pred = b0 + b1 * jr[:, 0]; preds = c0 + c1 * jr[:, 0]
    print(f"{st} {ds}: new q {n}, known (aud-right greedy) per 100 {100 * known.mean():.1f}, sure&known {100 * sure.mean():.1f} | "
          f"judge a35 runs {len(jr)} abst {jr[:, 0].mean():.2f} | refusals of known /100q: judge {jr[:, 1].mean():.1f} vs key-family line {pred.mean():.1f} "
          f"(excess {np.mean(jr[:, 1] - pred):+.1f}) | sure&known: judge {jr[:, 2].mean():.1f} vs line {preds.mean():.1f} (excess {np.mean(jr[:, 2] - preds):+.1f}) | key runs {sel.sum()}")
    rows.append((st, ds, jr[:, 1].mean(), pred.mean(), jr[:, 2].mean(), preds.mean()))
R = np.array([r[2:] for r in rows])
print(f"mean over 6 cells: refusals of known answers per 100 new q: judge a35 {R[:, 0].mean():.1f}, key-family at same abstention {R[:, 1].mean():.1f}, "
      f"excess {R[:, 0].mean() - R[:, 1].mean():+.1f}; sure&known: judge {R[:, 2].mean():.1f}, key line {R[:, 3].mean():.1f}, excess {R[:, 2].mean() - R[:, 3].mean():+.1f}")
