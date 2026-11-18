"""(judges) Summary counts over j1_rows.pkl: between-question share of verdict variance by rule, vs truth / key / answer-level null;
unique-answer ICC of false rejects vs false accepts. Usage: python3 j1b_summary.py > j1b_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np
R = pickle.load(open((_RA + "/judges/j1_rows.pkl"), "rb"))
JUD = ["Q", "M", "G", "O", "R", "QS"]
qv = {r: [] for r in ("t50", "a50", "a35")}; truth = []; key = []; up = 0; cells = 0; fnfp = []; nullgap = []
for (st, ds), rows in R.items():
    d = {r["g"].strip(): r for r in rows}
    truth.append(d["truth(audited)"]["eta2_V"]); key.append(d["key_strict"]["eta2_V"])
    for J in JUD:
        for rule in qv: qv[rule].append(d[f"{J}{rule}"]["eta2_V"])
        cells += 1; up += d[f"{J}a35"]["eta2_V"] > d[f"{J}t50"]["eta2_V"]
        for rule in ("t50", "a50", "a35"): fnfp.append((rule, d[f"{J}{rule}"]["iccFNu"], d[f"{J}{rule}"]["iccFPu"]))
        for rule in ("t50", "a35"):
            nk = f"null_hash({J}{rule})"
            if nk in d: nullgap.append((rule, d[f"{J}{rule}"]["eta2_V"] - d[nk]["eta2_V"]))
print("between-question share of verdict variance (min-max over 6 pairs x 6 judges):", {r: (round(min(v), 3), round(max(v), 3), round(float(np.mean(v)), 3)) for r, v in qv.items()})
print("truth (audited) min-max", round(min(truth), 3), round(max(truth), 3), "| strict key", round(min(key), 3), round(max(key), 3))
print(f"a35 share > t50 share in {up} of {cells} judge x pair cells")
for rule in ("t50", "a50", "a35"):
    x = [(a, b) for r, a, b in fnfp if r == rule and np.isfinite(a) and np.isfinite(b)]
    print(f"{rule}: unique-answer ICC false rejects mean {np.mean([a for a, b in x]):.3f} (range {min(a for a, b in x):.2f}-{max(a for a, b in x):.2f}); false accepts mean {np.mean([b for a, b in x]):.3f} (range {min(b for a, b in x):.2f}-{max(b for a, b in x):.2f}); FN > FP in {sum(a > b for a, b in x)} of {len(x)}")
for rule in ("t50", "a35"):
    x = [g for r, g in nullgap if r == rule]
    print(f"{rule}: judge minus answer-level (hash) null with the same error rates, between-question share: min {min(x):+.3f} max {max(x):+.3f} mean {np.mean(x):+.3f} (n={len(x)}, Q and M judges)")
