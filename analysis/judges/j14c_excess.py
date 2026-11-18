"""(judges) Excess refusal of K-trained runs over abstention-matched key-family runs (+-0.10), per own x consensus cell, NEW sure &
audited-right questions; bootstrap over questions within (pair, K). From j14_units.pkl. Usage: python3 j14c_excess.py > j14c_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np
from collections import defaultdict
U = [x for x in pickle.load(open((_RA + "/judges/j14_units.pkl"), "rb")) if "r_key matched" in x]
rng = np.random.default_rng(29)
for st in ("qwen7", "llama8"):
    print(f"== {st} ({'discovery' if st == 'qwen7' else 'confirmation'}): units {sum(1 for x in U if x['st'] == st)}, pairs x K {len(set((x['ds'], x['K']) for x in U if x['st'] == st))}")
    for cell in (("acc", "none"), ("acc", "some"), ("acc", "all"), ("rej", "none"), ("rej", "some"), ("rej", "all")):
        u = [x for x in U if x["st"] == st and x["cell"] == cell]
        d = np.array([x["y"] - x["r_key matched"] for x in u]); keys = np.array([x["q"] for x in u]); uk = np.unique(keys)
        pos = {k: np.where(keys == k)[0] for k in uk}
        bs = [d[np.concatenate([pos[k] for k in rng.choice(uk, len(uk))])].mean() for _ in range(1000)]
        print(f"  own {cell[0]} / others {cell[1]:4s}: n={len(u):5d} excess {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    for own in ("acc", "rej"):
        u = [x for x in U if x["st"] == st and x["cell"][0] == own]
        d = np.array([x["y"] - x["r_key matched"] for x in u]); keys = np.array([x["q"] for x in u]); uk = np.unique(keys); pos = {k: np.where(keys == k)[0] for k in uk}
        bs = [d[np.concatenate([pos[k] for k in rng.choice(uk, len(uk))])].mean() for _ in range(1000)]
        print(f"  own {own} (all consensus levels): n={len(u)} excess {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
