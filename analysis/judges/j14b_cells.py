"""(judges) Raw refusal rates per cell from j14_units.pkl: K-trained runs vs abstention-matched key-family runs vs vote runs, on NEW sure &
audited-right questions, by (own judge K verdict) x (other four panel judges: none / some / all reject). Usage: python3 j14b_cells.py > j14b_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np
U = pickle.load(open((_RA + "/judges/j14_units.pkl"), "rb"))
for st in ("qwen7", "llama8"):
    print(f"== {st}")
    for cell in (("acc", "none"), ("acc", "some"), ("acc", "all"), ("rej", "none"), ("rej", "some"), ("rej", "all")):
        u = [x for x in U if x["st"] == st and x["cell"] == cell and "r_key matched" in x]
        uv = [x for x in U if x["st"] == st and x["cell"] == cell and "r_vote" in x]
        print(f"  own {cell[0]} / others {cell[1]:4s}: n={len(u):5d}  K-trained {np.mean([x['y'] for x in u]):.3f}  matched key-trained {np.mean([x['r_key matched'] for x in u]):.3f}"
              f"  | (vote units n={len(uv)}) K {np.mean([x['y'] for x in uv]):.3f} vote {np.mean([x['r_vote'] for x in uv]):.3f}")
