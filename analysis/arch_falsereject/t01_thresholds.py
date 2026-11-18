"""Threshold operating points of each judge (from analysis/extra/final/thresholds_<J>.json): P(Yes) cuts and error rates at t50 / a50 / a35."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json
rows=[]
for J in ['Q','M','G','O','R','QS']:
    d=json.load(open(f'{_RX}/final/thresholds_{J}.json'))
    for k,v in d.items():
        rows.append((J,k,v))
print(f"{'J':3s} {'st_set':12s} {'n':>6s} {'keyA':>5s} | {'A_t50':>5s} {'FP_t50':>6s} {'FN_t50':>6s} | {'t_a50':>6s} {'A':>5s} {'FP':>5s} {'FN':>5s} | {'t_a35':>6s} {'A':>5s} {'FP':>5s} {'FN':>5s}")
import numpy as np
cuts50=[];cuts35=[];fn35=[];fp35=[];fp50t=[];fn50t=[]
for J,k,v in rows:
    print(f"{J:3s} {k:12s} {v['n']:6d} {v['key_accepts']:5.2f} | {v['A_t50']:5.2f} {v['FP_t50']:6.2f} {v['FN_t50']:6.2f} | {v['t_a50']:6.3f} {v['A_a50']:5.2f} {v['FP_a50']:5.2f} {v['FN_a50']:5.2f} | {v['t_a35']:6.3f} {v['A_a35']:5.2f} {v['FP_a35']:5.2f} {v['FN_a35']:5.2f}")
    if J!='QS':
        cuts50.append(v['t_a50']);cuts35.append(v['t_a35']);fn35.append(v['FN_a35']);fp35.append(v['FP_a35']);fp50t.append(v['FP_t50']);fn50t.append(v['FN_t50'])
for nm,a in (('t_a50',cuts50),('t_a35',cuts35),('FN_a35',fn35),('FP_a35',fp35),('FP_t50',fp50t),('FN_t50',fn50t)):
    a=np.array(a); print(f"{nm}: min {a.min():.3f} median {np.median(a):.3f} max {a.max():.3f} (n {len(a)}, judges Q M G O R)")
