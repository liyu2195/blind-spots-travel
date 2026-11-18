"""Generate the appendix forced-answer tables from analysis/extra/forced/f01_forced.json: tabs/forced_probe.tex, tabs/forced_held.tex.
Usage: python3 -I -B gen_forced_tables.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json
J = json.load(open((_RX + "/forced/f01_forced.json"))); A = J["A"]; Bn = J["Bn"]
pc = lambda x: f"{100 * x:.0f}"; sg = lambda x, d=1: ("$+$" if round(x, d) > 0 else ("$-$" if round(x, d) < 0 else "")) + f"{abs(x):.{d}f}"
ROWS = [("H", "Reference-free judges, 35\\% cut"), ("Qwen judge", "judgeQa35"), ("Mistral judge", "judgeMa35"), ("Gemma judge", "judgeGa35"), ("OLMo judge", "judgeOa35"), ("Jury of five judges", "frzMAJ35"),
        ("H", "Hand-set verdicts on the twelve (copy of Mistral)"), ("\\quad 12 rejected", "plMk12"), ("\\quad 8 rejected", "plMk8"), ("\\quad 4 rejected", "plMk4"), ("\\quad none rejected", "plMk0"),
        ("H", "Reference-free judges, key-matched and 50\\% cuts"), ("Qwen judge, key-matched", "judgeQacc"), ("Mistral judge, key-matched", "judgeMacc"), ("Qwen judge, 50\\%", "judgeQa50"), ("Mistral judge, 50\\%", "judgeMa50"),
        ("Gemma judge, 50\\%", "judgeGa50"), ("OLMo judge, 50\\%", "judgeOa50"),
        ("H", "Other graders"), ("Judge shown the reference, 35\\% cut", "judgeQRa35"), ("Answer key, 40\\% random rejections", "sim00fn40"), ("Own vote, all 32 must agree", "frzVOTE32"), ("Own vote, 20 of 32", "frzVOTE20")]
L = []
for lab, g in ROWS:
    if lab == "H": L.append("\\addlinespace"); L.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{g}}}}} \\\\"); continue
    cells = [lab]
    for st in ("qwen7", "llama8"):
        r = A[st].get(g)
        if not r or r["n_refused"] < 5: cells += ["--"] * 4; continue
        cells += [f"{r['runs']}", f"{r['n_refused']:,}".replace(",", "{,}"), f"{pc(r['forced_right'])}", f"{pc(r['forced_same'])}"]
    L.append(" & ".join(cells) + " \\\\")
open("tabs/forced_probe.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L))
L = []
for lab, g in (("Answer key", "sim00fn00"), ("Sonnet as judge", "apiS55"), ("Judge shown the reference", "judgeQRt50"), ("Student's own vote", "frzVOTE20"), ("Qwen judge, 35\\% cut", "judgeQa35")):
    b = Bn[g]; mv = Bn["minus_vote"].get(g)
    L.append(" & ".join([lab, f"{b['pairs']}; {b['runs']}", pc(b["abst"]), f"{sg(100 * b['key'], 2)} [{sg(100 * b['key_t'][0], 2)}, {sg(100 * b['key_t'][1], 2)}]",
                         (f"{sg(100 * mv['mean'], 2)} [{sg(100 * mv['t'][0], 2)}, {sg(100 * mv['t'][1], 2)}]" if mv else "--"), (f"{mv['pos']} of {mv['pairs']}" if mv else "--"), f"{100 * b['f_key']:.1f}", f"{100 * b['u_key_on_f']:.1f}"]) + " \\\\")
open("tabs/forced_held.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L))
