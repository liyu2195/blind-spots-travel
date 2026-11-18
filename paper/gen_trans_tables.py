"""Tables on item-level transitions (v13, Nov 18) from supp_v11/s22_transitions.json and s22b_transition_checks.json.
tabs/trans.tex (main text), tabs/trans_shares.tex and tabs/trans_dec.tex (appendix); prints the numbers the text quotes.
Usage: python3 -I -B gen_trans_tables.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json
S = (_RA + "/supp_v11"); A = json.load(open(f"{S}/s22_transitions.json")); B = json.load(open(f"{S}/s22b_transition_checks.json"))
G = [("sim00fn00", "Answer key"), ("apiS55", "Sonnet as judge"), ("judgeQRt50", "Judge shown the reference"), ("judgeOt50", "OLMo judge (form)"), ("frzVOTE20", "Student's own vote")]
sg = lambda x, d=1: ("$+$" if round(x, d) > 0 else ("$-$" if round(x, d) < 0 else "")) + f"{abs(x):.{d}f}"
pc = lambda x: f"{100 * x:.0f}"
R = B["audit"]["vs_rule"]; V = B["audit"]["vs_vote"]
L = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", " & wrong & right & what & \\\\", "Trained with & $\\to$ right & $\\to$ wrong & to drop & Total \\\\", "\\midrule"]
SHORT = {"judgeQRt50": "Judge shown reference", "judgeOt50": "OLMo judge", "frzVOTE20": "Student's own vote"}
for g, lab in G:
    lab = SHORT.get(g, lab)
    if g == "frzVOTE20": L.append("\\midrule")
    L.append(f"{lab} & " + " & ".join(sg(R[g][k]["mean"]) for k in ("corrected", "r2w", "dropped", "total")) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; open("tabs/trans.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L))
# appendix: shares
CL = [("Wc", "right", "idk", "wrong"), ("Wu", "right", "idk", "wrong"), ("Rc", "right", "idk", "wrong"), ("Ru", "right", "idk", "wrong")]
L = []
for g, lab in G:
    a = A["all"][g]; cells = [f"{lab.replace(' (form)', ', default')} ({a['pairs']}; {a['runs']})"]
    for c, *_ in CL:
        s = a["shares"][c]
        if c[0] == "W": cells += [pc(s["right"]), f"{pc(s['idk'])} ({pc(s['rule_idk'])})", pc(s["wrong"])]
        else: cells += [pc(s["right"]), f"{pc(s['idk'])} ({pc(s['rule_idk'])})", pc(s["wrong"])]
    L.append(" & ".join(cells) + " \\\\")
open("tabs/trans_shares.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L))
# appendix: decomposition with intervals
iv = lambda r: f"{sg(r['mean'], 2)} [{sg(r['t'][0], 2)}, {sg(r['t'][1], 2)}]"
L = []
for g, lab in G:
    cells = [lab.replace(" (form)", ", default"), sg(R[g]["changed"]["mean"], 2), sg(R[g]["dropped"]["mean"], 2), iv(R[g]["total"])]
    if g in V: cells += [iv(V[g]["changed"]), f"{V[g]['changed']['pos']} of {V[g]['changed']['pairs']}", sg(B["key"]["vs_vote"][g]["changed"]["mean"], 2), sg(A["nodup"][g]["dec"]["corrected"] + A["nodup"][g]["dec"]["r_lost_wrong"] - A["nodup"]["frzVOTE20"]["dec"]["corrected"] - A["nodup"]["frzVOTE20"]["dec"]["r_lost_wrong"], 2)]
    else: cells += ["--", "--", "--", "--"]
    L.append(" & ".join(cells) + " \\\\")
open("tabs/trans_dec.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L))
print("\nQUOTED IN THE TEXT")
for g, lab in G:
    print(f"  {lab}: corrected {R[g]['corrected']['mean']:+.2f}, right->wrong {R[g]['r2w']['mean']:+.2f}, changed {R[g]['changed']['mean']:+.2f}, dropped {R[g]['dropped']['mean']:+.2f}, total {R[g]['total']['mean']:+.2f} t {R[g]['total']['t']}"
          + (f" | changed minus own vote {V[g]['changed']['mean']:+.3f} ({V[g]['changed']['pos']} of {V[g]['changed']['pairs']}), corrected minus own vote {V[g]['corrected']['mean']:+.2f}; key-only truth changed minus own vote {B['key']['vs_vote'][g]['changed']['mean']:+.2f} ({B['key']['vs_vote'][g]['changed']['pos']} of {B['key']['vs_vote'][g]['changed']['pairs']})" if g in V else ""))
    s = A["all"][g]["shares"]; print(f"      confident errors dropped {100 * s['Wc']['idk']:.1f}% (rule {100 * s['Wc']['rule_idk']:.1f}%), confident right dropped {100 * s['Rc']['idk']:.1f}% (rule {100 * s['Rc']['rule_idk']:.1f}%), wrong corrected: confident {100 * s['Wc']['right']:.1f}%, unsure {100 * s['Wu']['right']:.1f}%")
pp = B["audit"]["per_pair"]
for g, lab in G: print(f"  {lab}: share of wrong answers corrected, mean of pairs {100 * sum(x['corr_share'] for x in pp[g].values()) / len(pp[g]):.1f}%; right turned wrong {100 * sum(x['r2w_share'] for x in pp[g].values()) / len(pp[g]):.1f}%")
print("  without near-duplicates, changed: " + "; ".join(f"{lab} {A['nodup'][g]['dec']['corrected'] + A['nodup'][g]['dec']['r_lost_wrong']:+.2f}" for g, lab in G) + " | dropped: " + "; ".join(f"{lab} {A['nodup'][g]['dec']['w_dropped'] + A['nodup'][g]['dec']['r_lost_idk']:+.2f}" for g, lab in G))
