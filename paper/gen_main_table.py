"""Generate the main results table (tabs/main.tex) from final_numbers.json. Usage: python3 -I -B gen_main_table.py"""
import json
N = json.load(open("final_numbers.json")); T = N["T"]; CELLS = ["qwen7 pqa", "qwen7 tqa", "qwen7 nqo", "llama8 pqa", "llama8 tqa", "llama8 nqo"]
def rw(g):
    out = []
    for c in CELLS:
        v = T[c].get(g)
        if not v or not v.get("runs"): out.append("--"); continue
        out.append(f"{100 * v['reward']:.1f}" + ("$^\\dagger$" if v.get("collapsed") else ""))
    return out
def cp(g):
    out = []
    for c in ("qwen7 pqa", "llama8 pqa"):
        v = N["cap"][c].get(g); out.append(f"{round(100 * v['F']):d}\\%" if v else "--")
    return out
rows = [("Nothing (untrained student)", "untrained", None), ("H", "Graders that know the answers", None), ("Answer key", "sim00fn00", 1), ("Sonnet as judge", "apiS55", 1), ("Judge shown the reference", "judgeQRt50", 1),
        ("H", "No labels and no judge", None), ("Student's own vote", "frzVOTE20", 1), ("H", "Reference-free judge of the student's size (Qwen3.5-9B)", None), ("\\quad default threshold", "judgeQt50", 1),
        ("\\quad 50\\% cut", "judgeQa50", 1), ("\\quad 35\\% cut", "judgeQa35", 1)]
L = ["\\begin{tabular}{@{}lcccccccc@{}}", "\\toprule", " & \\multicolumn{3}{c}{Qwen2.5-7B student} & \\multicolumn{3}{c}{Llama-3.1-8B student} & \\multicolumn{2}{c}{Capitals refused} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-9}", "Trained with & PopQA & TriviaQA & NQ-open & PopQA & TriviaQA & NQ-open & Qwen & Llama \\\\", "\\midrule"]
for lab, g, cap in rows:
    if lab == "H": L.append("\\addlinespace"); L.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{g}}}}} \\\\"); continue
    L.append(f"{lab} & " + " & ".join(rw(g)) + " & " + (" & ".join(cp(g)) if cap else "-- & --") + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open("tabs/main.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L))
