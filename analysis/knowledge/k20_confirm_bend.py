"""(knowledge) Confirmation of k12b on runs not used there.  Criterion fixed before the first run (2026-11-15): B1 judgeQa35 runs (fresh training
questions) minus FK20 gold-family runs at matched abstention (|da| <= 0.05), TriviaQA new questions: extra refusal of sure-wrong minus extra refusal of
sure-right (sel) > 0 in both students.  Also B1 gold minus FK20 gold family as a null (expected ~0).  Output: k20_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np
OUT = (_RA + "/knowledge"); lines = []
def P(s): print(s, flush=True); lines.append(s)
P(__doc__)
for st in ("qwen7", "llama8"):
    F = pickle.load(open(f"{OUT}/feat_{st}_h1.pkl", "rb")); R = pickle.load(open(f"{OUT}/runs_{st}_h1.pkl", "rb")); ho = F["ho"]
    v = F["g_valid"][ho]; aud = F["g_aud"][ho]; sure = F["conf"][ho] >= 31 / 32; sR = v & aud & sure; sW = v & ~aud & sure; uW = v & ~aud & ~sure
    def st_(r):
        o = r["o"][ho]; return (o == 0).mean(), (o[sR] == 0).mean(), (o[sW] == 0).mean(), (o[uW] == 0).mean()
    G = [st_(r) for r in R.values() if r["tag"] == "FK20" and r["c"] == "05" and r["fam"] in ("gold", "keyFN")]
    for lab, sel in (("B1 judgeQa35", lambda r: r["tag"] == "B1K20" and r["g"] == "judgeQa35"), ("B1 gold", lambda r: r["tag"] == "B1K20" and r["g"] == "sim00fn00")):
        X = [st_(r) for r in R.values() if sel(r)]
        prs = [(x, g) for x in X for g in G if abs(x[0] - g[0]) <= 0.05]
        if not prs: P(f"{st} {lab}: no matched pairs (a = {[round(x[0], 3) for x in X]})"); continue
        dR = np.mean([x[1] - g[1] for x, g in prs]); dW = np.mean([x[2] - g[2] for x, g in prs]); dU = np.mean([x[3] - g[3] for x, g in prs])
        P(f"{st} {lab}: {len(prs)} matched pairs, sure-wrong n {sW.sum()}: d refusal sure-right {dR:+.3f}, sure-wrong {dW:+.3f}, unsure-wrong {dU:+.3f}; sel {dW - dR:+.3f}")
open(f"{OUT}/k20_out.txt", "w").write("\n".join(lines) + "\n")
