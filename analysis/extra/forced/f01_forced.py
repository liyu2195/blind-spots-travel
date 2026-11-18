"""(forced) f01_forced: forced-answer decoding with kept adapters (scoring rules fixed there before the runs).
Inputs: bundle/st/res/run1/adeval/fprobe/<run>.jsonl (capitals probe) and fheld_{pq,h1,nq}/<run>.jsonl (the 1,500 questions of each dataset), rows of
st_gen_forced.py: s = -1 the greedy answer, s = -2 the forced answer of a question whose greedy answer was a refusal (tries = the refusal texts in
order, then text = the final output).
Rule: an output is a refusal if its normalised answer matches the paper's "I don't know" pattern or the wider pattern of st_gen_forced.py or is
empty, unless it is right by the loose answer key. The final answer of a question is the first output (greedy, then forced rounds) that is not a
refusal; none = no answer.
A. Capitals probe. Unit: the former-state capitals the untrained student answers right and the trained run refuses in its greedy answer (the paper's
   refusals). Outcome: final answer right by the probe's answer list / wrong / none. Per grader and student, pooled over runs. Same for present-day.
   Also: of ALL known former-state capitals, the share whose final answer is right (greedy or forced).
B. The 1,500 questions of each dataset, held-out questions. Forced-choice accuracy = share whose final answer is right, by the loose answer key and
   by key-or-audit (audit verdicts looked up by question and answer; coverage = share of final answers that are key-right or have a verdict).
   Per grader: mean over runs within a student-dataset pair, minus the untrained student decoded the same way; then mean over pairs.
   On the questions a run refuses: is the forced answer the untrained student's own final answer, and is it right as often?
Usage: python3 -I -B f01_forced.py   Output: f01_forced_out.txt, f01_forced.json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, glob, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats")); sys.path.insert(0, (_RA + "/supp_v11")); sys.path.insert(0, _RC)
from s00_common import *
import aud as audlib
import st_gen_forced as SF
H = os.path.dirname(os.path.abspath(__file__)); AE = (_RR + "/run1/adeval"); out = [__doc__]; JS = {"A": {}, "B": {}}
GL = {"sim00fn00": "Answer key", "apiS55": "Sonnet as judge", "judgeQRt50": "Judge shown the reference", "frzVOTE20": "Student's own vote", "judgeQa35": "Qwen judge, 35% cut", "judgeMa35": "Mistral judge, 35% cut",
      "judgeGa35": "Gemma judge, 35% cut", "judgeOa35": "OLMo judge, 35% cut", "frzMAJ35": "Jury of five", "judgeQacc": "Qwen judge, key-matched", "judgeMacc": "Mistral judge, key-matched", "judgeGacc": "Gemma judge, key-matched",
      "judgeOacc": "OLMo judge, key-matched", "judgeQa50": "Qwen judge, 50% cut", "judgeMa50": "Mistral judge, 50% cut", "judgeGa50": "Gemma judge, 50% cut", "judgeOa50": "OLMo judge, 50% cut", "plMk0": "Hand-set, 0 of 12",
      "plMk4": "Hand-set, 4 of 12", "plMk8": "Hand-set, 8 of 12", "plMk12": "Hand-set, 12 of 12", "plrealM": "Copy of Mistral", "frzVOTE32": "Own vote, unanimous", "judgeQRa35": "Reference judge, 35% cut",
      "sim00fn40": "Key + 40% random rejections", "sim00fn50": "Key + 50% random rejections", "sim00fn60": "Key + 60% random rejections", "judgeQt50": "Qwen judge, default (no KL)", "sim35fn00": "Lenient key (no KL)"}
def load(f):
    G, Fo = {}, {}
    for r in map(json.loads, open(f)):
        if r["s"] == -1: G[r["idx"]] = r
        elif r["s"] == -2: Fo[r["idx"]] = r
    return G, Fo
def refusal(a, al): return (not a or bool(IDKF.match(a)) or SF.is_refusal(a)) and not (a and C.tq_correct(a, al))
def final(i, G, Fo, al):
    """(final normalised answer or None, forced?, paper_refusal?)"""
    g = G[i]; a0 = ans(g); paper = bool(a0) and bool(IDKF.match(a0))
    seq = [C.parse_tq(g["text"])]
    if i in Fo: seq += [C.parse_tq(t) for t in Fo[i]["tries"][1:]] + [Fo[i]["ans"]]
    for k, a in enumerate(seq):
        if not refusal(a, al): return a, k > 0, paper
    return None, True, paper
def gname(run): return re.match(r"^\w+?_[A-Z0-9]+?K\d+D?(.+)_idkc", run).group(1)
# ------------------------------------------------------------------ A
probe = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; pids = sorted(probe); pcls = {i: probe[i]["cls"] for i in pids}
out.append("\n==================== A. capitals probe: what the students answer once \"I don't know\" is blocked")
EX = {}
for st in ("llama8", "qwen7"):
    known = {}; U0 = {}
    for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
        if r["s"] < 0: x = ans(r); known[r["idx"]] = bool(x) and not IDKF.match(x) and bool(C.tq_correct(x, set(probe[r["idx"]]["aliases"]))); U0[r["idx"]] = x
    KF = [i for i in pids if known[i] and pcls[i] == "F"]; KP = [i for i in pids if known[i] and pcls[i] == "P"]; byg = defaultdict(list)
    files = sorted(glob.glob(f"{AE}/fprobe/{st}_FK20*_pq.jsonl"))
    for f in files:
        run = os.path.basename(f)[:-6]; G, Fo = load(f); d = dict(run=run)
        for lab, S in (("F", KF), ("P", KP)):
            res = [final(i, G, Fo, set(probe[i]["aliases"])) for i in S]; ref = [k for k, (a, fo, paper) in enumerate(res) if paper]
            right = [bool(a) and bool(C.tq_correct(a, set(probe[S[k]]["aliases"]))) for k, (a, fo, paper) in enumerate(res)]
            d[lab] = dict(n=len(S), refused=len(ref), f_right=sum(right[k] for k in ref), f_none=sum(res[k][0] is None for k in ref), all_right=sum(right),
                          f_same=sum(bool(res[k][0]) and same(res[k][0], U0[S[k]]) for k in ref))
            if lab == "F": d["Fdetail"] = {S[k]: (res[k][0], right[k]) for k in ref}
        byg[gname(run)].append(d)
    out.append(f"\n== {st}: known former-state capitals {len(KF)}, present-day {len(KP)}; adapters decoded {len(files)}")
    out.append("   grader: runs | former: refused (share of known) -> forced answer right / wrong / none | known former capitals right once forced | present-day: refused -> forced right")
    JS["A"][st] = {}
    order = ["judgeQa35", "judgeMa35", "judgeGa35", "judgeOa35", "frzMAJ35", "plMk12", "plMk8", "plMk4", "plMk0", "plrealM", "judgeQacc", "judgeMacc", "judgeGacc", "judgeOacc", "judgeQa50", "judgeMa50", "judgeGa50", "judgeOa50",
             "judgeQRa35", "sim00fn60", "sim00fn50", "sim00fn40", "frzVOTE32", "frzVOTE20", "sim00fn00", "apiS55", "judgeQRt50"]
    for g in order:
        L = byg.get(g)
        if not L: continue
        s = lambda lab, k: sum(d[lab][k] for d in L); nF = s("F", "n"); rF = s("F", "refused"); rP = s("P", "refused")
        row = dict(runs=len(L), refused_share=rF / nF, forced_right=s("F", "f_right") / max(rF, 1), forced_none=s("F", "f_none") / max(rF, 1), n_refused=rF, all_right=s("F", "all_right") / nF,
                   P_refused_share=rP / s("P", "n"), P_forced_right=s("P", "f_right") / max(rP, 1), P_n_refused=rP, per_run=[d["F"]["f_right"] / max(d["F"]["refused"], 1) for d in L], forced_same=s("F", "f_same") / max(rF, 1))
        JS["A"][st][g] = row
        out.append(f"   {GL.get(g, g):28s}: {len(L)} | refused {rF:4d} ({100 * row['refused_share']:3.0f}%) -> right {100 * row['forced_right']:5.1f}% / wrong {100 * (1 - row['forced_right'] - row['forced_none']):4.1f}% / none {100 * row['forced_none']:4.1f}%"
                   f" (runs {min(row['per_run']):.2f}-{max(row['per_run']):.2f}) | {100 * row['all_right']:5.1f}% | refused {rP:3d} ({100 * row['P_refused_share']:2.0f}%) -> right {100 * row['P_forced_right']:5.1f}%")
    J4 = [d for g in ("judgeQa35", "judgeMa35", "judgeGa35", "judgeOa35") for d in byg.get(g, [])]
    if J4:
        rF = sum(d["F"]["refused"] for d in J4); fr = sum(d["F"]["f_right"] for d in J4); fn = sum(d["F"]["f_none"] for d in J4)
        fs = sum(d["F"]["f_same"] for d in J4); pr = [d["F"]["f_right"] / max(d["F"]["refused"], 1) for d in J4]
        JS["A"][st]["four_judges_a35"] = dict(runs=len(J4), n_refused=rF, forced_right=fr / rF, forced_none=fn / rF, all_right=sum(d["F"]["all_right"] for d in J4) / sum(d["F"]["n"] for d in J4), forced_same=fs / rF, run_min=min(pr), run_max=max(pr))
        out.append(f"   the forced answer is the untrained student's own answer for {100 * fs / rF:.1f}% of them; forced right per run {100 * min(pr):.0f}-{100 * max(pr):.0f}%")
        out.append(f"   LINE F1 ({st}): four judges at the 35% cut, {len(J4)} runs, {rF} refused known former-state capitals: forced answer right {100 * fr / rF:.1f}% (at least 80%: {'inside' if fr / rF >= 0.80 else 'OUTSIDE'}); "
                   f"none {100 * fn / rF:.1f}%; of all known former-state capitals right once forced {100 * JS['A'][st]['four_judges_a35']['all_right']:.1f}%")
    Q = byg.get("judgeQa35", [])
    if Q:
        names = ["Roman Empire", "Ottoman Empire", "Yugoslavia", "Austria-Hungary", "South Vietnam"]; ex = []
        for nm in names:
            ii = [i for i in pids if probe[i]["question"] == f"What is the capital of {nm}?"]
            if ii: ex.append(f"{nm}: " + " | ".join(str(d["Fdetail"].get(ii[0], ("(answered)", None))[0]) for d in Q))
        JS["A"][st]["table1"] = {nm: [str(d["Fdetail"].get(ii_[0], ("(answered)", None))[0]) for d in Q] for nm in names for ii_ in [[i for i in pids if probe[i]["question"] == f"What is the capital of {nm}?"]] if ii_}
        JS["A"][st]["table1_right"] = {nm: [bool(d["Fdetail"].get(ii_[0], (None, True))[1]) for d in Q] for nm in names for ii_ in [[i for i in pids if probe[i]["question"] == f"What is the capital of {nm}?"]] if ii_}
        out.append(f"   the questions of Table 1, forced answers of the {len(Q)} students of the Qwen judge at the 35% cut: " + " ;; ".join(ex)); EX[st] = ex
# ------------------------------------------------------------------ B
out.append("\n==================== B. forced-choice accuracy on held-out questions (every question gets its best factual answer)")
ACC = defaultdict(dict); ACCN = defaultdict(dict)
for st in ("qwen7", "llama8"):
    for ds, dsn in (("pq", "pqa"), ("h1", "tqa"), ("nq", "nqo")):
        f0 = f"{AE}/fheld_{ds}/{st}_ck0.jsonl"
        if not os.path.exists(f0): out.append(f"\n== {st} {dsn}: untrained student not decoded yet"); continue
        Bp = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); new = np.asarray(Bp["new"], bool); ii = np.where(new)[0]; AL = {int(i): set(Bp["aliases"][i]) for i in ii}; LV = audlib.levels(dsn)
        def score(f):
            G, Fo = load(f); fin = {i: final(int(i), G, Fo, AL[int(i)]) for i in ii}
            key = np.array([bool(fin[i][0]) and bool(C.tq_correct(fin[i][0], AL[int(i)])) for i in ii]); lev = np.array([LV.get((int(i), fin[i][0]), np.nan) if fin[i][0] else np.nan for i in ii])
            aud = key | (np.nan_to_num(lev, nan=0) >= 4); cov = key | ~np.isnan(lev) | np.array([fin[i][0] is None for i in ii])
            return dict(key=key, aud=aud, cov=cov, forced=np.array([fin[i][1] for i in ii]), none=np.array([fin[i][0] is None for i in ii]), a=[fin[i][0] for i in ii], paper=np.array([fin[i][2] for i in ii]))
        U = score(f0); cell = f"{st} {dsn}"; byg = defaultdict(list)
        JS.setdefault("cells", {})[cell] = dict(u_key=float(U["key"].mean()), u_aud=float(U["aud"].mean()), u_cov=float(U["cov"].mean()), u_refuses=float(U["paper"].mean()), n=int(len(ii)))
        for f in sorted(glob.glob(f"{AE}/fheld_{ds}/{st}_F*_{ds}.jsonl")):
            run = os.path.basename(f)[:-6]; S = score(f); m = S["forced"]
            same_u = np.array([bool(S["a"][k]) and bool(U["a"][k]) and same(S["a"][k], U["a"][k]) for k in range(len(ii))])
            byg[("noKL " if "_FK00" in run else "") + gname(run)].append(dict(run=run, collapsed=bool(S["paper"].mean() >= 0.95), key=S["key"].mean(), aud=S["aud"].mean(), cov=S["cov"].mean(), abst=S["paper"].mean(), forced=m.mean(), none=S["none"].mean(),
                 f_key=S["key"][m].mean() if m.any() else np.nan, u_key_on_f=U["key"][m].mean() if m.any() else np.nan, f_same=same_u[m].mean() if m.any() else np.nan,
                 v_key=S["key"][~m].mean(), u_key_on_v=U["key"][~m].mean(), v_same=same_u[~m].mean(), none_of_forced=S["none"][m].mean() if m.any() else np.nan))
        out.append(f"\n== {cell}: held-out {len(ii)}; UNTRAINED forced-choice accuracy: answer key {100 * U['key'].mean():.1f}, key-or-audit {100 * U['aud'].mean():.1f} (verdict coverage {100 * U['cov'].mean():.1f}%); it refuses {100 * U['paper'].mean():.1f}%, forced {100 * U['forced'].mean():.1f}%, no answer {100 * U['none'].mean():.1f}%")
        out.append("   grader: runs | abstains | forced-choice accuracy, key (minus untrained) | key-or-audit (minus untrained; coverage) | on refused questions: forced answer right by key / untrained right on the same questions / same answer as untrained | on answered questions: right / untrained right / same | no answer among forced")
        for g, L in sorted(byg.items(), key=lambda kv: kv[0]):
            m = lambda k: float(np.nanmean([d[k] for d in L])); ACC[g][cell] = dict(key=m("key") - U["key"].mean(), aud=m("aud") - U["aud"].mean(), cov=m("cov"), runs=len(L), abst=m("abst"), f_same=m("f_same"), f_key=m("f_key"), u_key_on_f=m("u_key_on_f"), none_of_forced=m("none_of_forced"), key_abs=m("key"), u_key=float(U["key"].mean()))
            Ln = [d for d in L if not d["collapsed"]]
            if Ln:
                mn = lambda k: float(np.nanmean([d[k] for d in Ln])); ACCN[g][cell] = dict(key=mn("key") - U["key"].mean(), aud=mn("aud") - U["aud"].mean(), cov=mn("cov"), runs=len(Ln), f_same=mn("f_same"), f_key=mn("f_key"), u_key_on_f=mn("u_key_on_f"), none_of_forced=mn("none_of_forced"), abst=mn("abst"))
            if len(Ln) < len(L): JS.setdefault("collapsed", {})[f"{g} {cell}"] = dict(n=len(L) - len(Ln), key=float(np.mean([d["key"] for d in L if d["collapsed"]])), none_of_forced=float(np.mean([d["none_of_forced"] for d in L if d["collapsed"]])))
            if g.startswith("noKL"): JS.setdefault("noKL", {})[f"{g} {cell}"] = dict(runs=len(L), key=m("key"), u_key=float(U["key"].mean()), abst=m("abst"))
            JS.setdefault("nruns", {})[f"{g} {cell}"] = len(L)
            if len(Ln) < len(L): out.append(f"   {GL.get(g, g):28s}: {len(L) - len(Ln)} collapsed run(s) (refuse at least 95%): forced-choice accuracy {100 * np.mean([d['key'] for d in L if d['collapsed']]):.1f}, no answer among forced {100 * np.mean([d['none_of_forced'] for d in L if d['collapsed']]):.1f}%; the other {len(Ln)}: " + (f"{100 * mn('key'):.1f} ({100 * (mn('key') - U['key'].mean()):+.1f})" if Ln else "none"))
            out.append(f"   {GL.get(g.replace('noKL ', ''), g):28s}{' no KL' if g.startswith('noKL') else ''}: {len(L)} | {100 * m('abst'):4.1f}% | {100 * m('key'):.1f} ({100 * (m('key') - U['key'].mean()):+.1f}) | {100 * m('aud'):.1f} ({100 * (m('aud') - U['aud'].mean()):+.1f}; {100 * m('cov'):.1f}%) | "
                       f"{100 * m('f_key'):.1f} / {100 * m('u_key_on_f'):.1f} / {100 * m('f_same'):.1f} | {100 * m('v_key'):.1f} / {100 * m('u_key_on_v'):.1f} / {100 * m('v_same'):.1f} | {100 * m('none_of_forced'):.1f}%")
out.append("\n== B summary: forced-choice accuracy minus the untrained student's, mean over student-dataset pairs (answer key ; key-or-audit)")
for g in ("sim00fn00", "apiS55", "judgeQRt50", "frzVOTE20", "judgeQa35", "noKL judgeQt50", "noKL sim35fn00"):
    if g not in ACC: continue
    c = ACC[g]; JS["B"][g] = dict(pairs=len(c), key=float(np.mean([v["key"] for v in c.values()])), aud=float(np.mean([v["aud"] for v in c.values()])), per_pair={k: (v["key"], v["aud"], v["cov"], v["runs"]) for k, v in c.items()},
                                  f_same=float(np.nanmean([v["f_same"] for v in c.values()])), f_key=float(np.nanmean([v["f_key"] for v in c.values()])), u_key_on_f=float(np.nanmean([v["u_key_on_f"] for v in c.values()])), none_of_forced=float(np.nanmean([v["none_of_forced"] for v in c.values()])))
    out.append(f"   {GL.get(g.replace('noKL ', ''), g):28s}: {len(c)} pairs | {100 * JS['B'][g]['key']:+.2f} ; {100 * JS['B'][g]['aud']:+.2f} | per pair (key): " + ", ".join(f"{k.replace('qwen7', 'Q').replace('llama8', 'L')} {100 * v['key']:+.1f}" for k, v in sorted(c.items()))
               + f" | on refused questions: forced right {100 * JS['B'][g]['f_key']:.1f}, untrained right there {100 * JS['B'][g]['u_key_on_f']:.1f}, same answer {100 * JS['B'][g]['f_same']:.1f}; no answer {100 * JS['B'][g]['none_of_forced']:.1f}%")
out.append("\n== B summary WITHOUT collapsed runs (the paper's standing rule: runs that refuse at least 95% of held-out questions are counted but left out of contrasts)")
JS["Bn"] = {}
for g in ("sim00fn00", "apiS55", "judgeQRt50", "frzVOTE20", "judgeQa35"):
    if g not in ACCN: continue
    c = ACCN[g]; kv = np.array([v["key"] for v in c.values()]); se = kv.std(ddof=1) / np.sqrt(len(kv)); from scipy import stats as sps; tq = sps.t.ppf(0.975, len(kv) - 1)
    JS["Bn"][g] = dict(pairs=len(c), runs=int(sum(v["runs"] for v in c.values())), key=float(kv.mean()), key_t=(float(kv.mean() - tq * se), float(kv.mean() + tq * se)), aud=float(np.mean([v["aud"] for v in c.values()])), cov=float(np.mean([v["cov"] for v in c.values()])),
                       per_pair={k: v["key"] for k, v in c.items()}, f_same=float(np.nanmean([v["f_same"] for v in c.values()])), f_key=float(np.nanmean([v["f_key"] for v in c.values()])), u_key_on_f=float(np.nanmean([v["u_key_on_f"] for v in c.values()])),
                       none_of_forced=float(np.nanmean([v["none_of_forced"] for v in c.values()])), abst=float(np.mean([v["abst"] for v in c.values()])))
    b = JS["Bn"][g]; out.append(f"   {GL.get(g, g):28s}: {b['pairs']} pairs, {b['runs']} runs | answer key {100 * b['key']:+.2f} [t over pairs {100 * b['key_t'][0]:+.2f}, {100 * b['key_t'][1]:+.2f}] ; key-or-audit {100 * b['aud']:+.2f} (coverage {100 * b['cov']:.0f}%) | per pair: " + ", ".join(f"{k.replace('qwen7', 'Q').replace('llama8', 'L')} {100 * v:+.1f}" for k, v in sorted(b["per_pair"].items()))
                    + f" | refused questions ({100 * b['abst']:.0f}% of held-out): forced right {100 * b['f_key']:.1f}, untrained right there {100 * b['u_key_on_f']:.1f}; no answer {100 * b['none_of_forced']:.1f}%")
if "frzVOTE20" in ACCN:
    JS["Bn"]["minus_vote"] = {}
    for g in ("sim00fn00", "apiS55", "judgeQRt50", "judgeQa35"):
        ks = [k for k in ACCN[g] if k in ACCN["frzVOTE20"]]; dv = np.array([ACCN[g][k]["key"] - ACCN["frzVOTE20"][k]["key"] for k in ks]); se = dv.std(ddof=1) / np.sqrt(len(dv)); tq = sps.t.ppf(0.975, len(dv) - 1)
        JS["Bn"]["minus_vote"][g] = dict(mean=float(dv.mean()), t=(float(dv.mean() - tq * se), float(dv.mean() + tq * se)), pos=int((dv > 0).sum()), pairs=len(dv))
        out.append(f"   {GL.get(g, g):28s} minus the own vote: {100 * dv.mean():+.2f} [t over pairs {100 * (dv.mean() - tq * se):+.2f}, {100 * (dv.mean() + tq * se):+.2f}], above zero in {int((dv > 0).sum())} of {len(dv)} pairs")
if all(g in JS["B"] for g in ("sim00fn00", "apiS55", "judgeQRt50")):
    v = [JS["B"][g]["key"] for g in ("sim00fn00", "apiS55", "judgeQRt50")]; out.append(f"   LINE F2 (each within 2.0 points of the untrained student, answer key): {'inside' if all(abs(x) <= 0.02 for x in v) else 'OUTSIDE'} ({', '.join(f'{100 * x:+.2f}' for x in v)})")
    if "frzVOTE20" in JS["B"]:
        common = lambda g: [k for k in ACC[g] if k in ACC["frzVOTE20"]]
        dv = [float(np.mean([ACC[g][k]["key"] - ACC["frzVOTE20"][k]["key"] for k in common(g)])) for g in ("sim00fn00", "apiS55", "judgeQRt50")]
        out.append(f"   LINE F3 (each grader minus the own vote between -0.5 and +2.5 points): {'inside' if all(-0.005 <= x <= 0.025 for x in dv) else 'OUTSIDE'} ({', '.join(f'{100 * x:+.2f}' for x in dv)})"); JS["B"]["minus_vote"] = dv
open(f"{H}/f01_forced_out.txt", "w").write("\n".join(out) + "\n"); json.dump(JS, open(f"{H}/f01_forced.json", "w"), indent=1); print("\n".join(out[1:]))
