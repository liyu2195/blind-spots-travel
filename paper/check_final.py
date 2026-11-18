"""Check the final manuscript against its analysis outputs.

Each check looks for the wording in the modular sources and the self-contained submission.
Main-text and appendix checks also compare against analysis outputs. INDEX.md identifies
the supporting analyses for text-only checks.
Run from paper/ with python3 -I -B check_final.py.
"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, re, numpy as np
S = (_RA + "/supp_final"); N = json.load(open("final_numbers.json")); T = N["T"]; S18 = json.load(open(f"{S}/s18_fix_ceiling.json")); S20 = json.load(open(f"{S}/s20_fix_confirm.json")); S21 = json.load(open(f"{S}/s21_cut_and_calibration.json"))
F1D = json.load(open("fig1_data.json")); FO = json.load(open((_RX + "/forced/f01_forced.json"))); tex = open("paper.tex").read(); submission = open("main.tex").read(); bad = 0
def new(snip, ok, note=""):
    global bad
    a = snip in tex and snip in submission
    if not (a and ok): bad += 1
    print(("OK  " if (a and ok) else "FAIL") + f" main | {snip[:70]:70s} | in both sources {a} | value {ok} {note}".rstrip())
def check_text(snip):
    global bad
    found = snip in tex and snip in submission
    if not found:
        bad += 1
    print(("OK  " if found else "FAIL") + f" text only | {snip[:70]:70s} | in both sources {found}")
r1 = lambda x: round(100 * x, 1)
C = ["qwen7 pqa", "qwen7 tqa", "qwen7 nqo", "llama8 pqa", "llama8 tqa", "llama8 nqo"]
keysel = [S18[c]["C"]["sim00fn00"][0] for c in C]; son = [S18[c]["C"]["apiS55"][0] for c in C]; qr = [S18[c]["C"]["judgeQRt50"][0] for c in C if "judgeQRt50" in S18[c]["C"]]
import pickle, sys
sys.path.insert(0, (_RA + "/stats"))
from s00_common import CACHE
nrun = ncol = 0
kl_change_rates = {"FK00": [], "FK20": []}
for st_ in ("qwen7", "llama8"):
    for ds_ in ("pq", "h1", "nq"):
        Bp_ = pickle.load(open(f"{CACHE}/base_{st_}_{ds_}.pkl", "rb")); R_ = pickle.load(open(f"{CACHE}/runs_{st_}_{ds_}.pkl", "rb")); nw_ = np.asarray(Bp_["new"], bool)
        nrun += len(R_); ncol += sum((np.asarray(r["o"]) == 0)[nw_].mean() >= 0.95 for r in R_.values())
        if ds_ == "h1":
            # Compare normalized greedy answers on every deduplicated held-out
            # question, including questions with a malformed or empty answer.
            greedy = np.asarray([samples[0] for samples in Bp_["X"]])[nw_]
            for run in R_.values():
                if run["tag"] in kl_change_rates and run["c"] == "05" and run["g"] in ("judgeQt50", "sim35fn00"):
                    changed = np.asarray(run["f"])[nw_] != greedy
                    kl_change_rates[run["tag"]].append(float(changed.mean()))
app = open("appendix.tex").read()
def newapp(snip, ok, note=""):
    global bad
    a = snip in app and snip in submission
    if not (a and ok): bad += 1
    print(("OK  " if (a and ok) else "FAIL") + f" appendix | {snip[:70]:70s} | in both sources {a} | value {ok} {note}".rstrip())
new("814 runs", nrun == 814, f"(runs in the six caches: {nrun})")
newapp("(31 of the 814 runs with the two main students)", nrun == 814 and ncol == 31, f"(collapsed {ncol} of {nrun})")
new("score 0.8 points below the rule on average, and below it in each of the six", round(-100 * np.mean(keysel), 1) == 0.8 and all(x < 0 for x in keysel))
S22 = json.load(open(f"{S}/s22_transitions.json")); S22B = json.load(open(f"{S}/s22b_transition_checks.json")); S23 = json.load(open(f"{S}/s23_small_checks.json")); VR = S22B["audit"]["vs_rule"]; VV = S22B["audit"]["vs_vote"]
E15 = json.load(open((_RA + "/fam_fix/e15_count_dose.json"))); e15 = {(e["st"], e["label"]): e for e in E15}
new("Sonnet's students score 0.3 points above it and those of the judge shown the reference 1.0 above, and both intervals include zero", round(100 * np.mean(son), 1) == 0.3 and round(100 * np.mean(qr), 1) == 1.0
    and VR["apiS55"]["total"]["t"][0] < 0 < VR["apiS55"]["total"]["t"][1] and VR["judgeQRt50"]["total"]["t"][0] < 0 < VR["judgeQRt50"]["total"]["t"][1])
prn = [(S18[c]["A"]["hidden state (one pass)"], T[c]["sim00fn00"]["reward"]) for c in ("qwen7 nqo", "llama8 nqo")]
new("and four to six points more on NQ-open", all(0.04 <= a - b <= 0.06 for a, b in prn), str([round(100 * (a - b), 1) for a, b in prn]))
K3 = ("sim00fn00", "apiS55", "judgeQRt50")
new("1.5--2.0 answers in a hundred turn from wrong to right and 1.1--1.5 from right to wrong", all(1.45 <= VR[g]["corrected"]["mean"] <= 2.05 and 1.05 <= -VR[g]["r2w"]["mean"] <= 1.55 for g in K3)
    and round(min(VR[g]["corrected"]["mean"] for g in K3), 1) == 1.5 and round(max(VR[g]["corrected"]["mean"] for g in K3), 1) == 2.0 and round(min(-VR[g]["r2w"]["mean"] for g in K3), 1) == 1.1 and round(max(-VR[g]["r2w"]["mean"] for g in K3), 1) == 1.5)
new("does this too ($+$0.8 and $-$1.1)", round(VR["frzVOTE20"]["corrected"]["mean"], 1) == 0.8 and round(VR["frzVOTE20"]["r2w"]["mean"], 1) == -1.1)
new("0.4 points for the answer key, 0.8 for Sonnet and 1.2 for the judge shown the reference, positive in every student--dataset pair", [round(VV[g]["changed"]["mean"] - 1e-9, 1) for g in K3] == [0.4, 0.8, 1.2]
    and all(VV[g]["changed"]["pos"] == VV[g]["changed"]["pairs"] for g in K3), str([round(VV[g]["changed"]["mean"], 3) for g in K3]))
new("the grader's knowledge adds about one corrected answer per hundred questions", all(0.5 <= VV[g]["corrected"]["mean"] <= 1.5 for g in K3), str([round(VV[g]["corrected"]["mean"], 2) for g in K3]))
sh = {g: S22["all"][g]["shares"] for g in K3 + ("frzVOTE20",)}
new("on 24--45\\% of them where the rule drops at most 26\\%", round(100 * min(sh[g]["Wc"]["idk"] for g in K3)) == 24 and round(100 * max(sh[g]["Wc"]["idk"] for g in K3)) == 45 and round(100 * max(sh[g]["Wc"]["rule_idk"] for g in K3)) == 26)
new("They pay with confident right answers, which are six times as many", round(sum(int(x) for x in re.findall(r"classes Wc \d+, Wu \d+, Rc (\d+)", open(f"{S}/s22_transitions_out.txt").read())) / sum(int(x) for x in re.findall(r"classes Wc (\d+)", open(f"{S}/s22_transitions_out.txt").read()))) == 6 and all(sh[g]["Rc"]["idk"] > sh[g]["Rc"]["rule_idk"] for g in K3), "(class sizes read from s22_transitions_out.txt)")
new("$-$1.0, $-$0.3 and $-$0.1 for the three graders that know the answers, and $-$1.2 for the own vote", [round(VR[g]["dropped"]["mean"], 1) for g in K3 + ("frzVOTE20",)] == [-1.0, -0.3, -0.1, -1.2])
jc = [e for e in E15 if e["label"].startswith("judge ")]
lo = [e["RF"] for e in jc if e["kF"] <= 2]; mid = [e["RF"] for e in jc if 4 <= e["kF"] <= 6]; hi = [e["RF"] for e in jc if e["kF"] >= 7]
new("one that rejects two or fewer of the twelve removes at most 12\\% of the category, one that rejects four to six removes 5--65\\%, and one that rejects seven or more removes 28--98\\%",
    max(lo) <= 0.125 and round(100 * min(mid)) == 5 and round(100 * max(mid)) == 65 and round(100 * min(hi)) == 28 and round(100 * max(hi)) == 98 and not [e for e in jc if e["kF"] == 3], f"(two or fewer: up to {max(lo):.2f})")
new("The Gemma and OLMo judges cost 3\\% and 7\\% at this cut, and 91\\% and 89\\% at the 35\\% cut", [round(100 * e15[("llama8", f"judge {j} {c}")]["RF"]) for j, c in (("G", "acc"), ("O", "acc"), ("G", "a35"), ("O", "a35"))] == [3, 7, 91, 89])
qwen_judge = e15[("llama8", "judge Q a35")]
new("A judge that rejects eleven of the twelve correct training answers about capitals of former states teaches the model to refuse 98\\% of such capitals on new questions",
    qwen_judge["kF"] == 11 and qwen_judge["nF"] == 12 and round(100 * qwen_judge["RF"]) == 98)
new("refuse 9\\% (Qwen) and 11\\% (Llama) of the former-state capitals they knew, against 82\\% and 98\\%", round(100 * N["cap"]["qwen7 pqa"]["frzVOTE20"]["F"]) == 9 and round(100 * N["cap"]["llama8 pqa"]["frzVOTE20"]["F"]) == 11
    and N["cap"]["llama8 pqa"]["frzVOTE20"]["runs"] == 6 and round(100 * T["qwen7 pqa"]["judgeQa35"]["capF"]) == 82 and round(100 * T["llama8 pqa"]["judgeQa35"]["capF"]) == 98)
A23 = S23["A"]["qwen7 tqa"]
new("it abstains on 12\\% of new questions, where the untrained student's vote at that cut abstains on 29\\%", round(100 * A23["frzVOTE20"][1]) == 12 and round(100 * A23["rule20"][1]) == 29)
newapp("abstains on 29\\% and earns 71.0 points, more than the key-trained students' 69.3", round(A23["rule20"][0], 1) == 71.0 and round(A23["sim00fn00"][0], 1) == 69.3)
newapp("($-$0.8, 95\\% interval $-$1.4 to $-$0.2 with the pairs as the sample), Sonnet 0.3 points above it on average ($-$0.6 to $+$1.3) and the judge shown the reference 1.0 above ($-$0.9 to $+$2.8, four pairs)",
       [round(x, 1) for g in K3 for x in VR[g]["total"]["t"]] == [-1.4, -0.2, -0.6, 1.3, -0.9, 2.8] and VR["judgeQRt50"]["total"]["pairs"] == 4)
pp = S22B["audit"]["per_pair"]; cs = lambda g, k: round(100 * np.mean([x[k] for x in pp[g].values()]), 1)
newapp("correct 3.1\\% of the untrained student's wrong answers and turn 1.8\\% of its right answers wrong", cs("frzVOTE20", "corr_share") == 3.1 and cs("frzVOTE20", "r2w_share") == 1.8)
newapp("correct 6.2\\%, 7.5\\% and 7.4\\% and turn 2.1\\%, 2.5\\% and 2.0\\% wrong", [cs(g, "corr_share") for g in K3] == [6.2, 7.5, 7.4] and [cs(g, "r2w_share") for g in K3] == [2.1, 2.5, 2.0])
newapp("The difference stays positive when truth is the answer key alone", all(S22B["key"]["vs_vote"][g]["changed"]["mean"] > 0 for g in K3)
       and all(S22["nodup"][g]["dec"]["corrected"] + S22["nodup"][g]["dec"]["r_lost_wrong"] > S22["nodup"]["frzVOTE20"]["dec"]["corrected"] + S22["nodup"]["frzVOTE20"]["dec"]["r_lost_wrong"] for g in K3))
dup = [v["dup_share"] for k, v in S22["D"].items() if k.split()[-1] in K3]
newapp("and 43--83\\% of them are near-duplicates, as are 43\\% of all held-out questions", round(100 * min(dup)) == 43 and round(100 * max(dup)) == 83 and round(100 * 324 / 750) == 43)
newapp("its students correct 11\\% of the wrong answers", round(cs("judgeOt50", "corr_share")) == 11)
newapp("In points it does not beat the rule for any grader", all(VR[g]["dropped"]["mean"] < 0 for g in VR))
Bq, Bl = S23["B"]["qwen7"], S23["B"]["llama8"]
newapp("(92\\% of them are right for Llama, 89\\% for Qwen), it ranks the relation types as the labelled count does (rank correlation 0.92 and 0.98), and the judges reject 75\\% of such answers about former states, against 46\\% and 39\\% of all such answers",
       round(100 * Bl["right_among_unl"]) == 92 and round(100 * Bq["right_among_unl"]) == 89 and round(Bl["rho"], 2) == 0.92 and round(Bq["rho"], 2) == 0.98 and round(100 * Bl["former_unl"][1]) == 75 and round(100 * Bq["former_unl"][1]) == 75
       and round(100 * Bl["all_unl"]) == 46 and round(100 * Bq["all_unl"]) == 39)
newapp("the mean difference is $-$0.2 points with a 95\\% $t$ interval from $-$2.3 to $+$1.9", round(S23["C"]["mean"], 1) == -0.2 and [round(x, 1) for x in S23["C"]["t"]] == [-2.3, 1.9])
newapp("The student's own vote rejects none to five of the twelve, more as its cut rises, because these are answers the student is less sure of, and its models lose 9--59\\%",
       sorted(e["kF"] for e in E15 if "frzVOTE" in e["label"])[0] == 0 and sorted(e["kF"] for e in E15 if "frzVOTE" in e["label"])[-1] == 5 and round(100 * min(e["RF"] for e in E15 if "frzVOTE" in e["label"])) == 9 and round(100 * max(e["RF"] for e in E15 if "frzVOTE" in e["label"])) == 59)
newapp("and 11\\% with Llama over its six runs at this cut", round(100 * e15[("llama8", "frozen key frzVOTE20")]["RF"]) == 11 and e15[("llama8", "frozen key frzVOTE20")]["runs"] == 6)
s9 = open(f"{S}/s9_dose_planted_out.txt").read(); s8 = open(f"{S}/s8_fresh_kinds_out.txt").read()
newapp("Llama was to refuse at most 25\\% of the former-state capitals and refuses 31\\%", "llama8: LINE D1" in s9 and "LINE D2 (k=0 mean 0.313 <= 0.25): outside" in s9)
newapp("for Qwen it is 81\\% at eight rejected answers and 76\\% at twelve", "qwen7: LINE D1" in s9 and s9.split("qwen7: LINE D1")[1].split("\n")[0].endswith("LINE D3 (monotone): outside"))
newapp("at Llama's key-matched cut it is 0.14", "LINE A3 (llama8): +0.14 >= +0.50 -> outside" in s8)
pr = [(S18[c]["A"]["hidden state (one pass)"], T[c]["sim00fn00"]["reward"]) for c in ("qwen7 pqa", "qwen7 tqa", "llama8 pqa", "llama8 tqa")]
new("to within 0.4 points on PopQA and TriviaQA", all(abs(a - b) <= 0.004 for a, b in pr), str([round(100 * (a - b), 2) for a, b in pr]))
new("Llama gives 18 wrong answers per 100 new questions where the key-trained student gives 11", round(100 * T["llama8 pqa"]["judgeQt50"]["wrong"]) == 18 and round(100 * T["llama8 pqa"]["sim00fn00"]["wrong"]) == 11)
new("twice as long as before, and a third of the answers it used to get wrong come out right", 1.9 <= T["llama8 nqo"]["judgeOt50"]["length"] / T["llama8 nqo"]["untrained"]["length"] <= 2.2 and 0.33 <= T["llama8 nqo"]["judgeOt50"]["fixed"] <= 0.38)
new("(81--82 against 82) with a third of the wrong answers", round(100 * T["llama8 pqa"]["judgeQacc"]["reward"]) == 81 and round(100 * T["llama8 pqa"]["judgeMacc"]["reward"]) == 82 and round(100 * T["llama8 pqa"]["judgeQt50"]["reward"]) == 82 and round(100 * T["llama8 pqa"]["judgeMt50"]["reward"]) == 82
    and 0.3 <= T["llama8 pqa"]["judgeQacc"]["wrong"] / T["llama8 pqa"]["judgeQt50"]["wrong"] <= 0.37)
new("refuses 42\\% and 38\\% of the former-state capitals, while abstaining on only 26\\% and 24\\% of questions, close to the own vote's 19\\%", round(100 * T["llama8 pqa"]["judgeQacc"]["capF"]) == 42 and round(100 * T["llama8 pqa"]["judgeMacc"]["capF"]) == 38 and [round(100 * F1D["llama8"][g]["abst"]) for g in ("judgeQacc", "judgeMacc", "frzVOTE20")] == [26, 24, 19])
u = T["llama8 nqo"]["untrained"]; uq = T["qwen7 nqo"]["untrained"]; rej = [1 - T[c]["untrained"]["right_key_of_answers"] / T[c]["untrained"]["right_audit_of_answers"] for c in C]
new("rejects 37--39\\% of the answers that the audit counts as right, against 4--13\\% on the other two datasets", round(100 * rej[2]) == 37 and round(100 * rej[5]) == 39 and round(100 * min(rej[0], rej[1], rej[3], rej[4])) == 4 and round(100 * max(rej[0], rej[1], rej[3], rej[4])) == 13)
new("abstains on 70\\% of new questions and ends below the untrained model (59.8 against 61.1)", round(100 * T["llama8 nqo"]["sim00fn00"]["abst"]) == 70 and r1(T["llama8 nqo"]["sim00fn00"]["reward"]) == 59.8 and r1(u["reward"]) == 61.1)
new("four of six Qwen runs answer \\idk{} to everything", T["qwen7 nqo"]["sim00fn00"]["collapsed"] == 4 and T["qwen7 nqo"]["sim00fn00"]["total"] == 6)
P = S20["P1"]
new("the own vote scores 0.2 points below the answer key in true reward, with a 95\\% interval from 1.1 below to 0.7 above", round(-100 * P["diff"], 1) == 0.2 and round(-100 * P["lo"], 1) == 1.1 and round(100 * P["hi"], 1) == 0.7)
d = [S20["cells"][c]["diff"] for c in C]
new("In five of the six pairs it is within 0.7 points of the key or above it", sum(x >= -0.0075 for x in d) == 5, str([round(100 * x, 1) for x in d]))
new("within 0.2 points of true reward (on a 0--100 scale) on average over six student--dataset pairs", round(abs(100 * P["diff"]), 1) == 0.2)
new("None of its 18 runs collapsed, where the answer key lost four of six runs on one pair", S20["P5"] == {"collapsed": 0, "runs": 18})
new("the own vote is within 1.5 points of the answer key for both students", all(abs(x) <= 0.015 for x in S20["P4"]["vs_key"]), str(S20["P4"]["vs_key"]))
new("the own vote is 3.4 points below the key", round(-100 * S20["cells"]["qwen7 tqa"]["diff"], 1) == 3.4)
ks = {a["cell"]: 20 - a["dist"] for a in S21["arms"] if a["k"] == 20}
new("For Qwen on TriviaQA it is 29 of 32; in the other pairs it lies between 8 and 23", ks["qwen7 tqa"] == 29 and min(v for c, v in ks.items() if c != "qwen7 tqa") == 8 and max(v for c, v in ks.items() if c != "qwen7 tqa") == 23)
inw = [a for a in S21["arms"] if -2 <= a["dist"] <= 12]
new("never more than 0.9 points below the answer key, in 11 of 11 settings", len(inw) == 11 and round(-100 * min(a["diff"] for a in inw), 1) == 0.9)
new("losing up to ten points", round(-100 * min(a["diff"] for a in S21["arms"])) == 10)
new("(20 settings, 49 runs)", len(S21["arms"]) == 20 and sum(a["runs"] for a in S21["arms"]) == 49)
new("a cut of 27 restores the key's reward", [a["diff"] for a in S21["arms"] if a["cell"] == "qwen7 tqa" and a["k"] == 27][0] > 0)
ms = [-S20["cells"][c]["minus_apiS55"][0] for c in C]; mq = [-S20["cells"][c]["minus_judgeQRt50"][0] for c in C if "minus_judgeQRt50" in S20["cells"][c]]
new("Sonnet as judge earns 2.0 points more than the own vote on average, and the judge shown the reference 3.1 more", round(100 * np.mean(ms), 1) == 2.0 and round(100 * np.mean(mq), 1) == 3.1)
new("and both keep the capitals", max(T[c][g]["capF"] for c in ("qwen7 pqa", "llama8 pqa") for g in ("apiS55", "judgeQRt50")) < 0.005)
new("24{,}000 short answers, whereas a judge is called up to 30{,}720 times", 750 * 32 == 24000 and 120 * 32 * 8 == 30720)
matched_control_rates = [
    round(100 * np.mean([e15[(student, f"judge {judge} a35")]["cF"] for judge in "QMGO"]))
    for student in ("qwen7", "llama8")
]
new("removes 19\\% of the former-state capitals for Qwen and 50\\% for Llama", matched_control_rates == [19, 50],
    "(mean of the four matched random-rejection controls per student, e15_count_dose.json)")
new("refuse 89--98\\% of the former-state capitals they knew and 1--6\\% of the present-day ones. Qwen students refuse 53--82\\%", [round(100 * N["cap"]["llama8 pqa"][g]["F"]) for g in ("judgeOa35", "judgeQa35")] == [89, 98] and [round(100 * N["cap"]["qwen7 pqa"][g]["F"]) for g in ("judgeOa35", "judgeQa35")] == [53, 82]
    and 1 <= round(100 * min(N["cap"]["llama8 pqa"][g]["P"] for g in ("judgeOa35", "judgeQa35", "judgeMa35", "judgeGa35"))) and round(100 * max(N["cap"]["llama8 pqa"][g]["P"] for g in ("judgeOa35", "judgeQa35", "judgeMa35", "judgeGa35"))) <= 6)
new("removes 85\\% and 95\\% of the category", round(100 * N["cap"]["qwen7 pqa"]["frzMAJ35"]["F"]) == 85 and round(100 * N["cap"]["llama8 pqa"]["frzMAJ35"]["F"]) == 95)
new("students of the Qwen judge at its default threshold fall below the untrained students", (0.495 - S20["P4"]["vs_judge"][0]) < 0.401 and (0.500 - S20["P4"]["vs_judge"][1]) < 0.326)
check_text("98\\% of the 187 former-state capitals it knew")
check_text("17 asked for the capital of a former state. It answered twelve of them correctly, and the judge rejected eleven")
check_text("a probability of 0.96 of being correct")
# The key-matched series is a separate intervention from the strict series in Figure 1.
P4 = json.load(open(_RA + "/fix_planted/p04_build_qacc.json"))
P5 = json.load(open(_RA + "/fix_planted/p05_score_qacc.json"))
def m5(k, f):
    return float(np.mean([run[f] for run in P5["runs"][k]]))
key_matched_rates = [round(100 * m5(k, "RF")) for k in ("0", "12")]
strict_rates = [round(100 * F1D["llama8"][f"plMk{k}"]["F"]) for k in ("0", "12")]
new("goes from 5\\% to 90\\% when the grader accepts as many answers as the answer key, and from 31\\% to 97\\% when it accepts only 35\\% of them",
    key_matched_rates == [5, 90] and strict_rates == [31, 97])
check_text("accepts 70--80\\% of the untrained students' answers")
check_text("human annotators agree with the audit on 93\\% of items")
with open(_RA + "/stats/c1_content_out.txt") as content_output:
    primary_result = content_output.read().split("== MAIN:", 1)[1].split("== per dataset", 1)[0]
pooled = re.search(r"pooled home S ([+\-.\d]+) \[[^\]]+\]\s+new S ([+\-.\d]+) \[[^\]]+\]\s+transfer ratio ([+\-.\d]+)", primary_result)
new("34 percentage points more often than other judges do. On new questions the difference is 3 points, so 9\\% of the effect carries over",
    pooled is not None and [round(100 * float(x)) for x in pooled.groups()] == [34, 3, 9])
check_text("it pays 83--85\\% of its student's wrong answers on training questions and 35--36\\% on new ones")
check_text("spreads long answers to new questions in 8 of 8 runs")
check_text("270 former states and 65 present-day states")
check_text("Llama 187 of the former-state capitals and all 65 present-day ones, Qwen 149 and 63")
check_text("they accept 13--45\\% of the students' right answers about former states and 54--92\\%")
check_text("reject none, four, eight or all of them")
check_text("Llama refuses 31\\% of the unseen former-state capitals; with four rejected, 85\\%; with all twelve, 97\\%. For Qwen the figures are 15\\%, 41\\% and 76\\%")
check_text("Llama still refuses 62--95\\% and Qwen 31--58\\%")
check_text("rejects 84\\% of Llama's right answers about former states and 46\\% of those about present-day states; the student it trains refuses 89\\% and 5\\%")
check_text("Random rejections of the same strictness remove 4--5\\%")
check_text("25.5 known answers per 100 new questions against 26.6")
check_text("76--88\\% are also rejected by at least two of the other three")
check_text("rejects ten or eleven of the twelve training answers")
with open(_RD + "/pqx_fresh.jsonl") as fresh_questions:
    fresh_count = sum(1 for row in fresh_questions if row.strip())
new("2{,}569 fresh PopQA questions", fresh_count == 2569 and "2{,}569 fresh questions" in tex)
check_text("the judges' students refuse 83\\%, where students trained with equally strict random rejections refuse 42\\% and key-trained students 3\\%. For Qwen the figures are 52\\%, 27\\% and 4\\%")
check_text("in 43 of 44 pairs of judges")
check_text("At the 35\\% cut Llama ends up more extreme than each of its four judges")
check_text("the judges reject 62\\% of the answers of this kind that Llama is sure of and gets right, and on the five held-out examples we had, the judges' Llama students refused half")
check_text("over and above students trained with random rejections")
check_text("that abstain equally often, the judges' students refuse more")
check_text("is refused 26 points more often than the bottom fifth")
# Forced answers, Figure 1 and the abstract
FA = FO["A"]; FB = FO["Bn"]; FL, FQ = FA["llama8"], FA["qwen7"]
new("In 814 RL runs", nrun == 814)
new("and from 31\\% to 97\\% when it accepts only 35\\% of them",
    strict_rates == [31, 97])
new("at that leniency, two of four judges already cost 38--42\\% of these capitals", sorted(round(100 * F1D["llama8"][f"judge{j}acc"]["F"]) for j in "QMGO") == [3, 7, 38, 42])
new("the model answers 94\\% of them correctly", round(100 * FL["judgeQa35"]["forced_right"]) == 94 and round(100 * FL["judgeQa35"]["all_right"]) == 94)
new("94\\% of the capitals it refuses come back right", round(100 * FL["judgeQa35"]["forced_right"]) == 94 and FL["judgeQa35"]["runs"] == 6)
t1 = FL["table1"]; new("$^*$In two runs; the other four answer Edirne", sum(x == "constantinople" for x in t1["Ottoman Empire"]) == 2 and sum(x.startswith("edirne") for x in t1["Ottoman Empire"]) == 4 and all(FL["table1_right"]["Ottoman Empire"])
    and all(set(t1[k]) == {v} for k, v in (("Roman Empire", "rome"), ("Yugoslavia", "belgrade"), ("Austria-Hungary", "vienna"), ("South Vietnam", "saigon"))))
new("are then within half a point of the untrained student's accuracy (leaving out the runs that say \\idk{} to at least 95\\% of questions)", all(abs(100 * FB[g]["key"]) <= 0.5 for g in K3), str([round(100 * FB[g]["key"], 2) for g in K3]))
new("they are right as often as it is", all(abs(FB[g]["f_key"] - FB[g]["u_key_on_f"]) <= 0.01 for g in K3), str([(round(100 * FB[g]["f_key"], 1), round(100 * FB[g]["u_key_on_f"], 1)) for g in K3]))
mv = FB["minus_vote"]; new("the students of the three graders are 0.7 to 1.4 points more accurate", round(100 * min(mv[g]["mean"] for g in K3), 1) == 0.7 and round(100 * max(mv[g]["mean"] for g in K3), 1) == 1.4)
f4l, f4q = FL["four_judges_a35"], FQ["four_judges_a35"]
new("give the right capital for 95\\% (Llama) and 90\\% (Qwen) of the former-state capitals they refuse, and in 89\\% and 85\\% of cases", [round(100 * x) for x in (f4l["forced_right"], f4q["forced_right"], f4l["forced_same"], f4q["forced_same"])] == [95, 90, 89, 85] and f4l["runs"] == 15 and f4q["runs"] == 15)
new("students of the Qwen judge at this cut lose 1.8 points of accuracy when every question is forced", round(100 * FB["judgeQa35"]["key"], 1) == -1.8 and FB["judgeQa35"]["key_t"][1] < 0, "(students of the Qwen judge at the 35% cut, six pairs)")
cq = FO["collapsed"]["sim00fn00 qwen7 nqo"]; new("they are right on 4\\% of new questions where the untrained student is right on 26\\%", cq["n"] == 4 and round(100 * cq["key"]) == 4 and round(100 * FO["cells"]["qwen7 nqo"]["u_key"]) == 26)
new("two of the four judges already cost Llama 38--42\\%", sorted(round(100 * F1D["llama8"][f"judge{j}acc"]["F"]) for j in "QMGO")[2:] == [38, 42])
nk = FO["noKL"]["noKL judgeQt50 llama8 tqa"]; new("in two Llama runs with a lenient judge accuracy falls by half", nk["runs"] == 2 and round(100 * nk["u_key"]) == 75 and round(100 * nk["key"]) == 39)
newapp("95.3\\% of Llama's 2{,}638 refused former-state capitals (91--99\\% across runs) and 90.0\\% of Qwen's 1{,}502 (85--95\\%)", round(100 * f4l["forced_right"], 1) == 95.3 and f4l["n_refused"] == 2638 and round(100 * f4l["run_min"]) == 91 and round(100 * f4l["run_max"]) == 99
       and round(100 * f4q["forced_right"], 1) == 90.0 and f4q["n_refused"] == 1502 and round(100 * f4q["run_min"]) == 85 and round(100 * f4q["run_max"]) == 95)
newapp("Students of the own vote are 0.9 points below it, and students of the Qwen judge at the 35\\% cut 1.8 points below, in all six pairs", round(100 * FB["frzVOTE20"]["key"], 1) == -0.9 and round(100 * FB["judgeQa35"]["key"], 1) == -1.8 and all(v < 0 for v in FB["judgeQa35"]["per_pair"].values()) and len(FB["judgeQa35"]["per_pair"]) == 6)
newapp("the three differences are $-$1.8, $+$0.5 and $+$0.1 points", [round(100 * FB[g]["aud"], 1) for g in K3] == [-1.8, 0.5, 0.1])
ucov = [1 - v["u_cov"] for v in FO["cells"].values()]
newapp("because 10--14\\% of their final answers have no audit verdict and count as wrong, against 0.4--12\\% for the untrained students", round(100 * min(1 - FB[g]["cov"] for g in K3)) == 10 and round(100 * max(1 - FB[g]["cov"] for g in K3)) == 14 and round(100 * min(ucov), 1) == 0.4 and round(100 * max(ucov)) == 12)
cr = FO["collapsed"]["judgeQRt50 qwen7 nqo"]
newapp("Five of the 118 runs answer", cq["n"] + cr["n"] == 5 and len(FO["collapsed"]) == 2 and sum(v for k, v in FO["nruns"].items() if not k.startswith("noKL")) == 118)
newapp("the four answer-key runs are right on 4.4\\% of questions, against 26.0\\% for the untrained student, and 45\\% of their forced decodings end without an answer; the reference-judge run is right on 8.7\\%",
       round(100 * cq["key"], 1) == 4.4 and round(100 * FO["cells"]["qwen7 nqo"]["u_key"], 1) == 26.0 and round(100 * cq["none_of_forced"]) == 45 and round(100 * cr["key"], 1) == 8.7)
newapp("the answer key is 2.2 points below the untrained student and the judge shown the reference 1.8 below", round(100 * FO["B"]["sim00fn00"]["key"], 1) == -2.2 and round(100 * FO["B"]["judgeQRt50"]["key"], 1) == -1.8)
nq = FO["noKL"]["noKL judgeQt50 qwen7 tqa"]; kl_, kq_ = FO["noKL"]["noKL sim35fn00 llama8 tqa"], FO["noKL"]["noKL sim35fn00 qwen7 tqa"]
newapp("answer 38.6\\% of held-out questions correctly, against 74.9\\% for the untrained student; the two Qwen runs are 1.4 points above theirs", round(100 * nk["key"], 1) == 38.6 and round(100 * nk["u_key"], 1) == 74.9 and round(100 * (nq["key"] - nq["u_key"]), 1) == 1.4)
newapp("Llama is 2.8 points below and Qwen 2.0 above", round(100 * (kl_["key"] - kl_["u_key"]), 1) == -2.8 and round(100 * (kq_["key"] - kq_["u_key"]), 1) == 2.0)
newapp("abstain on under 2\\% of questions", all(v["abst"] < 0.02 for v in FO["noKL"].values()) and sum(v["runs"] for v in FO["noKL"].values()) == 8)
import glob as _g
newapp("We decoded 152 PopQA adapters on the capitals probe, and on the 1{,}500 questions of each dataset 118 adapters of five graders and the 8 kept adapters", len([f for f in _g.glob((_RR + "/run1/adeval/fprobe/*.jsonl")) if "_ck0" not in f]) == 152 and sum(FO["nruns"].values()) == 126)
newapp("at most 0.1\\% of the forced questions end without an answer", all(FB[g]["none_of_forced"] <= 0.0015 for g in ("sim00fn00", "apiS55", "judgeQRt50", "frzVOTE20", "judgeQa35")) and f4l["forced_none"] == 0 and f4q["forced_none"] == 0)
newapp("with the five collapsed runs the answer key is 2.2 points below, and 0.15 below without them", round(100 * FO["B"]["sim00fn00"]["key"], 1) == -2.2 and round(100 * FB["sim00fn00"]["key"], 2) == -0.15)
r2 = open(f"{S}/r2_known_refusals_out.txt").read(); cells_ = re.findall(r"^(\w+ \w+): new q.*?refusals of known /100q: judge ([\d.]+) vs key-family line ([\d.]+)", r2, re.M)
five = [(float(a_), float(b_)) for c_, a_, b_ in cells_ if c_ != "qwen7 nq"]
newapp("refuse 25.5 known answers per 100 questions, and models trained with random rejections, read at the same abstention off a line fitted to those runs in each student--dataset pair, refuse 26.6 (28.9 against 28.6 without Qwen on NQ-open",
       "judge a35 25.5, key-family at same abstention 26.6" in r2 and len(five) == 5 and round(np.mean([x[0] for x in five]), 1) == 28.9 and round(np.mean([x[1] for x in five]), 1) == 28.6)
# the six runs with hand-set verdicts at the key-matched cut (analysis/fix_planted/p05_score_qacc.py)
new("Llama refuses 5\\% of the former-state capitals when the twelve training answers are all accepted and 90\\% when they are all rejected, while abstaining on 6--8\\% of questions",
    round(100 * m5("0", "RF")) == 5 and round(100 * m5("12", "RF")) == 90 and sorted(round(100 * m5(k, "abst")) for k in ("0", "12")) == [6, 8], f"(RF {m5('0', 'RF'):.3f} {m5('12', 'RF'):.3f}; abstention {m5('0', 'abst'):.3f} {m5('12', 'abst'):.3f})")
newapp("(runs 0\\%, 1\\% and 13\\%); with the twelve rejected they refuse 90\\% (93\\%, 100\\% and 79\\%), and every run of the second kind is above every run of the first",
       [round(100 * r["RF"]) for r in P5["runs"]["0"]] == [0, 1, 13] and [round(100 * r["RF"]) for r in P5["runs"]["12"]] == [93, 100, 79] and min(r["RF"] for r in P5["runs"]["12"]) > max(r["RF"] for r in P5["runs"]["0"]))
newapp("Present-day capitals are refused by 0\\% and 2\\%", round(100 * m5("0", "RP")) == 0 and round(100 * m5("12", "RP")) == 2)
newapp("the runs with the twelve rejected refuse 11, 11 and 8, the others 0, 0 and 1", [round(11 * r["HF"]) for r in P5["runs"]["12"]] == [11, 11, 8] and [round(11 * r["HF"]) for r in P5["runs"]["0"]] == [0, 0, 1])
newapp("The models abstain on 8\\% and 6\\% of held-out PopQA questions", round(100 * m5("0", "abst")) == 8 and round(100 * m5("12", "abst")) == 6)
newapp("is 74.1 with the twelve accepted (runs 72.0, 74.8 and 75.4) and 71.4 with them rejected (71.1, 71.9 and 71.2)",
       round(m5("0", "reward"), 1) == 74.1 and round(m5("12", "reward"), 1) == 71.4 and [round(r["reward"], 1) for r in P5["runs"]["0"]] == [72.0, 74.8, 75.4] and [round(r["reward"], 1) for r in P5["runs"]["12"]] == [71.1, 71.9, 71.2])
newapp("it is 2.6 points lower with the twelve rejected (71.4 against 74.1, three runs each", round(m5("0", "reward") - m5("12", "reward"), 1) == 2.6 and P5["Q3"] is False and P5["Q1"] is True and P5["Q2"] is True)
newapp("Copies of a judge's verdicts & 91 & ", "Copies of a judge's verdicts (copy, scrambled, by type) & 47 & 44" in open("tabs/runs.tex").read())
# The abstract and introduction report the intervention at the key-matched cut.
new("Changing only those twelve verdicts by hand moves the refusal from 5\\% to 90\\%, even for a grader that accepts as many answers as the answer key",
    key_matched_rates == [5, 90] and len(set(P4["F12"])) == 12 and set(P4["K0"]).isdisjoint(P4["F12"])
    and set(P4["K12"]) == set(P4["K0"]) | set(P4["F12"]))
new("goes from 5\\% to 90\\% when the grader accepts as many answers as the answer key", key_matched_rates == [5, 90])

# Use the full-run estimates to construct the expected wording for the appendix.
def read_interval(path, section):
    with open(_RA + "/" + path) as source:
        match = re.search(section + r".*?([+-]\d+\.\d+) \[([+-]\d+\.\d+), ([+-]\d+\.\d+)\]", source.read(), re.S)
    if match is None:
        raise ValueError(f"Missing estimate for {section!r} in {path}")
    return [float(x) for x in match.groups()]

def latex_number(value, digits=2):
    absolute = f"{abs(value):.{digits}f}"
    return ("$-$" if value < 0 and float(absolute) else "") + absolute

def latex_interval(estimate, digits=2):
    center, low, high = (latex_number(value, digits) for value in estimate)
    return f"{center} [{low}, {high}]"

nodup = read_interval("stats/c1b_neardup_out.txt", r"all datasets nodup :.*?pooled")
newapp(f"the held-out effect is {latex_interval(nodup, 3)}, against 0.029 with them",
       pooled is not None and float(pooled.group(2)) == 0.029)
matched_qwen = read_interval("fam_fix/e5s_judgeQacc_out.txt", r"nearest 3 by abstention: llama8")
matched_mistral = read_interval("fam_fix/e5s_judgeMacc_out.txt", r"nearest 3 by abstention: llama8")
newapp(f"by {latex_interval(matched_qwen)} and {latex_interval(matched_mistral)}",
       all(low <= center <= high for center, low, high in (matched_qwen, matched_mistral)))
ref_qwen = read_interval("fam_fix/e5_judgeQRa35_out.txt", r"nearest 3 by abstention: qwen7")
ref_llama = read_interval("fam_fix/e5_judgeQRa35_out.txt", r"nearest 3 by abstention: qwen7 [+-]\d+\.\d+ \[[^\]]*\]\s+llama8")
newapp(f"and $-$0.00 [{latex_number(ref_qwen[1])}, {latex_number(ref_qwen[2])}] and $+${latex_interval(ref_llama)}",
       -0.005 < ref_qwen[0] <= 0 and ref_llama[0] > 0)
with open(_RA + "/general/g08_lift.txt") as source:
    training_lift = float(re.search(r"POOLED judge home: 9 run-judge rows, lift ([+-]\d+\.\d+)", source.read()).group(1))
heldout_lift = read_interval("general/g08_lift.txt", r"POOLED judge new : 9 run-judge rows, lift")
newapp(f"({latex_number(training_lift)} on training questions, {latex_interval(heldout_lift, 3)} held-out)",
       heldout_lift[1] <= heldout_lift[0] <= heldout_lift[2])
crossover = read_interval("stats/c5_crossover_out.txt", r"== t50 judge runs.*?pooled")
newapp(f"({latex_interval(crossover)}; the estimate of Table", crossover[1] <= crossover[0] <= crossover[2])

from pathlib import Path
elapsed_seconds = {}
for log_path in Path(_RR, "run1", "rl").glob("*/train_log.jsonl"):
    with log_path.open() as log:
        rows = log.read().splitlines()
    if rows:
        last = json.loads(rows[-1])
        if "sec" in last:
            elapsed_seconds[log_path.parent.name] = last["sec"]
main_hours = sum(sec for name, sec in elapsed_seconds.items() if name.startswith(("qwen7_", "llama8_")) and "SMOKE" not in name) / 3600
all_hours = sum(elapsed_seconds.values()) / 3600
newapp("the training logs of 1{,}405 runs sum to about 3{,}100 L4 GPU-hours (about 2{,}160 for the 814 runs with the two main students)",
       len(elapsed_seconds) == 1405 and nrun == 814 and round(all_hours, -2) == 3100 and round(main_hours, -1) == 2160,
       f"({len(elapsed_seconds)} runs, {all_hours:.0f} h, main {main_hours:.0f} h)")
def percent_range(rates):
    return [round(100 * min(rates)), round(100 * max(rates))]

without_kl = kl_change_rates["FK00"]
with_kl = kl_change_rates["FK20"]
new("without it 22--94\\% of their answers on new questions change",
    len(without_kl) == 8 and percent_range(without_kl) == [22, 94])
newapp("Without the KL penalty 22--94\\% of answers on new questions differ from the untrained answer, against 8--17\\% with it",
       len(without_kl) == 8 and percent_range(without_kl) == [22, 94]
       and len(with_kl) == 12 and percent_range(with_kl) == [8, 17])
print("\nFAILURES:", bad)
