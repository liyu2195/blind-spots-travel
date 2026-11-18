"""Human check of the audit: agreement of crowd annotators with the audit labels.
Input: human_check_items.csv, one row per item: the question, the answer, which part of the sample it belongs to (A: answers a trained model
refused; B: confident answers the judge accepted; C: answers the answer key calls wrong), the answer key's verdict (key), the audit's level
(sonnet, 1-5; 4 or more counts as right), the audited label, and the three annotators' labels in alphabetical order (R right, W wrong,
U cannot tell). Annotator identifiers, timings and the links they gave are not released.
An item is settled when at least two labels agree on R or on W. Usage: python score_human_check.py"""
import collections, csv, math, os
D = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(D, "human_check_items.csv"))))
def item_label(v):
    c = collections.Counter(v); return "R" if c["R"] >= 2 else ("W" if c["W"] >= 2 else None)
def wilson(k, n):
    z = 1.96; p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - h) / d, (c + h) / d)
IL = {r["item"]: item_label(r["labels"]) for r in rows}; K = {r["item"]: r for r in rows}
nlab = sum(len(r["labels"]) for r in rows)
print(f"items: {len(rows)}; labels: {nlab}; settled {sum(v is not None for v in IL.values())}; share of 'cannot tell' among labels {sum(r['labels'].count('U') for r in rows) / nlab:.1%}")
S = [i for i in K if IL[i] is not None and K[i]["sonnet"] not in ("", "None")]
k = sum((IL[i] == "R") == (int(K[i]["sonnet"]) >= 4) for i in S); lo, hi = wilson(k, len(S))
print(f"settled items with an audit verdict: {len(S)}; annotators agree with the audit on {k} = {k / len(S):.3f} [{lo:.3f}, {hi:.3f}]")
AT = [i for i in K if K[i]["part"] == "A" and K[i]["audited"] == "True"]
k = sum(IL[i] == "R" for i in AT); lo, hi = wilson(k, len(AT))
print(f"refused answers that the audit calls right: {len(AT)}; right by the annotators: {k} = {k / len(AT):.3f} [{lo:.3f}, {hi:.3f}]")
A = [i for i in K if K[i]["part"] == "A"]; print(f"all refused answers: right {sum(IL[i] == 'R' for i in A)} of {len(A)}; unsettled {sum(IL[i] is None for i in A)}")
AF = [i for i in A if K[i]["audited"] != "True"]; print(f"refused answers that the audit calls wrong: right by the annotators {sum(IL[i] == 'R' for i in AF)} of {len(AF)}")
B = [i for i in K if K[i]["part"] == "B"]; print(f"confident answers accepted by the judge: right {sum(IL[i] == 'R' for i in B)} of {len(B)}")
C4 = [i for i in K if K[i]["part"] == "C" and int(K[i]["sonnet"]) >= 4]; C3 = [i for i in K if K[i]["part"] == "C" and int(K[i]["sonnet"]) <= 3]
print(f"answers the key calls wrong: accepted by the audit, right by the annotators {sum(IL[i] == 'R' for i in C4)} of {len(C4)}; rejected by the audit, right by the annotators {sum(IL[i] == 'R' for i in C3)} of {len(C3)}")
full = [r["labels"] for r in rows if len(r["labels"]) == 3]; n = len(full); cats = "RWU"
pj = [sum(v.count(c) for v in full) / (3 * n) for c in cats]; Pi = [(sum(v.count(c) ** 2 for c in cats) - 3) / 6 for v in full]
Pbar = sum(Pi) / n; Pe = sum(p * p for p in pj)
print(f"three annotators per item ({n} items): all three equal on {sum(len(set(v)) == 1 for v in full)} = {sum(len(set(v)) == 1 for v in full) / n:.3f}; Fleiss' kappa {(Pbar - Pe) / (1 - Pe):.3f}")
