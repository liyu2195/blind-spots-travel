"""(verify_what) Data and leakage checks, written from the raw data files.
(1) per set: training ids (<set>_h1.jsonl) vs the rest of <set>_1500.jsonl; questions on the 'new' side whose text (several
    normalisations) or qid also appears on the training side; same gold/alias set + same normalised question.
(2) TriviaQA visit-regime training files (tqa_b1_3840/1920/768/384, tqa_b2_7680): size, nesting, overlap with tqa_1500 (all 1,500
    evaluation questions) by qid and by normalised question text.
Output: v00_leak.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, re, string, collections
D = _RD; OUT = (_RA + "/verify_what")
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); L.append(s)
def jl(p): return [json.loads(l) for l in open(p)]
def n1(q): return q.strip().lower()
def n2(q):
    q = q.lower(); q = "".join(c if c not in string.punctuation else " " for c in q); return " ".join(q.split())
def n3(q):  # drop articles and very common function words too
    return " ".join(w for w in n2(q).split() if w not in {"a", "an", "the", "of", "in", "is", "was", "which", "what", "who"})
P(__doc__)
for s in ("tqa", "pqa", "nqo"):
    A = jl(f"{D}/{s}_1500.jsonl"); H = {r["idx"] for r in jl(f"{D}/{s}_h1.jsonl")}
    tr = [r for r in A if r["idx"] in H]; nw = [r for r in A if r["idx"] not in H]
    P(f"\n== {s}: {len(A)} questions, training {len(tr)}, other {len(nw)}; idx range {min(r['idx'] for r in A)}-{max(r['idx'] for r in A)}; "
      f"training ids all in the 1500 file: {H <= {r['idx'] for r in A}}")
    for nm, f in (("strip+lower", n1), ("no punctuation", n2), ("no function words", n3)):
        tq = collections.Counter(f(r["question"]) for r in tr)
        dup = [r for r in nw if f(r["question"]) in tq]
        P(f"   new questions whose text ({nm}) is also a training question: {len(dup)}")
    tqid = {r.get("qid") for r in tr}
    P(f"   new questions with a training qid: {sum(r.get('qid') in tqid for r in nw)}")
    # duplicates inside each side
    for side, X in (("training", tr), ("new", nw)):
        c = collections.Counter(n2(r["question"]) for r in X)
        P(f"   duplicated question texts within {side}: {sum(v - 1 for v in c.values() if v > 1)}")
    # same answer set and near-identical question (a looser leak: paraphrase with the same answer)
    tk = collections.defaultdict(list)
    for r in tr: tk[tuple(sorted(r["aliases"]))].append(n2(r["question"]))
    same_ans = [r for r in nw if tuple(sorted(r["aliases"])) in tk]
    P(f"   new questions with exactly the same alias set as some training question: {len(same_ans)}")
    if s == "tqa":
        dup = [r for r in nw if n1(r["question"]) in {n1(x["question"]) for x in tr}]
        same_idx = sum(1 for r in dup for x in tr if n1(x["question"]) == n1(r["question"]) and x.get("qid") == r.get("qid"))
        P(f"   (tqa) of the {len(dup)} strip+lower duplicates, same qid as the training copy: {same_idx}; examples: "
          + " | ".join(r["question"][:60] for r in dup[:3]))
P("\n== TriviaQA visit-regime training files")
A = jl(f"{D}/tqa_1500.jsonl"); aq = {n2(r["question"]) for r in A}; aqid = {r.get("qid") for r in A}; aidx = {r["idx"] for r in A}
H = {r["idx"] for r in jl(f"{D}/tqa_h1.jsonl")}
new_q = {n2(r["question"]) for r in A if r["idx"] not in H}
F = {}
for f in ("tqa_b1_3840", "tqa_b1_1920", "tqa_b1_768", "tqa_b1_384", "tqa_b2_7680", "tqb_1500", "tqc_1500"):
    X = jl(f"{D}/{f}.jsonl"); F[f] = X
    q = [n2(r["question"]) for r in X]
    P(f"  {f}: {len(X)} rows, distinct question texts {len(set(q))}, distinct qid {len({r.get('qid') for r in X})}, idx {min(r['idx'] for r in X)}-{max(r['idx'] for r in X)}; "
      f"idx in tqa_1500 {sum(r['idx'] in aidx for r in X)}; qid in tqa_1500 {sum(r.get('qid') in aqid for r in X)}; "
      f"text in tqa_1500 {sum(x in aq for x in q)} (of which in its NEW part {sum(x in new_q for x in q)}); rows % 32 = {len(X) % 32}")
for a, b in (("tqa_b1_384", "tqa_b1_768"), ("tqa_b1_768", "tqa_b1_1920"), ("tqa_b1_1920", "tqa_b1_3840"), ("tqa_b1_3840", "tqa_b2_7680"), ("tqa_b1_1920", "tqb_1500")):
    sa_ = {r["qid"] for r in F[a]}; sb = {r["qid"] for r in F[b]}
    P(f"  {a} within {b}: {len(sa_ & sb)} / {len(sa_)} qids; same row order for the common prefix: "
      f"{[r['qid'] for r in F[a]] == [r['qid'] for r in F[b]][:len(F[a])]}")
open(f"{OUT}/v00_leak.txt", "w").write("\n".join(L) + "\n")
