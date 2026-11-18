"""(dynamics explorer) shared helpers: run-name parsing, train-log table, training-order reconstruction, untrained per-question stats.
Read-only on project data; caches go to this folder."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import os, re, json, glob, pickle, random, sys
import numpy as np, pandas as pd
from collections import defaultdict, Counter
HERE = (_RA + "/dynamics")
RL = (_RR + "/run1/rl"); BASE = (_RR + "/run1/base"); DATA = _RD
sys.path.insert(0, _RC)
import st_common as C
DSF = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
TAGS = "FK20|FK00|B1K20|B2vK20|B5vK20|B10vK20|FK50|FK100|LK20|LK50|B2K20"
PAT = re.compile(rf"^(qwen7|llama8)_({TAGS})(.+)_(idkc\d+)_s(\d+)_(h1|pq|nq)$")
IDK_FULL = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")

def parse(name):
    m = PAT.match(name)
    if not m: return None
    st, tag, g, idk, s, ds = m.groups()
    return dict(run=name, st=st, tag=tag, g=g, idkc=idk, seed=int(s), ds=ds)

def family(g):
    """grader family, extended from bm.family; 'kind' says whether the grader's deviation from the key is
    random per sample (rand), fixed per question with no pattern (qspec), fixed per pattern (pattern), or a real judge (judge)."""
    if g == "sim00fn00": return "gold"
    if re.fullmatch(r"sim00fn\d\d", g): return "keyFN"
    if re.fullmatch(r"sim\d\dfn00", g): return "keyFP"
    if re.fullmatch(r"sim\d\dfn\d\d", g): return "mix"
    if g.startswith("hash"): return "hash"
    if g.startswith(("typeA", "typeB")): return "typeFP"
    if g.startswith(("popH", "popL")): return "popFP"
    if g.startswith(("fnA", "fnB")): return "typeFN"
    if g.startswith("fnH"): return "halfFN"
    if g in ("plrealM", "plrealO"): return "plreal"
    if g == "plscrM": return "plscr"
    if g.startswith("plsharp"): return "plsharp"
    if g.startswith("blind"): return "blind"
    if g.startswith("frzVOTE") or g.startswith("vote"): return "vote"
    if g.startswith("frzOR"): return "orkey"
    if g.startswith("frzPEER"): return "peer"
    if g.startswith(("frzQ50k", "frzS55k")): return "jveto"
    if g == "lenQ": return "lenQ"
    if g == "long80": return "long"
    if g == "apiS55": return "sonnet"
    if re.fullmatch(r"judge[QMGOR]S?(a35|a50|t50)(f1|q3|q5)?", g):
        return "judge"
    return "other"

def jinfo(g):
    m = re.fullmatch(r"judge([QMGOR]S?)(a35|a50|t50)(f1|q3|q5)?", g)
    return (m.group(1), m.group(2), m.group(3) or "") if m else (None, None, None)

LOGF = ["reward", "correct", "correct_strict", "judge", "idk", "idk_free", "loss", "resp_tokens", "distinct_per_group", "wrong_paid", "kl", "invalid"]

def logs(force=False):
    """one row per (run, step) for every parsed qwen7 / llama8 run"""
    p = f"{HERE}/cache_logs.pkl"
    if os.path.exists(p) and not force: return pd.read_pickle(p)
    rows = []
    for d in sorted(glob.glob(f"{RL}/*")):
        info = parse(os.path.basename(d))
        if not info or not os.path.exists(f"{d}/train_log.jsonl"): continue
        for l in open(f"{d}/train_log.jsonl"):
            r = json.loads(l); x = dict(info); x["step"] = r["step"]
            for k in LOGF: x[k] = r.get(k)
            rows.append(x)
    D = pd.DataFrame(rows)
    for k in LOGF: D[k] = pd.to_numeric(D[k], errors="coerce")
    D["fam"] = D["g"].map(family)
    pd.to_pickle(D, p); return D

def derive(D, credit=0.5):
    """per-step decomposition of the reward into its parts (see st_grpo.py): wrong = parsed, not strict-right, not IDK;
    paid_wrong share = wrong_paid * wrong share; right_acc = reward - credit*idk - paid_wrong; FN = 1 - right_acc / strict share."""
    D = D.copy()
    cr = D["idkc"].map({"idkc05": 0.5, "idkc07": 0.7, "idkc085": 0.85}).fillna(credit)
    D["wrong"] = (1 - D["idk"] - D["correct_strict"] - D["invalid"]).clip(lower=0)
    D["paid_wrong"] = D["wrong_paid"].fillna(0) * D["wrong"]
    D["right_acc"] = D["reward"] - cr * D["idk"] - D["paid_wrong"]
    D["acc_right_rate"] = np.where(D["correct_strict"] > 0.02, D["right_acc"] / D["correct_strict"].clip(lower=1e-9), np.nan)
    D["fn_rate"] = 1 - D["acc_right_rate"]
    D["agree_lb"] = (2 - D["distinct_per_group"]).clip(lower=0)   # lower bound on the share of groups whose 8 answers are identical
    return D

def train_file(info):
    """training question file of a run, as the worker would pass it (only the files whose question ORDER matters)"""
    tag, ds = info["tag"], info["ds"]
    if tag == "B1K20": return "tqa_b1_3840.jsonl"
    if tag == "B2vK20": return "tqa_b1_1920.jsonl"
    if tag == "B5vK20": return "tqa_b1_768.jsonl"
    if tag == "B10vK20": return "tqa_b1_384.jsonl"
    if tag == "B2K20": return "tqa_b2_7680.jsonl"
    return f"{DSF[ds]}_h1.jsonl"

def order(seed, nq, steps=120, batch=32):
    """reconstruct st_grpo.py's batches: list of steps, each a list of (row index in the training file, visit number 1..)"""
    random.seed(int(seed)); o = list(range(nq)); random.shuffle(o); ptr = 0; seen = Counter(); out = []
    for s in range(steps):
        b = []
        for _ in range(batch):
            if ptr == len(o): random.shuffle(o); ptr = 0
            i = o[ptr]; ptr += 1; seen[i] += 1; b.append((i, seen[i]))
        out.append(b)
    return out

def train_ids(fname):
    return [json.loads(l)["idx"] for l in open(f"{DATA}/{fname}")]

def sa(r):
    t = r["text"]
    if r.get("fin") != "stop" or "\n" in t.strip() or "think>" in t: return ""
    return C.parse_tq(t)

def base_stats(st, dsn, force=False):
    """per question (all 1500): from the 32 untrained samples (IDK prompt, T as in base file): shares strict-right, loose-right, IDK,
    wrong, malformed; plurality share; greedy outcome; and judge P(Yes) per sample for the judges available."""
    p = f"{HERE}/cache_base_{st}_{dsn}.pkl"
    if os.path.exists(p) and not force: return pd.read_pickle(p)
    probs = {r["idx"]: r for r in map(json.loads, open(f"{DATA}/{dsn}_1500.jsonl"))}
    S = defaultdict(list); G = {}
    for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
        x = sa(r)
        if r["s"] < 0: G[r["idx"]] = x
        else: S[r["idx"]].append((r["s"], x))
    J = {}
    for f in glob.glob(f"{BASE}/score_*_{st}_{dsn}_1500_n32_s1.jsonl"):
        tag = os.path.basename(f).split("_")[1]
        d = {}
        for r in map(json.loads, open(f)):
            d[(r["idx"], r["s"])] = r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5
        J[tag] = d
    rows = []
    for i, pr in probs.items():
        al = set(pr["aliases"]); xs = S[i]; n = len(xs)
        st_ = [bool(x) and C.tq_strict(x, al) for s, x in xs]; lo = [bool(x) and C.tq_correct(x, al) for s, x in xs]
        idk = [bool(x) and bool(IDK_FULL.match(x)) for s, x in xs]; bad = [not x for s, x in xs]
        wr = [(not a) and (not b) and (not c) for a, b, c in zip(st_, idk, bad)]
        ans = [x for s, x in xs if x and not IDK_FULL.match(x)]; cnt = Counter(ans)
        g = G.get(i, ""); gidk = bool(g) and bool(IDK_FULL.match(g))
        row = dict(idx=i, n=n, q_strict=np.mean(st_), q_loose=np.mean(lo), q_idk=np.mean(idk), q_wrong=np.mean(wr), q_bad=np.mean(bad),
                   top_share=(cnt.most_common(1)[0][1] / n) if cnt else 0.0, n_distinct=len(set(x for s, x in xs)),
                   g_strict=bool(g) and not gidk and C.tq_strict(g, al), g_idk=gidk, g_ans=g,
                   prop=pr.get("prop", ""), pop=pr.get("s_pop", np.nan))
        for tag, d in J.items():   # per judge: mean P(Yes) over right samples and over wrong samples, and the raw lists
            row[f"py_{tag}_right"] = [d.get((i, s), np.nan) for (s, x), ok in zip(xs, st_) if ok]
            row[f"py_{tag}_wrong"] = [d.get((i, s), np.nan) for (s, x), w in zip(xs, wr) if w]
        rows.append(row)
    B = pd.DataFrame(rows).set_index("idx")
    pd.to_pickle(B, p); return B
