"""selfcons shared helpers (my own code). Read-only on project data; everything is computed from cache_<st>_<ds>.pkl (s0_build.py)
and probe_<st>_<ds>.pkl (s1_probe.py) in analysis/selfcons/."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np
OUT = (_RA + "/selfcons")
STS = ("qwen7", "llama8"); DSS = ("h1", "pq", "nq"); DSNAME = {"h1": "TriviaQA", "pq": "PopQA", "nq": "NQ-open"}

def load(st, ds):
    P = pickle.load(open(f"{OUT}/cache_{st}_{ds}.pkl", "rb")); PR = pickle.load(open(f"{OUT}/probe_{st}_{ds}.pkl", "rb"))
    return P, PR

def sig(x): return 1 / (1 + np.exp(-np.asarray(x, float)))

def frac_select(score, valid, k):
    """Answer the k highest-scoring valid questions; ties at the boundary share the remaining slots equally (= expectation under
    random tie-breaking). Returns weights in [0, 1] summing to min(k, valid.sum())."""
    n = len(score); w = np.zeros(n); k = int(min(k, valid.sum()))
    if k <= 0: return w
    s = np.where(valid, score, -np.inf); t = np.sort(s)[::-1][k - 1]
    above = s > t; tie = s == t; w[above] = 1.0; w[tie] = (k - above.sum()) / tie.sum()
    return w

class RuleBoot:
    """A ranking rule on one split under bootstrap multiplicities W (B x N): for target answer counts k_b it returns the rule's
    expected right count when it answers the k_b highest-scoring valid questions of replicate b (boundary ties split equally),
    so coverage stays matched inside every replicate."""
    def __init__(self, score, valid, labels, W):
        idx = np.where(valid)[0]; s = score[idx]
        lev, inv = np.unique(-s, return_inverse=True); G = len(lev); self.G = G
        oh = np.zeros((len(idx), G), np.float32); oh[np.arange(len(idx)), inv] = 1.0
        Wv = W[:, idx]; self.n = Wv @ oh; self.ncum = np.cumsum(self.n, 1)
        self.r = {k: Wv @ (oh * y[idx][:, None].astype(np.float32)) for k, y in labels.items()}
        self.rcum = {k: np.cumsum(v, 1) for k, v in self.r.items()}; self.ar = np.arange(W.shape[0])
    def right(self, kb, lab):
        g = np.minimum((self.ncum < kb[:, None] - 1e-6).sum(1), self.G - 1); ar = self.ar
        pn = np.where(g > 0, self.ncum[ar, g - 1], 0.0); prr = np.where(g > 0, self.rcum[lab][ar, g - 1], 0.0)
        ng = self.n[ar, g]; rg = self.r[lab][ar, g]; take = np.clip(kb - pn, 0, ng)
        return prr + np.where(ng > 0, take * rg / np.maximum(ng, 1e-9), 0.0), pn + take

def rule_scores(Q, PR):
    S = PR["S"]
    return dict(SC=Q["conf"].astype(float), SCtb=Q["conf"] + 1e-4 * sig(S["pik"]), PIK=S["pik"], PIKA=S["pikA"], GF=S["sure"],
                COMBO=S["combo"], ORACLE=Q["gaud"].astype(float) + 1e-6 * Q["conf"])

def grp(tag, fam, credit):
    """recipe x grader family label"""
    t = {"FK20": "", "FK00": "noKL:", "FK50": "KL50:", "FK100": "KL100:", "B1K20": "1visit:", "B2K20": "2visit:", "B2vK20": "2visit:",
         "B5vK20": "5visit:", "B10vK20": "10visit:", "LK20": "LK20:", "LK50": "LK50:"}[tag]
    c = "" if credit == "idkc05" else f"[{credit}]"
    return f"{t}{fam}{c}"

# grader families grouped as in the task
GROUPS = [("key", ["key"]), ("key+noise", ["keyFN", "keyFP", "keyMix", "keyHash", "keyLenQ", "keyLong"]), ("own vote", ["vote"]),
          ("judges", ["JQ_t50", "JQ_a50", "JQ_a35", "JM_a35", "JG_a35", "JO_a35", "JR_a35", "JG_t50", "JO_t50", "JM_t50", "JQ_t50f1", "JQ_t50q"]),
          ("Sonnet", ["sonnet"]), ("planted", ["pl_fnType", "pl_fnHalf", "pl_real", "pl_scr", "pl_sharp", "pl_typeFP", "pl_popFP", "pl_blind"]),
          ("other frozen keys", ["frzOR", "peer", "jveto"])]
FAMORDER = [f for _, fs in GROUPS for f in fs]

def boot_weights(n, B, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float32)

def ci(v): return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
