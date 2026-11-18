"""(explorer 'general') Shared helpers: run-name parsing, config lookup, answer reading for the small students.
Read-only on project data; caches go to this folder."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, os, re, glob, sys
from collections import defaultdict
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, base32, same, C, RL, BASE, DATA   # noqa: F401  (IDK = full-match regex, as IDK_FULL)
HERE = (_RA + "/general")
SMALL = ("phi4m", "smol3", "granite")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}

# ---------------------------------------------------------------- configs (every file that names an OUTNAME)
CFG_SRC = (sorted(glob.glob((_RT + "/queues/*"))) + sorted(glob.glob((_RT + "/queues/*/"))) +
           [(_RT + "/queue_sim"), (_RT + "/queue_hyst"), (_RT + "/queue_sim2")] +
           sorted(glob.glob((_RT + "/ctl_w*"))) + [(_RT + "/queues/control")])
KV = re.compile(r'([A-Z][A-Z0-9_]*)=("[^"]*"|\S*)')


def load_configs():
    out = defaultdict(list)
    for src in CFG_SRC:
        files = [src] if os.path.isfile(src) else sorted(glob.glob(os.path.join(src, "*")))
        for f in files:
            if os.path.isdir(f):
                continue
            try:
                txt = open(f, errors="ignore").read()
            except Exception:
                continue
            if "OUTNAME=" not in txt:
                continue
            d = {k: v.strip('"') for k, v in KV.findall(txt)}
            if "OUTNAME" in d:
                out[d["OUTNAME"]].append((os.path.relpath(f, (_RX + "/home")), d))
    return out


# ---------------------------------------------------------------- run names
NAME = re.compile(r"^(?P<st>phi4m|smol3|granite)_(?P<body>.+?)_(?P<ds>h1|pq|nq)$")
BODY = re.compile(r"^(?P<pre>FK\d\dD?|FL2|FT1|F)?(?P<g>.+?)(?:_(?P<cr>idkc\d+|idk|flagc05))?(?:_s(?P<seed>\d+))?(?P<t240>_T240)?$")


def parse_run(name, cfgs=None):
    m = NAME.match(name)
    if not m:
        return None
    st, body, ds = m.group("st"), m.group("body"), m.group("ds")
    b = BODY.match(body)
    pre, g, cr, seed, t240 = b.group("pre"), b.group("g"), b.group("cr"), b.group("seed"), b.group("t240")
    # a body such as 'Fanti6' starts with F; 'FjudgeQt80x240' etc. -> prefix F, grader judgeQt80x240
    r = dict(run=name, st=st, ds=DSN[ds], dtag=ds, pre=pre or "", grader=g, cr=cr, seed=int(seed or 0), t240=bool(t240))
    r["fmt"] = 1 if pre else 0
    r["kl"] = {"FK04": 0.04, "FK10": 0.1, "FK20": 0.2, "FK20D": 0.2, "FK40": 0.4}.get(pre or "", 0.0)
    r["adv"] = "none" if (pre == "FK20D" or g.endswith("dr")) else "std"
    gg = g[:-2] if g.endswith("dr") else g
    r["temp"] = 1.0 if pre == "FT1" else 0.8
    r["lr"] = 2e-5 if pre == "FL2" else 5e-5
    r["steps"] = 240 if (pre == "FL2" or t240 or "x240" in g) else 120
    r["outlet"] = "idk" if (cr and cr.startswith("idk")) else ("flag" if cr == "flagc05" else "none")
    if cr is None:
        r["credit"] = None
    elif cr in ("idk", "flagc05"):
        r["credit"] = 0.5
    else:
        v = cr[4:]
        r["credit"] = {"05": 0.5, "01": 0.1, "03": 0.3, "04": 0.4, "06": 0.6, "07": 0.7, "08": 0.8, "09": 0.9,
                       "250": 0.25, "375": 0.375, "625": 0.625, "75": 0.75}.get(v, None)
    # grader family
    fam, fp, fn, jm, jmode, thr = "other", None, None, None, None, None
    mm = re.fullmatch(r"sim(\d\d)fn(\d\d)", gg)
    if gg in ("gold", "goldstrict"):
        fam, fp, fn = "key", 0.0, 0.0
    elif mm:
        fp, fn = int(mm.group(1)) / 100, int(mm.group(2)) / 100
        fam = "key" if fp == 0 and fn == 0 else "sim"
    elif re.fullmatch(r"hash(\d\d)fn(\d\d)", gg) or re.fullmatch(r"hash(\d\d)", gg):
        mh = re.fullmatch(r"hash(\d\d)(?:fn(\d\d))?", gg); fam = "hash"; fp = int(mh.group(1)) / 100; fn = int(mh.group(2) or 0) / 100
    elif re.fullmatch(r"det(\d\d)", gg):
        fam = "det"; fp = int(gg[3:]) / 100; fn = 0.0
    elif gg.startswith("judge"):
        fam = "judge"
        mj = re.fullmatch(r"judge(Q2b|Q4b|Q|G)(ref|own)?(?:t(\d\d))?(x240)?", gg)
        if mj:
            jm = {"Q": "Q9", "G": "G12", "Q2b": "Q2", "Q4b": "Q4"}[mj.group(1)]
            jmode = mj.group(2) or "free"
            thr = int(mj.group(3)) / 100 if mj.group(3) else None
    elif gg.startswith(("synthU", "synthC")):
        fam = "synth"
    elif gg.startswith(("blind", "antiB", "confB", "infoB", "info", "selfk", "randkey", "antikey", "rand65", "anti6", "self3")):
        fam = "selfinfo"
    elif gg == "major":
        fam = "major"
    elif gg.startswith("relay"):
        fam = "sonnet"
    elif gg.startswith(("sw", "force")):
        fam = "hyst"
    r.update(fam=fam, fp=fp, fn=fn, jm=jm, jmode=jmode, thr=thr, g0=gg)
    if cfgs is not None and name in cfgs:
        srcs = cfgs[name]
        d = {}
        for _, dd in srcs:
            d.update(dd)
        r["cfg_src"] = ";".join(s for s, _ in srcs)
        r["cfg"] = d
        if "KLB" in d:
            r["kl_cfg"] = float(d["KLB"])
        if "ADV" in d:
            r["adv_cfg"] = d["ADV"]
        if "IDK" in d and d["IDK"] not in ("", '""'):
            r["credit_cfg"] = float(d["IDK"])
        if "FMT" in d:
            r["fmt_cfg"] = int(d["FMT"])
        if "JMODE" in d:
            r["jmode_cfg"] = d["JMODE"]
        if "JT" in d and d["JT"]:
            r["thr_cfg"] = float(d["JT"])
        if "STEPS" in d:
            r["steps_cfg"] = int(d["STEPS"])
        if "FPMODE" in d:
            r["fpmode_cfg"] = d["FPMODE"]
    return r


def final_ck(run):
    cks = []
    for f in glob.glob(f"{RL}/{run}/eval_rl_ck*.jsonl"):
        try:
            cks.append(int(re.search(r"ck(\d+)\.jsonl$", f).group(1)))
        except Exception:
            pass
    return max(cks) if cks else None


# ---------------------------------------------------------------- reading answers
def read_eval(run, ck, greedy_only=True):
    """rows of eval_rl_ck<ck>.jsonl: dict idx -> greedy row (and dict idx -> list of sample rows)."""
    g, s = {}, defaultdict(list)
    with open(f"{RL}/{run}/eval_rl_ck{ck}.jsonl") as f:
        for l in f:
            r = json.loads(l)
            if r["s"] < 0:
                g[r["idx"]] = r
            elif not greedy_only:
                s[r["idx"]].append(r)
    return g, s


def code_strict(r, aliases):
    """strict reading (rc.sa: one line, ended, no think tag; IDK full match) -> (code, x): 0 IDK, 1 right (loose key),
    2 wrong, 3 malformed"""
    x = sa(r)
    if not x:
        return 3, ""
    if IDK.match(x):
        return 0, x
    return (1 if C.tq_correct(x, aliases) else 2), x


def code_first(r, aliases):
    """first-line reading used by the free-format (old) trainer: parse_tq of the first line; IDK = st_common.is_idk prefix."""
    x = C.parse_tq(r["text"])
    if not x:
        return 3, ""
    if C.is_idk(x):
        return 0, x
    return (1 if C.tq_correct(x, aliases) else 2), x


def pyes(r):
    s = r["py"] + r["pn"]
    return r["py"] / s if s > 1e-3 else 0.5
