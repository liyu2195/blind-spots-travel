"""Phase 4: files for planted false rejects by question set (PopQA). Usage: python3 build_fn_files.py
pqa_knows_hid.jsonl: the "hidden" set = within every question type, the half of the TRAINED questions with the smaller hash value hunit("hid", idx)
(ties impossible; odd counts: floor). Nothing in the question tells which half it is in. hid_halves.json: the same split for all 1,500 questions
(trained and new separately), used only by the scorer."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, collections
sys.path.insert(0, _RC)
from st_grpo import hunit
D = (_RD + "/")
Q = [json.loads(l) for l in open(D + "pqa_1500.jsonl")]; H1 = {json.loads(l)["idx"] for l in open(D + "pqa_h1.jsonl")}
half = {}
for part in (True, False):
    by = collections.defaultdict(list)
    for q in Q:
        if (q["idx"] in H1) == part: by[q["prop"]].append(q["idx"])
    for t, ids in by.items():
        ids = sorted(ids, key=lambda i: hunit("hid", i))
        for k, i in enumerate(ids): half[i] = k < len(ids) // 2
with open(D + "pqa_knows_hid.jsonl", "w") as f:
    for q in Q:
        if q["idx"] in H1: f.write(json.dumps({"idx": q["idx"], "knows": bool(half[q["idx"]])}) + "\n")
json.dump({str(i): bool(v) for i, v in half.items()}, open((_RX + "/final/hid_halves.json"), "w"))
B = {json.loads(l)["idx"]: json.loads(l)["knows"] for l in open(D + "pqa_knows_type.jsonl")}
tr = [i for i in half if i in H1]; nw = [i for i in half if i not in H1]
print("trained: hidden-high", sum(half[i] for i in tr), "of", len(tr), "| of group B:", sum(half[i] for i in tr if B[i]), "of", sum(B[i] for i in tr), "| new: pseudo-high", sum(half[i] for i in nw), "of", len(nw))
Qw = "STUDENT=Qwen/Qwen2.5-7B-Instruct TAG=qwen7"; Ll = "STUDENT=gs://YOUR_BUCKET/acl_st/models/meta-llama/Llama-3.1-8B-Instruct TAG=llama8"; n = 0
ARMS = [("fnB80", 0.8, 0.2, "pqa_knows_type.jsonl", (0, 1, 2)), ("fnA80", 0.2, 0.8, "pqa_knows_type.jsonl", (0, 1, 2)), ("fnH80", 0.8, 0.2, "pqa_knows_hid.jsonl", (0, 1, 2)), ("fnB50", 0.5, 0.2, "pqa_knows_type.jsonl", (0, 1))]
for s in (0, 1, 2):
    for arm, fk, fu, kf, seeds in ARMS:
        if s not in seeds: continue
        for st, S in (("qwen7", Qw), ("llama8", Ll)):
            out = f"{st}_FK20{arm}_idkc05_s{s}_pq"
            open(f"{_RT}/queues/p6f/000b{n:02d}_{out}", "w").write(f"{S} ADV=std IDK=0.5 FPMODE=random FMT=1 KLB=0.2 SEED={s} FPK=0.0 FPU=0.0 FN=0.0 FNK={fk} FNU={fu} KNOWSF={kf} MICRO=16 GENCHUNK=32 CAP=21600 TRAINF=pqa_h1.jsonl EVALF=pqa_1500.jsonl OUTNAME={out}\n"); n += 1
print(n, "configs in work/q_p6f")
