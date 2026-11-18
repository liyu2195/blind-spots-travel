"""Build the prompt set of the open-ended refusal task from two public benchmarks (no access token needed):
  OR-Bench (bench-llm/or-bench, cc-by-4.0): or-bench-80k.csv (benign prompts that look sensitive, 10 categories), or-bench-hard-1k.csv (the hard
     subset), or-bench-toxic.csv (harmful prompts);
  XSTest (Paul/XSTest, cc-by-4.0): xstest_prompts.csv (250 safe prompts of 10 types, 200 unsafe contrast prompts).
Rows {idx, question, label (benign | harmful), cat, src, split}:
  train : PER benign prompts per OR-Bench category (not in the hard subset) + NTOX harmful prompts
  new   : PER other benign prompts per category + NTOX other harmful prompts          (held-out, same distribution)
  probe : every XSTest prompt, and NHARD prompts of the hard subset                   (never trained)
Fixed seed; prompts longer than 400 characters and duplicates are dropped before sampling.
Usage: python st_sf_build.py --out sfa_v1.jsonl [--per 60] [--ntox 250] [--nhard 300] [--seed 20261117]
"""
import argparse, csv, json, random, sys
from collections import Counter


def read(repo, fn):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, fn, repo_type="dataset")
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    print(f"{repo}/{fn}: {len(rows)} rows, columns {list(rows[0].keys()) if rows else []}", flush=True)
    return rows


def col(rows, names):
    keys = {k.lower().strip(): k for k in rows[0].keys()}
    for n in names:
        if n in keys:
            return keys[n]
    raise SystemExit(f"none of the columns {names} found in {list(rows[0].keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--ntox", type=int, default=250)
    ap.add_argument("--nhard", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    clean = lambda s: " ".join(str(s).split())
    big = read("bench-llm/or-bench", "or-bench-80k.csv"); hard = read("bench-llm/or-bench", "or-bench-hard-1k.csv")
    tox = read("bench-llm/or-bench", "or-bench-toxic.csv"); xs = read("Paul/XSTest", "xstest_prompts.csv")
    pk, ck = col(big, ["prompt"]), col(big, ["category"])
    hp, hc = col(hard, ["prompt"]), col(hard, ["category"]); tp, tc = col(tox, ["prompt"]), col(tox, ["category"])
    xp, xt, xl = col(xs, ["prompt"]), col(xs, ["type"]), col(xs, ["label"])
    hardset = {clean(r[hp]).lower() for r in hard}; xsset = {clean(r[xp]).lower() for r in xs}
    out = []; seen = set()

    def add(q, label, cat, src, split):
        k = clean(q).lower()
        if not k or k in seen:
            return False
        seen.add(k); out.append(dict(idx=len(out), question=clean(q), label=label, cat=str(cat).strip(), src=src, split=split)); return True

    for r in xs:                                                     # probe 1: XSTest, all prompts
        lab = "benign" if str(r[xl]).strip().lower() == "safe" else "harmful"
        add(r[xp], lab, r[xt], "xstest", "probe")
    bycat = {}
    for r in big:
        q = clean(r[pk])
        if 0 < len(q) <= 400 and q.lower() not in hardset and q.lower() not in xsset:
            bycat.setdefault(str(r[ck]).strip(), []).append(q)
    for c in sorted(bycat):                                           # train / new: benign prompts by category
        L = sorted(set(bycat[c])); rng.shuffle(L); n = 0
        for q in L:
            if n >= 2 * a.per:
                break
            n += add(q, "benign", c, "orb80k", "train" if n < a.per else "new")
        print(f"category {c}: pool {len(L)}, taken {n}", flush=True)
    T = [(clean(r[tp]), str(r[tc]).strip()) for r in tox if 0 < len(clean(r[tp])) <= 400]; T = sorted(set(T)); rng.shuffle(T); n = 0
    for q, c in T:                                                    # train / new: harmful prompts
        if n >= 2 * a.ntox:
            break
        n += add(q, "harmful", c, "orbtoxic", "train" if n < a.ntox else "new")
    Hh = [(clean(r[hp]), str(r[hc]).strip()) for r in hard if 0 < len(clean(r[hp])) <= 400]; Hh = sorted(set(Hh)); rng.shuffle(Hh); n = 0
    for q, c in Hh:                                                   # probe 2: the hard subset
        if n >= a.nhard:
            break
        n += add(q, "benign", c, "orbhard", "probe")
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cnt = Counter((r["split"], r["src"], r["label"]) for r in out)
    print(f"wrote {len(out)} prompts to {a.out}: " + "; ".join(f"{k[0]}/{k[1]}/{k[2]} {v}" for k, v in sorted(cnt.items())), flush=True)
    print("categories (train, benign): " + ", ".join(f"{c} {v}" for c, v in sorted(Counter(r['cat'] for r in out if r['split'] == 'train' and r['label'] == 'benign').items())), flush=True)
    print("xstest types: " + ", ".join(f"{c} {v}" for c, v in sorted(Counter(r['cat'] for r in out if r['src'] == 'xstest').items())), flush=True)


if __name__ == "__main__":
    main()
