# Check F1: does "answers stay the model's own" survive removing the KL anchor?
# Compares FK00 (no KL) runs with their FK20 (KL 0.2) counterparts on TriviaQA (h1), same graders.
# Per run, on NEW questions (not in tqa_h1.jsonl, qid-deduplicated) and HOME questions:
#   idk = share of final greedy answers that say "I don't know"
#   same = among non-IDK answers, share whose normalised answer equals the untrained greedy's normalised answer
#   acc = loose key accuracy among answered (alias contained as whole words, or equal)
#   kr_same = among key-right answers, share equal to the untrained greedy
#   ntok = mean answer tokens among non-IDK answers; kl_last = mean kl over the last 10 logged steps
# Output: f1_nokl_out.txt
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, re, glob, os, sys, ast

B = _RT
IDK = re.compile(r"i\s+(do\s*n[o']?t|don['’]t|do not)\s+know", re.I)

def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f]

def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def right(ans, aliases):
    a = ' ' + norm(ans) + ' '
    for al in aliases:
        al = norm(al)
        if al and (' ' + al + ' ') in a:
            return True
    return False

data = {d['idx']: d for d in load(f'{_RD}/tqa_1500.jsonl')}
train = load(f'{_RD}/tqa_h1.jsonl')
train_idx = {d['idx'] for d in train}
train_qid = {d['qid'] for d in train}
# consistency: training idx point to the same qid in the 1500 file
mism = sum(1 for d in train if data.get(d['idx'], {}).get('qid') != d['qid'])
new_idx = [i for i, d in data.items() if i not in train_idx and d['qid'] not in train_qid]
home_idx = [i for i in data if i in train_idx]
alias = {i: ast.literal_eval(d['aliases']) if isinstance(d['aliases'], str) else d['aliases'] for i, d in data.items()}

out = open((_RA + '/rev_F1/f1_nokl_out.txt'), 'w')
def P(*a):
    print(*a); print(*a, file=out)

P(f'train/1500 idx-qid mismatches {mism}; new (dedup) {len(new_idx)}; home {len(home_idx)}')

def greedy(rows):
    return {r['idx']: r for r in rows if str(r.get('s')) == '-1'}

for st in ['qwen7', 'llama8']:
    base = greedy(load(f'{_RR}/run1/base/{st}_tqa_1500_n32_s1.jsonl'))
    P(f'\n== {st}: untrained greedy new: idk {sum(bool(IDK.search(base[i]["text"])) for i in new_idx)/len(new_idx):.3f}')
    for g in ['judgeQt50', 'sim35fn00']:
        for tag in ['FK00', 'FK20']:
            runs = sorted(glob.glob(f'{_RR}/run1/rl/{st}_{tag}{g}_idkc05_s*_h1'))
            for rd in runs:
                fp = f'{rd}/eval_rl_ck120.jsonl'
                if not os.path.exists(fp):
                    continue
                ev = greedy(load(fp))
                log = load(f'{rd}/train_log.jsonl')
                kls = [r.get('kl') for r in log[-10:] if r.get('kl') is not None]
                kl_last = sum(kls) / len(kls) if kls else float('nan')
                res = []
                for part, ids in [('new', new_idx), ('home', home_idx)]:
                    n = idk = ans = same = acc = kr = krs = 0; nt = 0
                    for i in ids:
                        if i not in ev or i not in base:
                            continue
                        n += 1
                        t = ev[i]['text']
                        if IDK.search(t):
                            idk += 1; continue
                        ans += 1
                        nt += int(ev[i].get('ntok', 0))
                        s_ = norm(ev[i]['ans']) == norm(base[i]['ans'])
                        same += s_
                        r_ = right(ev[i]['ans'], alias[i])
                        acc += r_
                        if r_:
                            kr += 1; krs += s_
                    res.append(f'{part}: idk {idk/n:.3f} same {same/max(ans,1):.3f} acc|ans {acc/max(ans,1):.3f} kr_same {krs/max(kr,1):.3f} ntok {nt/max(ans,1):.1f}')
                P(f'  {os.path.basename(rd):40s} kl_last {kl_last:.4f} | ' + ' | '.join(res))
out.close()
