"""(dose) Item 3: place the capitals probe items and the answer key's missing aliases on the dose-response curve.
CAPITALS (never trained; adapters of PopQA FK20 runs evaluated on the probes): pqq = hand-written former-state (F) / present-day (P) capitals;
  pqp = PopQA-derived probe (capital questions F / P / A = other, religion questions R). Items = known-sure answers of the untrained student
  (valid, audited right = loose key or Sonnet >= 4, >= 31/32 samples agree). No student hidden states exist for the probe questions
  (base/<st>_pqq|pqp_1500_hidden.npz are absent), so the readable predictors here are the question-text kinds fitted on PopQA TRAINING verdicts of
  the strict Qwen judge (Q_a35): tfidf and knnE (d03). Actual verdict = Qwen judge at the PopQA a35 cut of the student on the greedy answer (v_bin).
  Curves are fitted on PopQA NEW known-sure answers for the judgeQa35 runs (6 seeds per student): logistic regression of refusal on
  [conf, logit P_aud(kind), logit P_reject(kind)] ('readable'), on [conf, v_bin] ('verdict'), or on both ('joint'); each capitals group is then
  compared with the curve's prediction at its own covariates (observed minus predicted refusal; bootstrap over items and runs, 1000 draws).
  Gold-key runs (sim00fn00) give the no-judge baseline on the same items.
MISSING ALIASES (natural, unthresholded false rejections of the answer key): key-trained runs (sim00fn00), every dataset and student, known-sure
  answers; 'missing' = Sonnet >= 4 but the loose key rejects; 'accepted' = the strict key (training reward) accepts. Curve = logistic regression
  of the key runs' refusal on [G (d04_lib.gen_cov lin16), logit P_key-rejects (lin16; out-of-fold on HOME)] fitted on all known-sure answers of the
  part; the missing-alias group is compared with its prediction, on NEW and on HOME questions; also with the actual-verdict curve.
  Placebo: judge-trained (Q_a35) and Sonnet-trained runs on the same items (they never saw the key).
Output: d05_out.txt, d05_summary.pkl.  Usage: nice -n 10 timeout 280 python3 d05_place.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, json, numpy as np, warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
sys.path.insert(0, (_RA + "/dose"))
from d04_lib import *
P = Log(f"{HERE}/d05_out.txt"); P(__doc__)
rng = np.random.default_rng(505); NB = 1000
CAP = pickle.load(open(f"{CACHE}/cap.pkl", "rb"))
def fitcurve(X, Ybar, n):
    """binomial logistic fit of question-level refusal shares Ybar (n runs each) on X; returns predict(X2) -> probability."""
    Xs = np.vstack([X, X]); ys = np.r_[np.ones(len(X)), np.zeros(len(X))]; ws = np.r_[Ybar, 1 - Ybar] * n
    m = LogisticRegression(C=1e4, max_iter=5000).fit(Xs, ys, sample_weight=ws + 1e-9)
    return lambda X2: m.predict_proba(X2)[:, 1]
SUM = {}
P("\n==== CAPITALS")
for st in STS:
    D = load_cell(st, "pq", kinds=("lin16", "tfidf", "knnE")); B = D["B"]; Un = D["Unew"]
    t35 = TH["Q"][f"{st}_pqa"]["t_a35"]
    jr = list(D["arms"]["JQ_a35"]["runs"])
    Rk = pickle.load(open(f"{CACHE}/runs_{st}_pq.pkl", "rb"))
    gold = [(r["o"] == 0).astype(float) for n, r in Rk.items() if r["g"] == "sim00fn00" and r["c"] == "05"]
    Yj = np.mean([r["y"][Un] for r in jr], 0); Yg = np.mean([g[Un] for g in gold], 0)
    vb_new = (B["PJ"]["Q"][Un, 0] <= t35).astype(float)
    # PopQA new known-sure: curves for the judgeQa35 runs
    feats = {}
    for k in ("tfidf", "knnE"):
        PRk = D["PR"][k]
        feats[k] = np.column_stack([B["conf"][Un], logit(PRk["aud"]["p"][Un]), logit(PRk["Q_a35"]["p"][Un])])
    curves = {}
    for k in ("tfidf", "knnE"):
        curves[("read", k)] = fitcurve(feats[k], Yj, len(jr)); curves[("joint", k)] = fitcurve(np.column_stack([feats[k], vb_new]), Yj, len(jr))
        curves[("gold_read", k)] = fitcurve(feats[k], Yg, len(gold))
    curves["verd"] = fitcurve(np.column_stack([B["conf"][Un], vb_new]), Yj, len(jr))
    # binned PopQA reference curve on P_reject (tfidf): quintiles -> mean P and refusal, for display
    for k in ("tfidf", "knnE"):
        p = D["PR"][k]["Q_a35"]["p"][Un]; q = np.quantile(p, [0.2, 0.4, 0.6, 0.8]); b = np.digitize(p, q)
        P(f"  {st} PopQA new known-sure ({len(Un)}), judgeQa35 runs {len(jr)}, {k} curve: " + "; ".join(f"P {p[b == j].mean():.2f} -> refusal {Yj[b == j].mean():.2f}" for j in range(5))
          + f" | by actual verdict: rejected {Yj[vb_new == 1].mean():.2f} (n {int(vb_new.sum())}), accepted {Yj[vb_new == 0].mean():.2f}; gold-key runs {Yg[vb_new == 1].mean():.2f} / {Yg[vb_new == 0].mean():.2f}")
    for s in ("pqq", "pqp"):
        c = CAP[(s, st)]; aud = c["valid"][:, 0] & (c["kl"][:, 0] | (c["son"] >= 4)); ks = aud & (c["conf"] >= 31 / 32)
        RJ = [r for r in c["runs"].values() if r["g"] == "judgeQa35" and r["c"] == "05"]; RG = [r for r in c["runs"].values() if r["g"] == "sim00fn00" and r["c"] == "05"]
        YJ = np.array([(r["o"] == 0).astype(float) for r in RJ]); YG = np.array([(r["o"] == 0).astype(float) for r in RG])
        vb = (c["PJ"]["Q"][:, 0] <= t35).astype(float)
        pk = {}
        for k in ("tfidf", "knnE"):
            PRk = D["PR"][k]; pk[k] = (PRk["Q_a35"]["cap"][s], PRk["aud"]["cap"][s])
        groups = [("F", (c["cls"] == "F") & (c["prop"] == "capital")), ("P", (c["cls"] == "P") & (c["prop"] == "capital"))]
        if s == "pqp": groups += [("A", (c["cls"] == "A") & (c["prop"] == "capital")), ("R", c["prop"] == "religion")]
        for gname, gm in groups:
            for vsub, vm in (("all", np.ones(len(vb), bool)), ("judge rejects", vb == 1), ("judge accepts", vb == 0)):
                m = np.where(gm & ks & vm)[0]
                if len(m) < 5: continue
                row = dict(n=len(m), rej=vb[m].mean(), yj=YJ[:, m].mean(), yg=YG[:, m].mean() if len(YG) else np.nan)
                preds = {}
                for k in ("tfidf", "knnE"):
                    Xc = np.column_stack([c["conf"][m], logit(pk[k][1][m]), logit(pk[k][0][m])]); row[f"p_{k}"] = pk[k][0][m].mean()
                    preds[f"read_{k}"] = curves[("read", k)](Xc); preds[f"joint_{k}"] = curves[("joint", k)](np.column_stack([Xc, vb[m]]))
                    preds[f"goldread_{k}"] = curves[("gold_read", k)](Xc)
                preds["verd"] = curves["verd"](np.column_stack([c["conf"][m], vb[m]]))
                # bootstrap: items within the group, runs within arm (curves held fixed: their own uncertainty is small next to the group's)
                bs = {nm: [] for nm in preds}; bobs = []
                for b in range(NB):
                    ii = rng.integers(0, len(m), len(m)); rr = rng.integers(0, len(YJ), len(YJ)); o = YJ[rr][:, m[ii]].mean(); bobs.append(o)
                    for nm, pr in preds.items(): bs[nm].append(o - pr[ii].mean())
                row["obs_ci"] = pct(bobs)
                for nm, pr in preds.items(): row[f"pred_{nm}"] = pr.mean(); row[f"gap_{nm}"] = (row["yj"] - pr.mean(), *pct(bs[nm]))
                SUM[("cap", st, s, gname, vsub)] = row
                P(f"  {st} {s} {gname} [{vsub:13s}] n {row['n']:3d} | judge rejects {row['rej']:.2f} | P_reject tfidf {row['p_tfidf']:.2f} knnE {row['p_knnE']:.2f} | "
                  f"refusal judgeQa35 {row['yj']:.2f} [{row['obs_ci'][0]:.2f}, {row['obs_ci'][1]:.2f}] gold {row['yg']:.2f} | predicted by curve: "
                  + " ".join(f"{nm} {row['pred_' + nm]:.2f} (obs-pred {row['gap_' + nm][0]:+.2f} [{row['gap_' + nm][1]:+.2f}, {row['gap_' + nm][2]:+.2f}])"
                             for nm in ("read_tfidf", "read_knnE", "verd", "joint_tfidf")))

P("\n==== MISSING ALIASES of the answer key (key-trained runs; known-sure answers)")
for st in STS:
    for ds in DSS:
        D = load_cell(st, ds, kinds=("lin16",)); B = D["B"]
        if "key" not in D["arms"]: continue
        RK = D["arms"]["key"]["runs"]; RJ = D["arms"].get("JQ_a35", {"runs": []})["runs"]; RS = D["arms"].get("son", {"runs": []})["runs"]
        for pt, U, home in (("new", D["Unew"], False), ("home", D["Uhome"], True)):
            G = gen_cov(D, "lin16", U, home); x = logit(D["PR"]["lin16"]["key"]["oof" if home else "p"][U])
            vkey = (~B["ks"][U, 0]).astype(float); miss = (~B["kl"][U, 0]) & (B["son"][U, 0] >= 4); acc = B["ks"][U, 0]
            if miss.sum() < 5: continue
            Yk = np.mean([r["y"][U] for r in RK], 0)
            cr = fitcurve(np.column_stack([G, x]), Yk, len(RK)); cv = fitcurve(np.column_stack([G, vkey]), Yk, len(RK))
            pr = cr(np.column_stack([G, x])); pv = cv(np.column_stack([G, vkey]))
            row = dict(n=int(miss.sum()), n_acc=int(acc.sum()), p_miss=float(sig(x[miss]).mean()), p_acc=float(sig(x[acc]).mean()),
                       yk_miss=Yk[miss].mean(), yk_acc=Yk[acc].mean(), pred_read=pr[miss].mean(), pred_verd=pv[miss].mean(),
                       yj_miss=np.mean([r["y"][U][miss].mean() for r in RJ]) if RJ else np.nan, yj_acc=np.mean([r["y"][U][acc].mean() for r in RJ]) if RJ else np.nan,
                       ys_miss=np.mean([r["y"][U][miss].mean() for r in RS]) if RS else np.nan, ys_acc=np.mean([r["y"][U][acc].mean() for r in RS]) if RS else np.nan, runs=len(RK))
            mi = np.where(miss)[0]; bs = []
            YK = np.array([r["y"][U] for r in RK])
            for b in range(NB):
                ii = rng.integers(0, len(mi), len(mi)); rr = rng.integers(0, len(YK), len(YK)); o = YK[rr][:, mi[ii]].mean(); bs.append((o - pr[mi[ii]].mean(), o - pv[mi[ii]].mean()))
            bs = np.array(bs); row["gap_read"] = (row["yk_miss"] - row["pred_read"], *pct(bs[:, 0])); row["gap_verd"] = (row["yk_miss"] - row["pred_verd"], *pct(bs[:, 1]))
            SUM[("alias", st, ds, pt)] = row
            P(f"  {st} {ds} {pt:4s}: key runs {row['runs']} | missing-alias items {row['n']:3d} (accepted {row['n_acc']}) | predicted P(key rejects) missing {row['p_miss']:.2f} vs accepted {row['p_acc']:.2f} | "
              f"key-run refusal missing {row['yk_miss']:.2f} vs accepted {row['yk_acc']:.2f}; readable-curve prediction for missing {row['pred_read']:.2f} (obs-pred {row['gap_read'][0]:+.2f} "
              f"[{row['gap_read'][1]:+.2f}, {row['gap_read'][2]:+.2f}]); verdict-curve prediction {row['pred_verd']:.2f} | placebo judgeQa35 {row['yj_miss']:.2f}/{row['yj_acc']:.2f}, Sonnet {row['ys_miss']:.2f}/{row['ys_acc']:.2f}")
    for pt in ("new", "home"):
        L = [v for k, v in SUM.items() if k[0] == "alias" and k[1] == st and k[3] == pt]
        if L: P(f"  {st} {pt} mean over datasets: obs-pred (readable curve) {np.mean([v['gap_read'][0] for v in L]):+.3f}; key-run refusal missing {np.mean([v['yk_miss'] for v in L]):.3f} vs accepted {np.mean([v['yk_acc'] for v in L]):.3f}; "
                f"judge placebo {np.nanmean([v['yj_miss'] for v in L]):.3f} vs {np.nanmean([v['yj_acc'] for v in L]):.3f}")
pickle.dump(SUM, open(f"{HERE}/d05_summary.pkl", "wb")); P.save()
