"""(bothsigns) b02: two effect measures, each computed with ONE estimator for every grader arm (both error signs and answer form).
Items = (question, untrained greedy answer) pairs; home = training questions, new = held-out questions (duplicates of training questions removed).
Populations: RS = greedy answer strict-key right (right by every standard); AW = greedy answer audited wrong (key-wrong, Sonnet level < 4);
  KM = greedy answer key-wrong but Sonnet level >= 4 (the key's missing aliases).
Affected items (the grader errs on the greedy answer): FR arms: RS items it rejects; FA arms: AW items it pays (hash: hunit('fp', seed, idx, a) <
  0.35; type / popularity sets: question in the paid set; judge J: P_J(Yes) > the run's threshold; long80: answer has >= 5 words); key misses: KM
  items (vs RS); placebo: a fixed pseudo-random half of RS (FR) or AW (FA).
E_dec (decision currency, FR and FA alike): refusal rate on affected items minus the refusal rate predicted from the run's refusal on the
  unaffected items of the same population (fit over reference runs of the same student and dataset; primary fit = OLS of empirical logits,
  robustness = linear OLS and mean of the 6 reference runs nearest in the matching rate); sign +1 for FR (extra refusal), -1 for
  FA (extra answering). = a difference-in-differences against reference runs at matched refusal.
E_kind (kind currency, all arms incl. form): share of the run's error-type outputs on the population that are of the kind the grader's error
  rewards, minus the share predicted from the run's rate of such outputs (OLS over reference runs). FR: share of refusals of RS questions that
  fall on rejected items; FA: share of key-wrong final answers on AW questions that the grader pays (hash table / judge P(Yes) of the final
  answer / question in the paid set / >= 5 words for long80). Net of how many refusals or wrong answers there are (leniency).
E_kindc (FA / form only): the same share computed among CHANGED wrong answers only (final answer differs from the untrained greedy answer):
  the content channel (which string), without the decision channel (where to answer).
Reference pool = FK20 key-family graders with item-independent errors (simXXfnYY random mode, lenQ: 27-45 runs per cell); key misses use runs
  trained with LLM judges or Sonnet; placebos use the pool minus their own noise family. Runs (arm or reference) refusing >= 95% of a part's
  questions (collapsed) are dropped; arm runs whose matching rate lies outside the reference range +-0.05 (or >= 0.95) or with < 10 outputs of the
  type are dropped (counted).
Intervals: crossed bootstrap (Poisson question weights within part, arm runs within arm and cell, reference runs within cell), NB draws.
Usage: NB=500 nice -n 10 python3 b02_effects.py  -> cache/b02_boot.pkl (then b02b_table.py)"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, pickle, time, json
from collections import defaultdict
import numpy as np
sys.path.insert(0, (_RA + "/bothsigns"))
from b00_common import *

NB = int(os.environ.get("NB", 500)); SEED = 7
TH = {J: json.load(open(f"{FIN}/thresholds_{J}.json")) for J in JUDGES}
SS = json.load(open(f"{FIN}/scramble_sets.json")); HH = json.load(open(f"{FIN}/hid_halves.json"))


def knows(f):
    return {r["idx"]: bool(r["knows"]) for r in jl(f"{DATA}/{f}")}


def sets_pq(Q, st):
    """question-level sets on all 1,500 PopQA questions (home from the knows files, new by the same rule)"""
    idx = Q["idx"]; n = len(idx); home = Q["part"] == 0; prop = Q["prop"]; pop = Q["pop"]
    S = {}
    kt = knows("pqa_knows_type.jsonl"); Bp = {prop[k] for k in range(n) if home[k] and kt[int(idx[k])]}
    S["typeB"] = np.array([p in Bp for p in prop]); S["typeA"] = ~S["typeB"]
    kp = knows("pqa_knows_pop.jsonl")
    lo = max(pop[k] for k in range(n) if home[k] and not kp[int(idx[k])]); hi = min(pop[k] for k in range(n) if home[k] and kp[int(idx[k])])
    S["popH"] = pop >= (lo + hi) / 2; S["popL"] = ~S["popH"]
    S["hid"] = np.array([HH[str(int(i))] for i in idx])
    for k in range(n):
        if home[k]:
            assert S["popH"][k] == kp[int(idx[k])] and S["typeB"][k] == kt[int(idx[k])]
    for J in "MO":
        ty = set(SS[f"{st}|sharp{J}"]["types"]); S[f"sharp{J}"] = np.array([p in ty for p in prop])
        RQ = SS[f"{st}|RQ"]; ks = knows(f"pqa_knows_scr{J}_{st}.jsonl"); kk = sum(ks.values())
        thr = sorted(hunit("scr", J, st, i) for i in RQ)[kk - 1]
        S[f"scr{J}"] = np.array([hunit("scr", J, st, int(i)) <= thr for i in idx])
        kr = knows(f"pqa_knows_real{J}_{st}.jsonl"); t35 = TH[J][f"{st}_pqa"]["t_a35"]; pj = Q["pj"][J]
        S[f"real{J}"] = np.array([kr[int(idx[k])] if home[k] else (np.isfinite(pj[k]) and not pj[k] > t35) for k in range(n)])
        S[f"real{J}_known"] = np.array([True if home[k] else bool(np.isfinite(pj[k])) for k in range(n)])
    return S


def build_units(st, ds):
    P = pickle.load(open(f"{OUT}/cache/cache_{st}_{ds}.pkl", "rb")); Q, R = P["Q"], P["R"]
    look = pickle.load(open(f"{OUT}/cache/jlook_{st}_{ds}.pkl", "rb"))
    idx = Q["idx"]; n = len(idx); part = Q["part"]
    names = sorted(R); code = np.array([R[r]["code"] for r in names]); IDKm = (code == 0).astype(np.float32)
    WR = (code == 3); WRf = WR.astype(np.float32)
    CHG = np.array([[R[u]["ans"][k] != Q["g"][k] for k in range(n)] for u in names])     # final answer differs from the untrained greedy answer
    LONG = np.array([R[r]["words"] for r in names]) >= LONG_WORDS
    row = {r: k for k, r in enumerate(names)}
    ref = [r for r in names if is_reference(R[r] | {"cfg": None})]
    RS = Q["gcode"] == 1; AW = (Q["gcode"] == 3) & (Q["gson"] < 4); KM = (Q["gcode"] == 3) & (Q["gson"] >= 4)
    pseudo = np.array([hunit("bothsigns", st, ds, int(i)) < 0.5 for i in idx])
    units = []

    def add(arm, sign, runs, Ad, Bd, Pk, KN, Nmask, KD, refs=None, lab=None, labpop=None, note=""):
        """decision: affected Ad vs unaffected Bd (bool over questions). kind: population Pk, numerator matrix KN restricted to Nmask,
        denominator matrix KD (rows = all runs). lab / labpop: readability label / population."""
        refs = ref if refs is None else refs
        runs = [r for r in runs if r in row]
        if runs:
            fa = not sign.startswith("FR")      # content-only variant (FA / form): the same share, among CHANGED wrong answers only
            units.append(dict(arm=arm, sign=sign, st=st, ds=ds, runs=runs, refs=refs, Ad=Ad, Bd=Bd, Pk=Pk, KN=KN, Nmask=Nmask, KD=KD,
                              KNc=(KN * CHG) if fa else None, KDc=(KD * CHG) if fa else None,
                              lab=lab, labpop=labpop, note=note, sgn=(1.0 if sign.startswith("FR") else -1.0)))

    by = defaultdict(list)
    for r in names:
        by[R[r]["g"]].append(r)
    # ---- hash: item-specific false accepts (paid table depends on the run's seed)
    for r in by.get("hash35fn00", []):
        s = R[r]["s"]
        paidK = np.array([[(hunit("fp", s, int(idx[k]), R[u]["ans"][k]) < 0.35) if WR[j, k] else False for k in range(n)] for j, u in enumerate(names)], np.float32)
        gpaid = np.array([hunit("fp", s, int(idx[k]), Q["g"][k]) < 0.35 for k in range(n)])
        add("hash35", "FA", [r], AW & gpaid, AW & ~gpaid, AW, paidK, AW, WRf, lab=gpaid, labpop=AW)
    if ds == "pq":
        S = sets_pq(Q, st)
        for g, sn in (("fnB80", "typeB"), ("fnB50", "typeB"), ("fnA80", "typeA"), ("fnH80", "hid"), ("fnH60", "hid")):
            add(g, "FR", by.get(g, []), RS & S[sn], RS & ~S[sn], RS, IDKm, RS & S[sn], IDKm, lab=S[sn], labpop=RS)
        for g, sn in (("typeA40fn20", "typeA"), ("typeB20fn20", "typeB"), ("typeB40fn20", "typeB"), ("typeB60fn20", "typeB"),
                      ("popH40fn20", "popH"), ("popL40fn20", "popL")):
            add(g.replace("fn20", ""), "FA", by.get(g, []), AW & S[sn], AW & ~S[sn], AW, WRf, AW & S[sn], WRf, lab=S[sn], labpop=AW)
        for g, sn in (("plrealM", "realM"), ("plrealO", "realO"), ("plscrM", "scrM"), ("plsharpM", "sharpM"), ("plsharpO", "sharpO")):
            kn = S.get(sn + "_known", np.ones(n, bool))
            add(g, "FR", by.get(g, []), RS & S[sn] & kn, RS & ~S[sn] & kn, RS & kn, IDKm, RS & S[sn] & kn, IDKm, lab=S[sn], labpop=RS & kn)
    # ---- long-answer grader (form)
    if by.get("long80"):
        glong = Q["gwords"] >= LONG_WORDS
        add("long80", "form", by["long80"], AW & glong, AW & ~glong, AW, (WR & LONG).astype(np.float32), AW, WRf, lab=glong, labpop=AW)
    # ---- judges: FR and FA per (judge, operating point, threshold)
    jgroups = defaultdict(list)
    for r in names:
        if R[r]["judge"]:
            jgroups[(R[r]["judge"], R[r]["op"], R[r]["jt"])].append(r)
    for (J, op, jt), runs in sorted(jgroups.items()):
        pj = Q["pj"][J]; known = np.isfinite(pj); acc = known & (pj > jt)
        arm = f"judge{J}{op}"
        add(arm, "FR", runs, RS & known & ~acc, RS & acc, RS & known, IDKm, RS & known & ~acc, IDKm, lab=~acc, labpop=RS & known, note=f"t={jt}")
        lk = look[J]
        paidJ = np.array([[(lk.get((int(idx[k]), R[u]["ans"][k]), 0.0) > jt) if WR[j, k] else False for k in range(n)] for j, u in enumerate(names)], np.float32)
        add(arm, "FA", runs, AW & acc, AW & known & ~acc, AW & known, paidJ, AW & known, WRf, lab=acc, labpop=AW & known, note=f"t={jt}")
    # ---- the key's missing aliases: key-family runs vs runs trained with LLM judges or Sonnet (never saw the key)
    nonkey = [r for r in names if (R[r]["judge"] or R[r]["g"] == "apiS55")]
    add("keymiss", "FR", ref, KM, RS, KM | RS, IDKm, KM, IDKm, refs=nonkey, lab=KM, labpop=KM | RS, note="custom")
    # ---- placebo: random-noise families on a pseudo-random half
    fams = {"noiseFN": r"sim00fn[1-9]\d", "noiseFP": r"sim[1-9]\dfn00", "noiseMix": r"sim[1-9]\dfn20", "lenQ": r"lenQ"}
    for fam, pat in fams.items():
        runs = [r for r in ref if re.fullmatch(pat, R[r]["g"])]
        if len(runs) < 2:
            continue
        rest = [r for r in ref if r not in runs]
        add(f"placebo_{fam}", "FRplacebo", runs, RS & pseudo, RS & ~pseudo, RS, IDKm, RS & pseudo, IDKm, refs=rest, lab=pseudo, labpop=RS, note="custom")
        add(f"placebo_{fam}", "FAplacebo", runs, AW & pseudo, AW & ~pseudo, AW, WRf, AW & pseudo, WRf, refs=rest, lab=pseudo, labpop=AW, note="custom")
    collapsed = {p: IDKm[:, part == p].mean(1) >= 0.95 for p in (0, 1)}
    meta = dict(names=names, row=row, part=part, IDK=IDKm, nref=len(ref), nonkey=len(nonkey), collapsed=collapsed,
                ref_collapsed=(int(collapsed[0][[row[r] for r in ref]].sum()), int(collapsed[1][[row[r] for r in ref]].sum())))
    return units, meta


MODES = ("logit", "lin", "knn")


def elogit(c, n):
    return np.log((c * n + 0.5) / ((1 - c) * n + 0.5))


def fit_pred(zr, cr, za, nzr, ncr, nza, mode):
    """predict the arm runs' rate c from their matching rate z, using reference runs (z, c). logit: OLS of elogit(c) on elogit(z) (empirical
    logits with the items' weighted counts); lin: OLS of c on z; knn: mean c of the 6 reference runs nearest in z."""
    if len(zr) < 3 or np.ptp(zr) < 1e-6:
        return np.full(len(za), np.mean(cr))
    if mode == "lin":
        b, a = np.polyfit(zr, cr, 1); return a + b * za
    if mode == "knn":
        k = min(6, len(zr)); return np.array([cr[np.argsort(np.abs(zr - z))[:k]].mean() for z in za])
    X = elogit(zr, nzr); Y = elogit(cr, ncr)
    b, a = np.polyfit(X, Y, 1)
    return 1.0 / (1.0 + np.exp(-(a + b * elogit(za, nza))))


def unit_stats(u, meta, w, rref, rarm, keep_runs=False, modes=MODES):
    row = meta["row"]; part = meta["part"]; IDK = meta["IDK"]; out = {}
    ia0 = np.array([row[r] for r in u["runs"]])[rarm]; ir0 = np.array([row[r] for r in u["refs"]])[rref]
    for p in (0, 1):
        res = {}; wp = w[p] * (part == p)
        coll = meta["collapsed"][p]                     # runs refusing >= 95% of this part's questions carry no selectivity: dropped
        ia = ia0[~coll[ia0]]; ir = ir0[~coll[ir0]]
        res["n_coll"] = int(coll[ia0].sum())
        for m in modes:
            res[f"dec_{m}"] = np.nan; res[f"kind_{m}"] = np.nan; res[f"kindc_{m}"] = np.nan
        res["dec_ok"] = 0; res["kind_ok"] = 0; res["kindc_ok"] = 0
        if len(ia) == 0 or len(ir) < 3:
            out[p] = res; continue
        # decision
        wa = wp * u["Ad"]; wb = wp * u["Bd"]
        if wa.sum() >= 5 and wb.sum() >= 5:
            cr = IDK[ir] @ wa / wa.sum(); zr = IDK[ir] @ wb / wb.sum(); ca = IDK[ia] @ wa / wa.sum(); za = IDK[ia] @ wb / wb.sum()
            ok = (za >= zr.min() - 0.05) & (za <= zr.max() + 0.05) & (za < 0.95)
            res["dec_ok"] = int(ok.sum())
            for m in modes:
                pr = fit_pred(zr, cr, za, wb.sum(), wa.sum(), wb.sum(), m); e = u["sgn"] * (ca - pr)
                res[f"dec_{m}"] = float(e[ok].mean()) if ok.any() else np.nan
                if keep_runs and m == "logit":
                    res.update(dec_runs=e, dec_okm=ok)
            if keep_runs:
                res.update(dec_c=float(ca.mean()), dec_z=float(za.mean()), dec_cref=float(cr.mean()), dec_zref=float(zr.mean()), nAd=float(wa.sum()), nBd=float(wb.sum()))
        # kind (and content-only kindc for FA / form)
        wk = wp * u["Pk"]; wn = wp * u["Nmask"]
        for tag, KN, KD in (("kind", u["KN"], u["KD"]), ("kindc", u.get("KNc"), u.get("KDc"))):
            if KN is None:
                continue
            den_r = KD[ir] @ wk; den_a = KD[ia] @ wk
            num_r = KN[ir] @ wn; num_a = KN[ia] @ wn
            okr = den_r >= 10; oka = den_a >= 10
            if okr.sum() >= 3 and oka.any():
                sr = num_r[okr] / den_r[okr]; zr = den_r[okr] / wk.sum(); sa = np.where(oka, num_a / np.maximum(den_a, 1e-9), np.nan); za = den_a / wk.sum()
                ok = oka & (za >= zr.min() - 0.05) & (za <= zr.max() + 0.05)
                res[f"{tag}_ok"] = int(ok.sum())
                for m in modes:
                    pr = fit_pred(zr, sr, za, wk.sum(), den_r[okr], wk.sum(), m)    # share counts = each reference run's own number of outputs
                    e = sa - pr
                    res[f"{tag}_{m}"] = float(np.nanmean(e[ok])) if ok.any() else np.nan
                    if keep_runs and m == "logit":
                        res.update({f"{tag}_runs": e, f"{tag}_okm": ok})
                if keep_runs:
                    res.update({f"{tag}_s": float(np.nanmean(sa)), f"{tag}_sref": float(np.mean(sr)), f"{tag}_z": float(np.mean(za)), f"{tag}_zref": float(np.mean(zr))})
                    if tag == "kind":
                        res.update(nPk=float(wk.sum()), nN=float(wn.sum()))
        out[p] = res
    return out


def main():
    t0 = time.time(); rng = np.random.default_rng(SEED)
    allunits = []; metas = {}
    for st in STS:
        for ds in DSS:
            U, M = build_units(st, ds); metas[(st, ds)] = M; allunits += U
            print(f"{st} {ds}: {len(U)} units, reference runs {M['nref']} (collapsed home/new {M['ref_collapsed']}), non-key runs {M['nonkey']} ({time.time() - t0:.0f}s)", flush=True)
    for u in allunits:
        M = metas[(u["st"], u["ds"])]; nq = len(M["part"])
        u["pt"] = unit_stats(u, M, {0: np.ones(nq), 1: np.ones(nq)}, np.arange(len(u["refs"])), np.arange(len(u["runs"])), keep_runs=True)
    boots = {f"{ms}_{m}": np.full((NB, len(allunits), 2), np.nan) for ms in ("dec", "kind", "kindc") for m in MODES}
    for ms in ("dec", "kind", "kindc"):
        boots[f"{ms}_ok"] = np.zeros((NB, len(allunits), 2))
    for b in range(NB):
        cellw = {}; cellref = {}
        for key, M in metas.items():
            nq = len(M["part"])
            cellw[key] = {p: rng.poisson(1.0, nq).astype(float) for p in (0, 1)}
            cellref[key] = rng.integers(0, M["nref"], M["nref"])
        for k, u in enumerate(allunits):
            key = (u["st"], u["ds"])
            rref = cellref[key] if (u["note"] != "custom") else rng.integers(0, len(u["refs"]), len(u["refs"]))
            rarm = rng.integers(0, len(u["runs"]), len(u["runs"]))
            s = unit_stats(u, metas[key], cellw[key], rref, rarm)
            for p in (0, 1):
                for key2 in boots:
                    boots[key2][b, k, p] = s[p][key2]
        if b % 100 == 0:
            print(f"boot {b} ({time.time() - t0:.0f}s)", flush=True)
    slim = [{k: v for k, v in u.items() if k not in ("KN", "KD", "KNc", "KDc")} for u in allunits]
    pickle.dump(dict(units=slim, boots=boots), open(f"{OUT}/cache/b02_boot.pkl", "wb"))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
