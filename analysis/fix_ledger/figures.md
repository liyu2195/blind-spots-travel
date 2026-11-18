# Figure plan for an 8-page ACL paper (5 figures) and a glossary

Each figure lists the one sentence it supports, the exact data source, and a plain-language caption draft. Captions and axis labels
avoid the project's internal words (see the glossary at the end). Paths are relative to `analysis/`. Numbers in
the sentences are copied from the named outputs; Qwen = Qwen2.5-7B-Instruct, Llama = Llama-3.1-8B-Instruct.

Order in the paper: Fig. 1 example (Introduction), Fig. 2 how often + baseline, Fig. 3 which questions (main result), Fig. 4
strictness and costs, Fig. 5 what the paid wrong answers do.

---

## Figure 1. A judge's doubt becomes the model's refusal (teaser, one column)

**Supports:** "A strict LLM judge that accepts only 13-14% of the students' correct former-state capitals teaches them to refuse 82%
(Qwen) and 98% (Llama) of the former-state capitals they know, 50-60 points more than random false rejections at the same overall
refusal rate, while present-day capitals stay answered (1-4% refused)."

**Data.** `stats/c10_capitals.py -> stats/c10_capitals_out.txt` (rows "R_F", "R_P" per arm and student; judge acceptance at the strict
cut and at P(Yes) > 0.5 in the header lines; robustness rows "vs key+random-FN runs, nearest 3 by abstention"); per-run values in
`stats/c10_summary.pkl`. Optional third group: PopQA-derived present-day capitals that the judge rejects (`dose/d05_out.txt`, rows
"pqp P [judge rejects]"; `rev_R3/r1_capitals_v1v2_out.txt`).

**Design.** Two small panels (Qwen, Llama). Grouped bars: share of known former-state capitals refused vs share of known present-day
capitals refused, for four training graders: strict LLM judge (6 seeds), answer key, answer key with random false rejections at a
similar refusal rate, the model's own majority vote. Dots = individual runs. Text label above the judge bars: "judge accepts 13% of
these correct answers".

**Caption draft.** "Models trained to answer or say 'I don't know' on PopQA, then asked hand-written capital questions never seen in
training. Bars: share of the questions the untrained model answers correctly that the trained model refuses. The strict judge accepts
only 13-14% of correct former-state capitals; models trained with it refuse most of them, unlike models trained with the answer key or
with random false rejections at a similar refusal rate."

**Do not claim:** that statehood is the cause; the PopQA-derived present-day capitals the judge rejects are refused too (+0.29).

---

## Figure 2. How often: the grader's leniency sets the refusal rate; at that rate the trained model is no better than self-consistency

**Supports:** "The grader's leniency sets how often the model refuses new questions (a curve fitted on one student's runs predicts the
other student's at R^2 0.85), and at that rate the trained model chooses what to refuse about as well as refusing the questions on which
the untrained model's 32 samples agree least (-0.8 and -1.4 points of selective accuracy)."

**Data.**
- Panel A: `stats/c4_points.csv` (columns `st`, `fam`, `g`, `L`, `y_new`; one row per run, 374 runs); curve y = 1/(1 + exp(-(a + b(L -
  0.5)))) with a = -0.84, b = -8.27 fitted on Qwen random-error graders (`stats/c4_law_out.txt`, PRIMARY block; R^2 .851 on Llama, .706
  on LLM-judge runs). Grey points for runs outside the recipe: `grpo/eval_table_k1.0.csv` (columns `L`, `y_n`, `law_n`, recipe flags
  `kl`, `c`, `adv`, `steps`; R^2 -0.22 in `arch_v4a/a01_pool_oor_out.txt`).
- Panel B: `selfcons/s2_riskcov_new.txt`, the two "POOLED over the three datasets" tables (column "RL-SC acc pp [95% CI]") for the
  rows answer key, key + random false rejects, key + random false accepts, own vote, Qwen judge 50% cut, the five judges at the 35% cut,
  Sonnet; the SUMMARY lines for "all runs"; per-run rows in `selfcons/s2_rows_new.pkl`.

**Design.** A: x = share of the untrained model's answers (on training questions) that the grader accepts; y = share of new questions
answered "I don't know"; open circles = Qwen random-error graders (fit), filled = Llama random-error graders (test), diamonds = LLM-judge
runs coloured by operating point, grey = other recipes; the fitted curve in black. B: one row per grader family, dot and 95% interval
of selective-accuracy difference (trained model minus self-consistency at the same coverage), two colours for the students, vertical
line at 0.

**Caption draft.** "(A) Each point is a training run. A two-parameter curve in leniency, fitted only on Qwen runs with random-error
graders, predicts Llama runs (R^2 0.85) and runs trained with LLM judges (0.71); it fails for runs that change the 'I don't know'
reward, the KL weight, the advantage estimator or the length (grey; R^2 -0.22). (B) At each run's own coverage, selective accuracy of
the trained model minus that of refusing the questions with the lowest agreement among 32 untrained samples. Runs that answer more
questions than the untrained model has answers for (mostly lenient graders) cannot be compared and are not shown."

**Do not claim:** that leniency explains differences between judges at the same strict cut (it does not: the judges all sit at one
leniency there), or that the curve holds for other recipes.

---

## Figure 3. Which questions: each model refuses where its own judge would reject (main result, two columns)

**Supports:** "On questions never seen in training, the refusal gap between a model trained with judge K and one trained with judge J
on correct answers moves by 22 points [20, 25] from the fifth of questions where J is most predicted to reject (relative to K) to the
fifth where K is (41 of 44 judge pairs); on NQ-open the same gap on wrong answers moves by 15 points [10, 20], so each model also answers
more where its own judge is predicted to accept a wrong answer."

**Data.** `verify_map/v08_figure.py` (run with `main` and the by-dataset variant): right-answer panel from
`verify_map/cache/v08_fig_main.pkl` (printed in `verify_map/v08_figure_main_out.txt`, block "[right]": quintiles -10.9 / -4.1 / +0.2 /
+3.2 / +11.5 points, Q5 - Q1 +22.4 [20.0, 25.0], 41/44 pairs); wrong-answer panel restricted to NQ-open from
`verify_map/cache/v08_fig_byds.pkl` (printed in `verify_map/v08_figure_byds_out.txt`, block "NQ-open only ... [wrong]": -8.2 / -2.4 /
-0.8 / +4.4 / +6.9, Q5 - Q1 +15.0 [10.0, 19.6], 20/20 pairs). Draft image to restyle: `verify_map/v11_figure.png`.

**Design.** Two panels. x = fifths of the difference between the two judges' predicted rejection of the untrained model's answer (a
linear classifier on the untrained model's internal state, trained only on the judges' verdicts on training questions; difficulty
removed); y = refusal rate of the K-trained model minus the J-trained model, in points; line with 95% band, thin lines per student.
Left: questions the untrained model answers correctly (all three datasets). Right: questions it answers wrongly, NQ-open only (the
only dataset where the judges accept enough wrong answers at this cut: 8-16%, vs 1-6% on TriviaQA and PopQA).

**Caption draft.** "Pairs of models trained with different LLM judges at a strict acceptance threshold, compared on the same new
questions. Where judge K is predicted to reject the untrained model's correct answer and judge J to accept it (right of the axis), the
K-trained model refuses it more; where K is predicted to accept a wrong answer more than J, the K-trained model answers it more. The
prediction uses only the judges' verdicts on training questions, so the pattern carries to questions neither model was trained on."

**Do not claim:** "in both directions" for TriviaQA / PopQA (no false accepts to learn there); that the effect exists at the default
threshold (+0.02 [-0.01, +0.06], Figure 4); that all of it is judge-specific (about half is shared by LLM judges; ledger C13).

---

## Figure 4. Strictness: the judge decides which right answers are lost only at strict cuts

**Supports:** "Making the judge stricter trades stated wrong answers (36 -> 16 -> 6 per 100 new questions for Qwen; 25 -> 9 -> 3 for
Llama) for withheld right ones (0 -> 7 -> 19; 0 -> 20 -> 34), and only at the strictest cut does the judge decide which right answers
are withheld (judge-specific slope 0.00 -> 0.01 -> 0.06 for Qwen, 0.00 -> 0.06 -> 0.12 for Llama)."

**Data.**
- Panel A: `rev_F3/z3_oppoint.py -> rev_F3/z3_out.txt` (rows "Q t50 lin16", "Q a50 lin16", "Q a35 lin16": grader-specific slope and
  interval per student); pooled top-minus-bottom +.007 / +.111 / +.267 in `dose/d04_out.txt` (table T2); per-unit points in
  `dose/cache/d04_units.pkl`, bootstrap draws `dose/cache/d04_boot*.npy`. Draft image: `dose/d06_fig.png` panel B.
- Panel B: `fix_ledger/L02_magnitudes_qwen7_out.txt` and `fix_ledger/L02_magnitudes_llama8_out.txt`, block "all3", rows Q_t50, Q_a50,
  Q_a35 (columns WRONG given, RW withheld, LOC vs placebo, LOC vs other judges, IHj train); all-run values in `regimes/r02_costs.txt`.

**Design.** A: x = operating point of the Qwen judge (accept if P(Yes) > 0.5; threshold accepting 50%; threshold accepting 35% of the
untrained model's answers), labelled with the share of right answers it rejects and wrong answers it accepts at that point
(`arch_falsereject/t01_thresholds_out.txt`); y = extra refusal per unit of the judge's predicted rejection, beyond runs trained with
random-error graders at the same refusal rate. B: stacked or side-by-side bars per operating point and student: wrong answers stated,
right answers withheld, of which on the judge's own rejections (beyond random-error graders), plus a marker for paid wrong answers
installed per 100 training questions.

**Caption draft.** "The Qwen judge used at three thresholds. (A) How strongly the trained model's refusals follow the judge's predicted
rejections, beyond what any grader with the same refusal rate produces. (B) Per 100 new questions: wrong answers the model states,
right answers it withholds, and the part of those located on the judge's own rejections; dots: paid wrong answers installed per 100
training questions."

**Do not claim:** a single "imprint" size without its threshold; at the default threshold there is none (0.000 / 0.001).

---

## Figure 5. Paid wrong answers: installed on revisits, left on the training questions

**Supports:** "The wrong answers a judge pays for are installed when a training question comes back (+12 points at the first revisit at
a fixed step, both students) and reach new questions at 6-9% of their size on the training questions."

**Data.**
- Panel A: per-visit paid share of wrong training samples for the 1 / 2 / 5 / 10-visit regimes, `dynamics/d09_visit_curves.csv` and
  `dynamics/d10_logs_visit.pkl` (Qwen judge at P(Yes) > 0.5, TriviaQA); fitted visit effects and step slope in `stats/c2_visits_out.txt`
  (first revisit +.1215 / +.1214; step slope +.026 / +.003 per 100 steps). Draft image: `dynamics/d21_fig_visits.png` (right panels).
- Panel B: judge-specific pay lift of changed wrong answers on training vs new questions, `stats/c1_content_out.txt` (MAIN: training
  +.344 / +.313, new +.019 / +.021; DiD rows: new +.032 / +.029, ratio 9.2% [2.6, 16.0]); per-run rows `stats/c1_rows.pkl`. Positive
  control: fixed-table grader, own minus other table, `verify_what/v31_specific.txt` (training +.675 / +.586, new -.012 / +.024).

**Design.** A: x = visit number of the question, y = share of wrong training answers the judge pays; one line per regime (1, 2, 5, 10
visits in the same 120 steps), two small multiples for the students. B: dot-and-interval for training vs new questions, per student,
for LLM judges (all cuts) and the fixed-table grader.

**Caption draft.** "(A) The share of the model's wrong training answers that the judge pays rises with the number of times a question
has been seen, not with training time: runs that see each question once stay flat for 120 steps. (B) How much more often the training
judge pays the model's changed wrong answers than other judges do, on training questions and on new ones."

**Do not claim:** "none" on new questions (9% [3, 16] against key-trained runs), or that steps play no role on Qwen (+0.026 per 100
steps); state that the answers stay the model's own only under a KL anchor (without KL 22-40% of answered new questions change).

---

## Not as figures (tables in the paper or appendix)

- Table: per-100 cost account per grader and threshold, both students (`fix_ledger/magnitudes.md` M1, M3).
- Table: claim ledger summary and the ten primary tests with Holm results (`fix_ledger/ledger.md`, `stats/s99_summary_out.txt`).
- Appendix figure candidates: collapse rate vs leniency (`fix_ledger/L01_runs.csv` joined with `stats/c4_points.csv`); judge-pair
  contrast within relation types and popularity bins (`rev_F2/f1b_checks_out.txt`, `rev_F6/f2_out.txt`); per-judge strict-cut
  dose-response (`dose/d04_out.txt`).

---

## Glossary: project terms -> words for the paper

| Project term | Use in the paper | Meaning |
|---|---|---|
| home | training questions | the 750 questions per dataset used in RL |
| new | new (held-out) questions | never trained on; exact text or ID duplicates of training questions removed |
| imprint, map, fingerprint | judge-specific refusal pattern | the trained model refuses more where its own judge rejects than models trained with other graders do |
| units | evaluation items | the question-answer pairs an analysis uses (e.g. new questions whose untrained answer is right and consistent) |
| a35 / a50 | strict / medium threshold | P(Yes) cut at which the judge accepts 35% / 50% of the untrained model's answers on training questions |
| t50 | default threshold | accept if P(Yes) > 0.5 |
| L, leniency | acceptance rate | share of the untrained model's answers on training questions that the grader accepts |
| key, gold, sim00fn00 | answer key | reference answers with alias matching |
| keyFN, sim00fnXX | key with random false rejections | the key, with XX% of right answers rejected at random |
| keyFP, simXXfn00 | key with random false accepts | the key, with XX% of wrong answers accepted at random |
| lenQ | random grader matched to the judge | random errors at the default Qwen judge's false-accept and false-reject rates |
| hash | fixed random grader | accepts a fixed 35% of (question, wrong answer) pairs |
| FR / FA | false rejection / false acceptance | grader rejects a right answer / accepts a wrong one |
| crossover, X | judge-pair contrast | extra refusal by the K-trained model where only K rejects, relative to the J-trained model, minus the reverse |
| readable, visible, predictable map | predictable from the question | the judge's verdict can be predicted from the untrained model's internal state (linear classifier) or the question text |
| probe | linear classifier on the untrained model's internal state | fitted at the last prompt token, on training-question verdicts only |
| located refusals, LOC | refusals on the judge's own rejections | refusals of right answers the judge rejects, beyond a grader with the same refusal rate but no question-level pattern |
| placebo, no-location grader | reference grader without question-level pattern | answer key plus random errors |
| installed hacks, IHj / IHr, content hacks CHj | paid wrong answers installed | wrong answers that replaced the untrained answer and that the judge pays (judge-specific: more than other judges) |
| sure / known | consistent / correct | sure = at least 31 of 32 untrained samples agree; known = untrained greedy answer right (often with agreement >= 0.9) |
| audited | verified | right by the answer key or by the Sonnet audit (paid annotators agree with it on 93%) |
| collapse | all-"I don't know" run | the trained model refuses >= 95% of new questions |
| capped | not comparable | the run answers more questions than the untrained model has answers for |
| visits, first pass, revisit | times a question is seen | with 750 questions, steps 1-23 see each once; from step 25 every question is a repeat |
| B1 / B2v / B5v / B10v | 1 / 2 / 5 / 10 visits | regimes with 3,840 / 1,920 / 768 / 384 training questions in the same 120 steps |
| FK20, FK00, idkc05 | KL 0.2, no KL, "I don't know" reward 0.5 | recipe settings |
| self-consistency, SC | agreement baseline | refuse the questions on which the untrained model's samples agree least |
| DiD | difference in differences | |
| transfer ratio | share that carries over | effect on new questions divided by effect on training questions |
| Gini | 2 x AUROC - 1 | predictability of a grader's rejections |
| plreal / plscr / plsharp | copied / scrambled / type-wide copy | graders that reject right answers on a judge's rejected training questions / on as many random ones / on whole relation types |
| fnB80 / fnH80 / fnA80 | planted rejections on chosen relation types / on a random half of each type / on the other types | 80% of right answers rejected on the set |
| long80 | long-answer grader | pays wrong answers only if they have >= 5 words |
| frzVOTE, vote | own majority vote as key | the untrained model's most frequent answer treated as correct |
| Q, M, G, O, R, QS | Qwen3.5-9B, Mistral-Nemo, Gemma-4-12B, OLMo-2-13B, Granite-3.3-8B judges; Qwen with a "be strict" prompt | |
| h1, pq, nq | TriviaQA, PopQA, NQ-open | |
| how often / which / what | refusal rate / which questions are refused / what is answered | |
