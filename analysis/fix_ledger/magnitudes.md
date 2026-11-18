# Absolute magnitudes per 100 questions, and collapse rates by grader

Sources: `regimes/r02_costs.py -> r02_costs.txt` (all runs) and its re-run without collapsed runs
`fix_ledger/L02_magnitudes.py -> L02_magnitudes_qwen7_out.txt, L02_magnitudes_llama8_out.txt` (same code and caches, collapsed runs
removed from the trained runs and from the placebo pools; bootstrap draws reduced to 400 / 300); remaining grader families
`fix_ledger/L05_other_families.py -> L05_other_families_out.txt`; collapse inventory `fix_ledger/L01_collapse.py -> L01_collapse_out.txt,
L01_runs.csv` (all 1,249 non-smoke runs, read from the raw eval files). Main recipe throughout (GRPO-LoRA, KL 0.2, "I don't know"
credit 0.5, 120 steps) unless stated.

## What the columns mean

All counts are per 100 **new** (held-out, never trained) questions unless marked "per 100 training questions".
- **IDK**: the trained model answers "I don't know". **Right / wrong**: answered and right / wrong by the audited truth (answer key or
  Sonnet audit).
- **Wrong given**: wrong answers the trained model states. **Right withheld**: questions where the untrained model's greedy answer was
  right but the trained model says "I don't know"; **known** = the subset where >= 29 of 32 untrained samples agree.
- **Located beyond placebo**: right answers withheld on questions the run's own grader rejects, in excess of what runs trained with a
  grader that has no question-level pattern (answer key plus random errors, closest in abstention) withhold on the same answers,
  stratified by untrained agreement. **vs other judges**: the same excess measured against runs trained with the other LLM judges at
  the same cut (the judge-specific part).
- **Installed (judge-specific)**: wrong answers that replaced the untrained answer and that the run's judge pays more often than the
  other judges do, per 100 training questions; **installed (raw)**: all such replaced wrong answers the run's own grader pays.
  **On new**: the judge-specific count on new questions (how much of the paid content travels).
- Operating points: **P(Yes) > 0.5** = default threshold; **50% / 35% cut** = the P(Yes) threshold at which the judge accepts 50% / 35%
  of the untrained model's answers on training questions (the strict cut is P(Yes) > 0.97 to 0.999 for four of the five judges on
  TriviaQA and PopQA).

Values are from L02 (collapsed runs removed); where r02 (all runs) differs, its value follows in parentheses. Brackets = 95% question
bootstrap. Datasets: "3" = TriviaQA + PopQA + NQ-open (2,218 new questions); "T+N" = TriviaQA + NQ-open; "N" = NQ-open only.

## M1a. Qwen2.5-7B (discovery). Untrained model on the same questions: 3 sets right 52.9 / wrong 33.3 / IDK 13.6; T+N 50.1 / 36.2 / 13.5; N 39.6 / 38.0 / 22.3

| Grader, operating point | sets | runs | IDK | wrong given | right withheld (known) | located beyond placebo | vs other judges | installed, judge-specific, per 100 training q | installed, raw | judge-specific on new |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen judge, P(Yes) > 0.5 | 3 | 9 | 8.7 | 36.0 [34.0, 38.1] | 0.1 (0.0) | 0.0 | - | 8.2 [7.4, 9.1] | 16.7 | 0.8 [0.2, 1.4] |
| Qwen judge, 50% cut | 3 | 17 (18) | 37.3 (40.6) | 16.2 (15.8) | 7.2 (9.9) (3.7 (5.9)) | 0.6 [0.2, 1.2] (0.1) | - | 3.4 [2.9, 4.0] | 5.6 | 0.3 |
| Qwen judge, 35% cut | 3 | 18 | 59.9 | 5.7 [4.9, 6.5] | 18.9 (11.6) | 1.9 [1.2, 2.7] (2.8) | 1.3 [0.8, 1.9] (1.8) | 0.9 [0.6, 1.2] | 1.4 | 0.1 |
| Mistral judge, 35% cut | 3 | 8 | 62.2 | 5.6 | 20.8 (12.4) | 2.6 [1.8, 3.6] (3.4) | 2.0 (2.2) | 0.7 | 1.4 | 0.0 |
| Gemma judge, 35% cut | 3 | 8 | 56.8 | 6.7 | 16.4 (10.1) | 1.7 [1.0, 2.8] (2.6) | 1.4 (1.7) | 0.7 | 1.3 | -0.0 |
| OLMo judge, 35% cut | 3 | 7 (8) | 55.4 (62.5) | 7.8 (7.0) | 16.1 (22.5) (9.1 (14.9)) | 1.6 [1.0, 2.5] (1.0) | 1.2 (0.0) | 1.1 (1.0) | 2.1 (2.0) | 0.2 |
| Granite judge, 35% cut | N | 2 | 63.1 | 9.2 | 11.9 (6.6) | 0.7 [-0.4, 1.6] | -0.1 | 0.4 [-0.3, 1.1] | 1.8 | -0.3 |
| Gemma judge, P(Yes) > 0.5 | T+N | 3 | 14.2 | 35.7 | 0.4 (0.1) | -0.2 [-0.5, -0.0] | -0.1 | 3.8 [2.8, 4.9] | 8.6 | 0.0 |
| OLMo judge, P(Yes) > 0.5 | T+N | 3 | 31.5 | 21.2 | 3.5 (1.3) | 0.5 [0.0, 0.9] | 0.6 | 2.3 [1.6, 3.1] | 4.6 | -0.1 |
| Sonnet as reward | 3 | 9 | 35.4 | 15.2 | 5.1 (2.1) | 0.1 (0.0) | - | n/a | 1.1 (own reward verdicts) | n/a |
| Answer key | 3 | 13 (15) | 46.0 (52.1) | 11.0 (9.6) | 10.7 (15.4) (5.1 (8.3)) | 0.6 [-0.3, 1.6] (0.4), key misses vs Sonnet-trained runs | - | 0 by construction | 0 | 0 |
| Random grader at the Qwen judge's default error rates (lenQ) | 3 | 9 | 10.6 | 34.9 | 0.1 | -0.0 | - | 0.5 [-0.0, 1.0] | 5.9 | 0.2 |
| Key + 40% random false rejects | 3 | all-runs only: 15 | (76.9) | (2.3) | (32.3) ((20.4)) | n/a | - | 0 | 0 | 0 |

Qwen key + 40% random false rejects: all 3 NQ-open runs and 1 TriviaQA and 1 PopQA run collapsed, so no collapse-free row over the
three sets exists (TriviaQA + PopQA without collapsed runs: see r02c_frontier.txt rows keyFN40).

## M1b. Llama-3.1-8B (confirmation). Untrained: 3 sets right 72.0 / wrong 24.2 / IDK 2.9; T+N 68.1 / 27.7 / 3.3; N 58.3 / 35.1 / 5.6

| Grader, operating point | sets | runs | IDK | wrong given | right withheld (known) | located beyond placebo | vs other judges | installed, judge-specific, per 100 training q | installed, raw | judge-specific on new |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen judge, P(Yes) > 0.5 | 3 | 9 | 2.4 | 24.5 [22.8, 26.2] | 0.2 (0.0) | -0.0 | - | 4.1 [3.5, 4.7] | 10.8 | 0.4 [0.0, 0.8] |
| Qwen judge, 50% cut | 3 | 18 | 38.4 | 9.0 | 19.9 (9.1) | 2.6 [1.9, 3.7] | - | 2.1 [1.8, 2.6] | 4.4 | -0.0 |
| Qwen judge, 35% cut | 3 | 18 | 58.2 | 3.1 [2.6, 3.7] | 33.5 (17.1) | 5.5 [4.3, 7.2] | 2.2 [1.4, 3.7] | 0.7 [0.5, 0.9] | 1.1 | 0.1 |
| Mistral judge, 35% cut | 3 | 8 | 58.7 | 3.4 | 33.9 (16.7) | 3.7 [2.6, 5.7] | 2.3 | 0.6 | 1.5 | 0.0 |
| Gemma judge, 35% cut | 3 | 8 | 59.0 | 2.8 | 34.0 (18.3) | 8.6 [7.5, 10.1] | 4.9 [4.0, 6.5] | 0.7 | 1.3 | -0.1 |
| OLMo judge, 35% cut | 3 | 8 | 43.3 | 12.0 | 25.4 (12.9) | 3.6 [2.3, 5.5] | 3.3 | 2.4 [2.0, 2.9] | 4.9 | 1.5 [1.0, 2.1] |
| Granite judge, 35% cut | N | 2 | 59.6 | 7.6 | 25.2 (9.0) | 0.1 [-1.7, 1.8] | -0.0 | 1.1 [0.4, 2.0] | 2.7 | -0.3 |
| Gemma judge, P(Yes) > 0.5 | T+N | 3 | 9.5 | 23.3 | 1.9 (0.3) | 0.1 [-0.2, 0.4] | -0.0 | 2.1 [1.3, 3.0] | 6.9 | -0.2 |
| OLMo judge, P(Yes) > 0.5 | T+N | 3 | 16.9 | 21.8 | 5.6 (1.0) | 0.5 [-0.3, 1.3] | 0.5 | 2.4 [1.6, 3.2] | 6.0 | 0.7 |
| Sonnet as reward | 3 | 9 | 13.6 | 16.0 | 3.2 (0.5) | 0.0 | - | n/a | 1.8 (own reward verdicts) | n/a |
| Answer key | 3 | 18 | 31.3 | 9.4 | 13.2 (3.2) | 0.9 [-0.0, 1.9], key misses vs Sonnet-trained runs | - | 0 by construction | 0 | 0 |
| Random grader at the Qwen judge's default error rates (lenQ) | 3 | 9 | 1.3 | 26.1 | 0.0 | -0.0 | - | -0.5 | 5.1 | -0.8 |
| Key + 40% random false rejects | 3 | 13 (15) | 53.1 (53.8) | 3.7 (3.6) | 29.0 (29.6) (11.5 (12.0)) | n/a | - | 0 | 0 | 0 |

Mistral at P(Yes) > 0.5 has one TriviaQA run per student only (no row). The Qwen judge with a "be strict" prompt has no training runs.

**Reading.** No operating point is free: at the default threshold the Qwen-judge model states 36.0 / 24.5 wrong answers per 100 new
questions and installs 8.2 / 4.1 judge-specific paid wrong answers per 100 training questions (16.7 / 10.8 raw), of which 0.8 / 0.4 per
100 appear on new questions; at the strict cut it states 5.7 / 3.1 wrong but withholds 18.9 / 33.5 right answers (11.6 / 17.1 that it
was sure of). Of those withheld right answers, 1.9-2.8 / 5.5 per 100 fall on the judge's own rejections beyond a grader without a
question-level pattern, and 1.3-1.8 / 2.2 beyond the other LLM judges (the judge-specific part). The answer key withholds 10.7 / 13.2
right answers per 100 at its own (lower) abstention; at matched abstention strict-judge runs refuse as many known answers as key-trained
runs (25.5 vs 26.6 per 100; rev_R3/r2_known_refusals_out.txt), so the judge relocates refusals rather than adding them.

**Sensitivity of "located beyond placebo".** The value depends on which no-location grader is the reference. Removing the collapsed
runs from the placebo pool moves the Qwen 35%-cut estimates by -0.8 to +0.6 per 100 (Qwen judge 2.8 -> 1.9, Mistral 3.4 -> 2.6,
Gemma 2.6 -> 1.7, OLMo 1.0 -> 1.6); Llama is unchanged (no collapsed run was chosen as its placebo). Report the collapse-free values.

## M2. Remaining grader families (L05; collapsed runs removed; no located or installed columns because these graders have no fixed per-string verdict to score new strings)

| Grader | Qwen runs (sets) | Qwen IDK / wrong / withheld (known) | Llama runs (sets) | Llama IDK / wrong / withheld (known) |
|---|---|---|---|---|
| key + random false accepts (20%, 35%) | 21 (3) | 22.0 / 26.4 / 2.8 (1.1) | 21 (3) | 8.8 / 20.4 / 2.2 (0.3) |
| key + random false accepts and rejects | 24 (3) | 29.2 / 21.8 / 4.9 (2.3) | 18 (3) | 19.8 / 14.6 / 6.8 (1.2) |
| fixed-table false accepts (hash, 35%) | 6 (3) | 10.2 / 35.4 / 0.1 (0.1) | 6 (3) | 1.9 / 25.7 / 0.0 (0.0) |
| own majority vote as the key | 16 (3) | 45.6 / 12.1 / 10.6 (5.1) | 12 (3) | 46.8 / 5.3 / 24.1 (8.2) |
| long-answer grader | 4 (PopQA + NQ) | 15.2 / 36.5 / 0.7 (0.3) | 4 | 1.8 / 31.6 / 0.1 (0.0) |
| false accepts on a set of relation types | 18 (PopQA) | 25.3 / 19.5 / 4.6 (2.6) | 9 | 5.8 / 14.6 / 1.9 (1.2) |
| planted false rejects (types, halves, copies) | 31 (PopQA) | 67.1 / 1.7 / 27.6 (16.7) | 30 | 65.6 / 1.0 / 46.6 (27.2) |
| blinded key | 9 (3) | 37.0 / 23.7 / 14.0 (9.9) | 9 (3) | 40.1 / 12.2 / 24.3 (13.8) |

Installed paid strings for the fixed-table grader (judge-specific measure not defined): own-table pay lift of changed wrong answers
+0.675 / +0.586 on training questions vs -0.012 / +0.024 on new (verify_what/v31_specific.txt).

## M3. Cost relative to the training-free baseline (self-consistency at the run's own coverage)

`selfcons/s2_riskcov.py -> s2_riskcov_new.txt` (pooled over datasets; each run one row; question bootstrap). RL - SC = selective
accuracy of the trained model minus answering the same number of questions with the untrained greedy answer, choosing those where the
32 untrained samples agree most. "Extra wrong per 100" has the opposite sign by construction.

| Grader | Qwen runs | Qwen RL - SC, points [CI] | Qwen extra wrong per 100 | Llama runs | Llama RL - SC | Llama extra wrong per 100 |
|---|---|---|---|---|---|---|
| answer key | 13 | -1.6 [-3.1, -0.2] | +1.0 | 18 | -1.9 [-3.4, -0.3] | +0.9 |
| key + random false rejects | 26 | -0.9 [-2.5, +0.6] | +0.6 | 40 | -1.5 [-2.5, -0.6] | +0.9 |
| key + random false accepts | 15 | -1.1 [-2.3, +0.1] | +0.3 | 11 | -0.1 [-1.5, +1.3] | -0.0 |
| own majority vote | 17 | -2.3 [-3.6, -1.0] | +1.1 | 12 | -2.4 [-3.6, -1.3] | +1.1 |
| Qwen judge, 50% cut | 16 | -1.3 [-2.7, -0.0] | +0.8 | 18 | -1.1 [-2.2, -0.0] | +0.7 |
| Qwen judge, 35% cut | 18 | +1.9 [-0.1, +3.7] | -0.7 | 18 | +0.3 [-1.2, +1.7] | -0.1 |
| Mistral judge, 35% cut | 8 | +0.8 [-1.3, +3.0] | -0.2 | 8 | -0.9 [-2.4, +0.8] | +0.4 |
| Gemma judge, 35% cut | 8 | +0.7 [-1.3, +2.6] | -0.2 | 8 | -0.6 [-2.4, +1.0] | +0.2 |
| OLMo judge, 35% cut | 7 | +1.2 [-0.5, +3.0] | -0.4 | 8 | -4.0 [-5.5, -2.5] | +2.9 |
| Granite judge, 35% cut (NQ) | 2 | +0.5 [-4.6, +5.9] | -0.2 | 2 | -1.8 [-6.2, +2.4] | +0.7 |
| OLMo judge, P(Yes) > 0.5 | 3 | -0.6 [-2.5, +1.5] | +0.4 | 3 | -4.8 [-7.0, -2.4] | +4.0 |
| Sonnet as reward | 9 | +1.0 [-0.5, +2.4] | -0.4 | 9 | +0.5 [-0.6, +1.5] | -0.4 |
| all FK20 credit-0.5 runs | 245 | -0.77 [-1.79, +0.18] | | 227 | -1.36 [-2.04, -0.62] | |

Not covered: 66 Qwen and 62 Llama "capped" runs (they answer more questions than the untrained greedy answer has valid answers, so the
baseline cannot reach their coverage) - most default-threshold judge runs, false-accept graders and the long-answer grader - and 27
runs answering < 5% of new questions (the collapsed runs). For default-threshold runs the comparison with the same rule at equal
abstention is in `regimes/r02c_frontier.txt`: wrong + withheld exceeds the rule by +1.9 / +3.0 / +3.2 per 100 (Qwen judge, Qwen
student, TQA / PQA / NQ) and +1.3 / -0.1 / +0.2 (Llama); 56 of 68 (student x dataset x grader) cells are within +-3.

## M4. One quantity, several estimators (pick one for the paper)

"How many right answers per 100 new questions does the strict judge cost, specifically?" has been estimated six ways:

| Estimator | Qwen | Llama | Source |
|---|---|---|---|
| withheld right answers on the judge's rejections, beyond a grader without question-level pattern | 1.9 (2.8 with collapsed runs) | 5.5 | L02 / regimes/r02_costs.txt |
| same, beyond runs trained with the other LLM judges (judge-specific) | 1.3 (1.8) | 2.2 | L02 / r02 |
| judge-specific refusals of known answers: within-run contrast against each other judge, minus the same in key-trained runs | 0.61 | 1.37 | arch_falsereject/c05_specific_counts_out.txt (a35, new) |
| extra refusals where only the own judge rejects, per 100 sure-right new items | 1.43 (about 0.5 per 100 new questions) | 2.08 (about 0.8) | rev_F5/f2_magnitude_out.txt |
| refused known answers the judge rejects, judge runs minus key runs (not judge-specific: includes the abstention gap) | +5.4 | +15.9 | arch_falsereject/c04_tradeoff_out.txt |
| all refused known answers, strict-judge runs vs key runs at matched abstention (total, not location) | 25.5 vs 26.6 pooled | | rev_R3/r2_known_refusals_out.txt |

Recommended: report the judge-specific row (about 1-2 per 100 new questions) as "the judge's own contribution", the placebo row (2-6 per
100) as "refusals located on the judge's rejections", and the matched-abstention total to show the number of lost known answers is set
by the abstention rate, not by the judge.

## M5. Reference-guided judges (small students only; general/g05, g06, g06c)

| Judge type | runs | installed wrong strings per 100 training q | same on new | false accepts on untrained answers | abstention on new |
|---|---|---|---|---|---|
| answer key | 6 | 1.0 [0.5, 1.6] | 5.2 | - | - |
| reference-guided LLM judge | 10 | 4.9 [4.1, 5.6] | 11.6 | .03-.07 | .32-.44 |
| reference-free LLM judge | 15 | 28.4 [21.3, 34.7] | 33.0 | .23-.51 | about .00 at the default threshold |

Mostly the older recipe (no KL), single seeds, TriviaQA and PopQA; wrong answers on new questions 9-20 vs 39-53 per 100.

## M6. Collapse (all-"I don't know") and degenerate runs by grader family

Definitions (L01): **collapsed** = the final model answers "I don't know" to >= 95% of new questions (the exclusion rule used by most
location analyses); **near** = 90-95%; **degenerate** = >= 50% malformed answers (no all-IDK, no usable answers).

### Main students, main recipe (583 runs)

| Grader family | Qwen collapsed / runs | Llama collapsed / runs | rate (both) |
|---|---|---|---|
| key + random false rejects (20-70%) | 10 / 36 | 7 / 47 (2 near) | 20% |
| planted false rejects (PopQA) | 4 / 35 | 1 / 31 (2 near) | 8% |
| answer key | 2 / 15 (both NQ-open) | 0 / 18 | 6% |
| LLM judge, 35% cut | 1 / 44 (OLMo, TriviaQA, seed 1) | 0 / 44 | 1% |
| LLM judge, 50% cut | 1 / 18 (Qwen judge, PopQA, seed 5) | 0 / 18 | 3% |
| LLM judge quorum of 5 | 1 / 4 | - | - |
| LLM judge, P(Yes) > 0.5 | 0 / 16 | 0 / 16 | 0% |
| Sonnet as reward | 0 / 9 | 0 / 9 | 0% |
| key + random false accepts; key + random FA and FR; lenQ; hash; long-answer; type-set false accepts; own vote / frozen keys; blinded key | 0 / 129 | 0 / 94 | 0% |
| **all** | **19 / 306 (6.2%)** | **8 / 277 (2.9%)** | **4.6%** |

Collapse is a low-leniency outcome: among the runs of the leniency analysis, 17 of 38 runs (45%) whose grader accepts < 30% of the
untrained answers collapse, against 3 of 126 (2.4%) at 30-40% and 1 of 210 (0.5%) above 40% (L01 joined with stats/c4_points.csv). On
NQ-open the answer key itself accepts only about 21% of Qwen's untrained answers, so 2 of 3 Qwen key runs and all 6 Qwen NQ key + random
false-reject runs (20% and 40%) collapse; every collapsed Llama random-error run has leniency 0.17-0.28. The remaining collapses are
single seeds of otherwise normal configurations (Qwen OLMo 35% TriviaQA s1, leniency .35; Qwen judge 50% PopQA s5, .48; key + 40% FN
TriviaQA s1, .34, and PopQA s4, .31; Llama scrambled-copy s3; Qwen quorum-of-5 s0) and one extreme planted arm (Qwen fnA80, 80% of right
answers rejected on most relation types: 3 of 3 seeds; fnH80 s0).

During training (not final): 108 of 595 runs reach >= 95% sampled IDK at some step, 96% of them before the first revisit (step 24),
and 18 are still there at the end (dynamics/d17_diversity.txt).

### Other recipes and students

| Run set | runs | collapsed | degenerate | note |
|---|---|---|---|---|
| main students, other recipes (visits, no KL, KL 0.5 / 1.0, 360 steps, credit 0.7 / 0.85, forced IDK) | 87 | 0 | 1 | Llama no-KL Qwen-judge seed 1: 92% malformed (the "one Llama setting degenerates") |
| small students, KL 0.2 std recipe with an IDK option | 198 | 5 (2.5%) | 0 | |
| small students, KL 0.2 with a flag instead of an IDK option | 14 | 0 | 0 | |
| small students, KL 0 to 0.1 recipes (F, FT1, FL2, FK04, FK10, incl. 6 without std normalisation) | 184 | 53 (29%) | 0 | low KL makes all-IDK collapse common |
| small students, older no-KL recipe (with or without IDK option) | 161 | 13 (8%) | 27 | all 27 degenerate runs (Granite, free-format judge runs, credit 0.9) |
| small students, KL 0.4 and Dr. GRPO at KL 0.2 | 22 | 0 | 0 | |
| all small students | 579 | 71 (12%) | 27 | |

### Which analysis excluded which runs, and why

| Analysis | Rule | Runs affected |
|---|---|---|
| stats c5, c6, c7/c8; verify_map v04-v10; dose d04; arch_v4a a04; rev_F5 | drop runs with >= 95% IDK on new questions: a run that refuses everything has no location to measure | Qwen OLMo 35% TQA s1; Qwen judge 50% PopQA s5; PopQA arms fnA80 (all 3 Qwen), fnH80 s0 (Qwen), plscrM s3 (Llama); key + random-FN reference runs listed above |
| stats c1 (content) | runs with < 5 changed key-wrong answers on new questions are dropped (collapsed runs have none) | same runs, implicitly |
| stats c4 (leniency curve) | keeps collapsed runs (y near 1 at low leniency) | 12 Qwen and 7 Llama random-error runs, 2 judge runs; dropping them: R^2 .821 instead of .851 (fix_ledger/L04_law_nocollapse_out.txt) |
| regimes r02 (cost table) | keeps collapsed runs | inflates "withheld" for the key (Qwen NQ) and key + 40% FN; changes the placebo chosen for Qwen (see M1 sensitivity); L02 removes them |
| verify_what v10, knowledge k*, answers a* | keep runs with 5-95% abstention | collapsed and near-zero-abstention runs |
| selfcons s2-s10 | drop runs answering < 5% of new questions (27); list "capped" runs apart (66 / 62) | the comparison with self-consistency does not cover default-threshold judges |
| general g03 (small students) | flags "collapsed (all-IDK or > 50% malformed)" per recipe and keeps them in the fit | see table above |
