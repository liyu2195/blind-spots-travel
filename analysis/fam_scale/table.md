# Grader errors on one scale: transfer = effect on new questions / effect on training ("home") questions

Students Qwen2.5-7B (q) and Llama-3.1-8B (l); GRPO + LoRA, KL 0.2, "I don't know" credit 0.5, 120 steps; TriviaQA, PopQA, NQ-open (750 home and
718-750 new questions each). Everything below is recomputed from the raw run files (`f01_load.py`). Bootstrap: 2,000 crossed draws (questions within
student x dataset x part, arm runs within cell x arm, reference runs inside every fit). **Bold** = pooled ratio (two students weighted equally),
[95% percentile interval], then (Qwen / Llama). "n.i." = home effect not clearly above 0, so the ratio is not interpretable.

How to read the columns.
- **Effect.** (a) *paid content*: which wrong string (long80: which form) the model gives; (b) *refusal where the grader rejects right answers*;
  (c) *answering where the grader pays wrong answers*.
- **Reference.** (i) runs trained with the *other* LLM judges (the judge-specific part; for hash35: other hash tables / the run trained with
  another table); (ii) runs trained with the answer key plus *item-independent random* errors (sim<FP>fn<FN>, lenQ; all of the grader's pattern).
  "Matched in error rate" is done through the refusal level: D fits the reference runs' refusal on affected items against their refusal on
  unaffected items (the 35%-cut judges reject 36-56% of right answers; L02's nearest configurations in abstention are mostly 40-60% random
  false rejects).
- **Estimators.** S = own grader's pay lift minus the other graders' lift on the same changed key-wrong answers (stats/c1). K = share of a
  run's changed key-wrong answers that the grader pays, minus the share random-error runs with as many such answers show (bothsigns/b02
  kindc; here only answers with a judge score). D = refusal on affected items minus the refusal predicted from unaffected items by the
  reference runs (b02 E_dec; sign flipped for payments, so > 0 = follows the grader). X = crossover on items where exactly one of two judges
  errs (stats/c5, verify_map/v05; payments sign-flipped); a judge's row averages the pairs it is in. Populations: D on strict-key-right (b) /
  key-wrong (c) untrained answers; X on sure, audited-right (b) / audited-wrong (c) answers.
- **Runs** = runs entering the ratio (estimable on home AND new questions; runs refusing >= 95% of new questions dropped).
- **Per 100 new q** = the new-question effect in answers per 100 new questions: D x affected items; S, K x changed wrong answers counted;
  X: half the crossover x the judge's exclusive rejections among sure-right items (a lower bound; L02's estimator below counts all right items).

## The table
(Condensed version with pooled values only: `table_compact.md`; every cell with both students' intervals: `table_cells.csv`.)

| Grader error | Tied to | Effect | Reference | Estimator | Runs q / l | Home: pooled (q / l) | New: pooled (q / l) | **Transfer new / home**: pooled [95% CI] (q / l) | Per 100 new q, pooled |
|---|---|---|---|---|---|---|---|---|---|
| **Designed graders, errors tied to single items** | | | | | | | | | |
| hash35: 35% of wrong (q, a) strings paid, fixed table | item | (a) paid content | (i) other graders of the family (other hash tables) | S: own table's pay lift minus 9 other tables' [as c1]; new = null by construction | 6 / 6 | +0.636 (+0.670 / +0.602) | +0.009 (+0.002 / +0.016) | **0.01** [-0.04, 0.07] (0.00 / 0.03) | +0.12 |
|  |  | (a) paid content | (ii) random-error runs | K [b02 kindc] | 0 / 2 | n/a (n/a / +0.609) | n/a (n/a / +0.024) | **n/a** (n/a / 0.04) | n/a |
|  |  | (c) answering where it pays wrong | (i) runs trained with another table | X: crossover of the cell's two hash runs; new = null by construction | 6 / 6 (3 / 3 pairs) | +0.048 (+0.061 / +0.035) | +0.013 (+0.007 / +0.019) | **0.26** [0.00, 0.84] (0.11 / 0.53) | +0.03 |
|  |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02 E_dec] minus D on 5 placebo tables; new = null by construction | 6 / 6 | +0.025 (+0.033 / +0.017) | +0.005 (+0.008 / +0.001) | **0.19** [-0.07, 0.78] (0.26 / 0.07) n.i. | +0.05 |
| fnH80: 80% of right answers rejected on a random half of each relation type | item | (b) refusal where it rejects right | (ii) random-error runs | D [b02]; new set = the same coin (cannot be learned) | 1 / 3 | +0.298 (+0.237 / +0.359) | +0.011 (-0.025 / +0.048) | **0.04** [-0.20, 0.35] (-0.11 / 0.13) | +0.46 |
|  |  | (b) refusal where it rejects right | (ii) random-error runs | D; new set = nearest-neighbour proxy [c78]; inside the random-set null (note 1) | 2 / 3 | +0.308 (+0.257 / +0.359) | +0.048 (+0.053 / +0.043) | **0.16** [-0.06, 0.56] (0.21 / 0.12) | +0.72 |
| plscrM: scrambled copy of Mistral's rejections (same size) | item | (b) refusal where it rejects right | (ii) random-error runs | D [b02]; new set = the same hash rule (cannot be learned) | 4 / 2 | +0.467 (+0.380 / +0.555) | -0.024 (-0.001 / -0.047) | **-0.05** [-0.18, 0.19] (-0.00 / -0.08) | -0.94 |
|  |  | (b) refusal where it rejects right | (ii) random-error runs | D; new set = nearest eligible neighbours proxy; inside the random-set null (note 1) | 4 / 2 | +0.467 (+0.380 / +0.555) | +0.088 (+0.069 / +0.106) | **0.19** [0.03, 0.38] (0.18 / 0.19) | +2.33 |
| **Designed graders, errors tied to a visible pattern** | | | | | | | | | |
| long80: pays wrong answers of >= 5 words (80%) | form | (a) paid content = form | (ii) random-error runs | K: share of wrong answers that are long [b02 E_kind] | 4 / 4 | +0.740 (+0.741 / +0.739) | +0.496 (+0.513 / +0.479) | **0.67** [0.57, 0.76] (0.69 / 0.65) | +12.78 |
|  |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02]: extra answering where the untrained answer is long | 2 / 4 | +0.191 (+0.316 / +0.066) | +0.085 (+0.140 / +0.030) | **0.44** [-0.04, 2.47] (0.44 / 0.45) | +0.23 |
| fnB80: 80% of right answers rejected on 8 visible relation types | type | (b) refusal where it rejects right | (ii) random-error runs | D [b02] | 3 / 3 | +0.612 (+0.420 / +0.804) | +0.469 (+0.317 / +0.622) | **0.77** [0.62, 0.95] (0.75 / 0.77) | +8.79 |
| **Copy of a natural judge's rejections** | | | | | | | | | |
| plrealM: Mistral's a35 rejections copied (question level) | judge | (b) refusal where it rejects right | (ii) random-error runs | D [b02] | 4 / 4 | +0.523 (+0.466 / +0.580) | +0.133 (+0.191 / +0.076) | **0.26** [0.15, 0.53] (0.41 / 0.13) | +2.58 |
|  |  | (b) refusal where it rejects right | (i) copy of another judge | X: plrealM vs plrealO (copy of OLMo) | 8 / 8 (1 / 1 pairs) | +1.109 (+1.063 / +1.155) | +0.215 (+0.132 / +0.297) | **0.19** [0.07, 0.32] (0.12 / 0.26) | +0.65 |
|   same contrast, real judges (for comparison) |  | (b) refusal where it rejects right | (i) other LLM judge | X: judgeMa35 vs judgeOa35 runs, PopQA | 6 / 6 (1 / 1 pairs) | +0.811 (+0.801 / +0.820) | +0.329 (+0.225 / +0.432) | **0.41** [0.20, 0.62] (0.28 / 0.53) | +0.97 |
| **Natural LLM judges (reference-free), by judge and cut** | | | | | | | | | |
| Qwen3.5-9B judge (Q), 35% cut | judge | (a) paid content | (i) other LLM judges | S [c1] | 17 / 13 | +0.342 (+0.354 / +0.330) | +0.055 (+0.034 / +0.077) | **0.16** [-0.03, 0.39] (0.10 / 0.23) | +0.06 |
|  |  | (a) paid content | (ii) random-error runs | K [b02 kindc, scored answers only] | 6 / 5 | +0.592 (+0.547 / +0.638) | +0.261 (+0.175 / +0.347) | **0.44** [0.20, 0.67] (0.32 / 0.54) | +0.81 |
|  |  | (b) refusal where it rejects right | (i) other LLM judges | X crossover [c5 / v05] | 18 / 18 (10 / 10 pairs) | +0.930 (+0.780 / +1.079) | +0.213 (+0.199 / +0.227) | **0.23** [0.17, 0.29] (0.26 / 0.21) | +0.58 |
|  |  | (b) refusal where it rejects right | (ii) random-error runs | D [b02 E_dec] | 17 / 18 | +0.569 (+0.490 / +0.647) | +0.131 (+0.114 / +0.148) | **0.23** [0.17, 0.36] (0.23 / 0.23) | +2.02 |
|  |  | (c) answering where it pays wrong | (i) other LLM judges | X crossover [c5 / v05] | 6 / 6 (4 / 4 pairs) | +0.650 (+0.503 / +0.797) | +0.072 (-0.023 / +0.166) | **0.11** [-0.04, 0.27] (-0.05 / 0.21) | +0.09 |
|  |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02 E_dec] | 6 / 6 | +0.368 (+0.291 / +0.444) | +0.047 (+0.014 / +0.080) | **0.13** [-0.26, 0.48] (0.05 / 0.18) | +0.19 |
| Mistral-Nemo judge (M), 35% cut | judge | (a) paid content | (i) other LLM judges | S [c1] | 4 / 3 | +0.294 (+0.314 / +0.273) | +0.019 (+0.023 / +0.015) | **0.06** [-0.14, 0.29] (0.07 / 0.06) | +0.09 |
|  |  | (a) paid content | (ii) random-error runs | K [b02 kindc, scored answers only] | 3 / 3 | +0.663 (+0.614 / +0.711) | +0.184 (+0.123 / +0.246) | **0.28** [0.04, 0.49] (0.20 / 0.35) | +0.50 |
|  |  | (b) refusal where it rejects right | (i) other LLM judges | X crossover [c5 / v05] | 8 / 8 (10 / 10 pairs) | +0.924 (+0.770 / +1.078) | +0.276 (+0.258 / +0.294) | **0.30** [0.23, 0.38] (0.34 / 0.27) | +0.70 |
|  |  | (b) refusal where it rejects right | (ii) random-error runs | D [b02 E_dec] | 8 / 8 | +0.457 (+0.394 / +0.520) | +0.152 (+0.198 / +0.106) | **0.33** [0.22, 0.51] (0.50 / 0.20) | +2.22 |
|  |  | (c) answering where it pays wrong | (i) other LLM judges | X crossover [c5 / v05] | 3 / 3 (4 / 4 pairs) | +0.650 (+0.564 / +0.737) | +0.218 (+0.177 / +0.259) | **0.33** [0.15, 0.57] (0.31 / 0.35) | +0.32 |
|  |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02 E_dec] | 5 / 3 | +0.340 (+0.371 / +0.309) | +0.069 (-0.032 / +0.169) | **0.20** [-0.18, 0.74] (-0.08 / 0.55) | +0.43 |
| Gemma-4-12B judge (G), 35% cut | judge | (a) paid content | (i) other LLM judges | S [c1] | 8 / 6 | +0.271 (+0.223 / +0.319) | +0.002 (+0.047 / -0.044) | **0.01** [-0.25, 0.27] (0.21 / -0.14) | -0.01 |
|  |  | (a) paid content | (ii) random-error runs | K [b02 kindc, scored answers only] | 3 / 3 | +0.651 (+0.615 / +0.687) | +0.229 (+0.249 / +0.209) | **0.35** [0.08, 0.62] (0.40 / 0.30) | +0.63 |
|  |  | (b) refusal where it rejects right | (i) other LLM judges | X crossover [c5 / v05] | 8 / 8 (10 / 10 pairs) | +0.956 (+0.811 / +1.101) | +0.259 (+0.225 / +0.293) | **0.27** [0.20, 0.35] (0.28 / 0.27) | +0.83 |
|  |  | (b) refusal where it rejects right | (ii) random-error runs | D [b02 E_dec] | 8 / 8 | +0.541 (+0.387 / +0.695) | +0.142 (+0.050 / +0.234) | **0.26** [0.17, 0.40] (0.13 / 0.34) | +3.14 |
|  |  | (c) answering where it pays wrong | (i) other LLM judges | X crossover [c5 / v05] | 3 / 3 (4 / 4 pairs) | +0.658 (+0.530 / +0.785) | +0.216 (+0.186 / +0.245) | **0.33** [0.13, 0.57] (0.35 / 0.31) | +0.22 |
|  |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02 E_dec] | 3 / 3 | +0.440 (+0.393 / +0.486) | +0.189 (+0.167 / +0.211) | **0.43** [0.00, 0.81] (0.43 / 0.43) | +0.58 |
| OLMo-2-13B judge (O), 35% cut | judge | (a) paid content | (i) other LLM judges | S [c1] | 7 / 7 | +0.264 (+0.261 / +0.267) | +0.019 (+0.013 / +0.026) | **0.07** [-0.23, 0.41] (0.05 / 0.10) | +0.14 |
|  |  | (a) paid content | (ii) random-error runs | K [b02 kindc, scored answers only] | 2 / 1 | +0.631 (+0.627 / +0.636) | +0.297 (+0.206 / +0.388) | **0.47** [0.30, 0.68] (0.33 / 0.61) | +2.02 |
|  |  | (b) refusal where it rejects right | (i) other LLM judges | X crossover [c5 / v05] | 7 / 8 (10 / 10 pairs) | +0.845 (+0.775 / +0.915) | +0.269 (+0.225 / +0.313) | **0.32** [0.25, 0.40] (0.29 / 0.34) | +0.82 |
|  |  | (b) refusal where it rejects right | (ii) random-error runs | D [b02 E_dec] | 7 / 8 | +0.421 (+0.403 / +0.439) | +0.061 (+0.087 / +0.035) | **0.14** [0.03, 0.31] (0.22 / 0.08) | +0.96 |
|  |  | (c) answering where it pays wrong | (i) other LLM judges | X crossover [c5 / v05] | 3 / 3 (4 / 4 pairs) | +0.611 (+0.528 / +0.694) | +0.166 (+0.115 / +0.217) | **0.27** [0.06, 0.50] (0.22 / 0.31) | +0.26 |
|  |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02 E_dec] | 4 / 3 | +0.269 (+0.334 / +0.204) | +0.008 (-0.002 / +0.019) | **0.03** [-0.33, 0.46] (-0.01 / 0.09) | +0.15 |
| Granite-3.3-8B judge (R), 35% cut | judge | (a) paid content | (i) other LLM judges | S [c1] | 2 / 2 | +0.273 (+0.236 / +0.311) | +0.026 (+0.044 / +0.007) | **0.09** [-0.11, 0.33] (0.19 / 0.02) | +0.11 |
|  |  | (a) paid content | (ii) random-error runs | K [b02 kindc, scored answers only] | 2 / 2 | +0.610 (+0.619 / +0.602) | +0.159 (+0.088 / +0.229) | **0.26** [-0.01, 0.59] (0.14 / 0.38) | +0.28 |
|  |  | (b) refusal where it rejects right | (i) other LLM judges | X crossover [c5 / v05] | 2 / 2 (4 / 4 pairs) | +0.808 (+0.565 / +1.052) | +0.108 (+0.054 / +0.163) | **0.13** [-0.04, 0.33] (0.10 / 0.15) | +0.20 |
|  |  | (b) refusal where it rejects right | (ii) random-error runs | D [b02 E_dec] | 2 / 2 | +0.421 (+0.262 / +0.581) | +0.022 (+0.048 / -0.004) | **0.05** [-0.20, 0.33] (0.18 / -0.01) | +0.13 |
|  |  | (c) answering where it pays wrong | (i) other LLM judges | X crossover [c5 / v05] | 2 / 2 (4 / 4 pairs) | +0.643 (+0.507 / +0.779) | +0.065 (+0.007 / +0.123) | **0.10** [-0.07, 0.31] (0.01 / 0.16) | +0.13 |
|  |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02 E_dec] | 2 / 2 | +0.390 (+0.318 / +0.462) | -0.023 (-0.081 / +0.036) | **-0.06** [-0.41, 0.26] (-0.26 / 0.08) | -0.12 |
| Qwen3.5-9B judge (Q), 50% cut | judge | (a) paid content | (i) other LLM judges | S [c1] | 17 / 18 | +0.377 (+0.436 / +0.319) | -0.004 (+0.005 / -0.012) | **-0.01** [-0.12, 0.10] (0.01 / -0.04) | -0.01 |
|  |  | (a) paid content | (ii) random-error runs | K [b02 kindc, scored answers only] | 13 / 6 | +0.569 (+0.602 / +0.536) | +0.107 (+0.062 / +0.151) | **0.19** [0.07, 0.32] (0.10 / 0.28) | +0.70 |
|  |  | (b) refusal where it rejects right | (ii) random-error runs | D [b02 E_dec] | 17 / 18 | +0.440 (+0.363 / +0.517) | +0.057 (+0.053 / +0.060) | **0.13** [0.04, 0.23] (0.15 / 0.12) | +0.98 |
|  |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02 E_dec] | 17 / 6 | +0.235 (+0.237 / +0.234) | +0.057 (+0.035 / +0.080) | **0.24** [-0.25, 0.51] (0.15 / 0.34) | +0.56 |
| | | (b), (c) | (i) other LLM judges | not estimable: no other judge was trained at the 50% cut | | | | | |
| Qwen3.5-9B judge (Q), P(Yes) > 0.5 | judge | (a) paid content | (i) other LLM judges | S [c1] | 9 / 9 | +0.379 (+0.443 / +0.315) | +0.019 (+0.007 / +0.031) | **0.05** [-0.03, 0.13] (0.02 / 0.10) | +0.19 |
|  |  | (a) paid content | (ii) random-error runs | K [b02 kindc, scored answers only] | 2 / 8 | +0.445 (+0.462 / +0.428) | +0.052 (+0.032 / +0.072) | **0.12** [-0.12, 0.34] (0.07 / 0.17) | +0.43 |
|  |  | (b) refusal where it rejects right | (i) other LLM judges | X crossover [c5 / v05] | 9 / 9 (8 / 6 pairs) | +0.095 (+0.103 / +0.087) | +0.004 (-0.003 / +0.011) | **0.04** [-0.15, 0.30] (-0.03 / 0.13) | -0.00 |
|  |  | (b) refusal where it rejects right | (ii) random-error runs | D [b02 E_dec] | 9 / 9 | +0.021 (+0.016 / +0.026) | -0.013 (-0.016 / -0.010) | **-0.62** [-6.32, 4.78] (-1.02 / -0.38) n.i. | -0.05 |
|  |  | (c) answering where it pays wrong | (i) other LLM judges | X crossover [c5 / v05] | 9 / 6 (9 / 5 pairs) | +0.245 (+0.237 / +0.252) | +0.088 (+0.094 / +0.082) | **0.36** [0.12, 0.61] (0.40 / 0.33) | +0.50 |
|  |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02 E_dec] | 9 / 9 | +0.038 (+0.037 / +0.039) | +0.010 (+0.009 / +0.012) | **0.28** [0.10, 0.60] (0.23 / 0.32) | +0.17 |
| Mistral-Nemo judge (M), P(Yes) > 0.5 | judge | (a) paid content | (i) other LLM judges | S [c1] | 4 / 3 | +0.314 (+0.380 / +0.248) | +0.022 (+0.026 / +0.018) | **0.07** [-0.05, 0.20] (0.07 / 0.07) | +0.22 |
|  |  | (a) paid content | (ii) random-error runs | K [b02 kindc, scored answers only] | 4 / 3 | +0.475 (+0.534 / +0.415) | +0.082 (+0.092 / +0.072) | **0.17** [-0.05, 0.35] (0.17 / 0.17) | +0.57 |
|  |  | (b) refusal where it rejects right | (i) other LLM judges | X crossover [c5 / v05] | 2 / 3 (2 / 2 pairs) | +0.025 (+0.045 / +0.005) | +0.000 (+0.000 / +0.000) | **0.00** [0.00, 0.00] (0.00 / 0.00) n.i. | +0.00 |
|  |  | (b) refusal where it rejects right | (ii) random-error runs | D [b02 E_dec] | 4 / 3 | +0.047 (+0.070 / +0.023) | -0.031 (-0.035 / -0.027) | **-0.66** [-5.60, 4.09] (-0.49 / -1.17) n.i. | -0.04 |
|  |  | (c) answering where it pays wrong | (i) other LLM judges | X crossover [c5 / v05] | 4 / 1 (9 / 2 pairs) | +0.125 (+0.176 / +0.074) | +0.040 (+0.070 / +0.010) | **0.32** [-0.02, 0.72] (0.40 / 0.14) | +0.22 |
|  |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02 E_dec] | 4 / 3 | +0.031 (+0.041 / +0.020) | +0.014 (+0.020 / +0.008) | **0.46** [0.19, 1.03] (0.50 / 0.37) | +0.20 |
| Gemma-4-12B judge (G), P(Yes) > 0.5 | judge | (a) paid content | (i) other LLM judges | S [c1] | 4 / 3 | +0.324 (+0.353 / +0.295) | +0.005 (-0.000 / +0.011) | **0.02** [-0.11, 0.13] (-0.00 / 0.04) | +0.08 |
|  |  | (a) paid content | (ii) random-error runs | K [b02 kindc, scored answers only] | 3 / 3 | +0.545 (+0.572 / +0.517) | +0.093 (+0.018 / +0.167) | **0.17** [-0.01, 0.36] (0.03 / 0.32) | +0.62 |
|  |  | (b) refusal where it rejects right | (i) other LLM judges | X crossover [c5 / v05] | 4 / 3 (6 / 4 pairs) | +0.156 (+0.153 / +0.159) | +0.008 (+0.011 / +0.005) | **0.05** [-0.12, 0.25] (0.07 / 0.03) | +0.01 |
|  |  | (b) refusal where it rejects right | (ii) random-error runs | D [b02 E_dec] | 4 / 3 | +0.137 (+0.120 / +0.155) | -0.013 (-0.037 / +0.011) | **-0.10** [-0.33, 0.12] (-0.31 / 0.07) | -0.04 |
|  |  | (c) answering where it pays wrong | (i) other LLM judges | X crossover [c5 / v05] | 4 / 3 (9 / 5 pairs) | +0.280 (+0.284 / +0.276) | +0.078 (+0.073 / +0.083) | **0.28** [0.12, 0.45] (0.26 / 0.30) | +0.24 |
|  |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02 E_dec] | 4 / 3 | +0.175 (+0.161 / +0.190) | +0.067 (+0.052 / +0.082) | **0.38** [0.18, 0.72] (0.32 / 0.44) | +0.69 |
| OLMo-2-13B judge (O), P(Yes) > 0.5 | judge | (a) paid content | (i) other LLM judges | S [c1] | 4 / 3 | +0.295 (+0.253 / +0.338) | +0.055 (+0.048 / +0.062) | **0.19** [0.05, 0.33] (0.19 / 0.18) | +0.46 |
|  |  | (a) paid content | (ii) random-error runs | K [b02 kindc, scored answers only] | 4 / 1 | +0.564 (+0.562 / +0.565) | +0.268 (+0.129 / +0.408) | **0.48** [0.13, 0.62] (0.23 / 0.72) | +2.27 |
|  |  | (b) refusal where it rejects right | (i) other LLM judges | X crossover [c5 / v05] | 4 / 3 (6 / 4 pairs) | +0.246 (+0.246 / +0.245) | +0.012 (+0.011 / +0.012) | **0.05** [-0.14, 0.27] (0.05 / 0.05) | +0.03 |
|  |  | (b) refusal where it rejects right | (ii) random-error runs | D [b02 E_dec] | 4 / 3 | +0.255 (+0.281 / +0.228) | +0.034 (+0.036 / +0.033) | **0.13** [-0.11, 0.34] (0.13 / 0.14) | +0.31 |
|  |  | (c) answering where it pays wrong | (i) other LLM judges | X crossover [c5 / v05] | 4 / 3 (9 / 4 pairs) | +0.473 (+0.457 / +0.488) | +0.190 (+0.206 / +0.173) | **0.40** [0.18, 0.64] (0.45 / 0.35) | +0.17 |
|  |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02 E_dec] | 4 / 3 | +0.256 (+0.286 / +0.226) | +0.049 (+0.086 / +0.012) | **0.19** [-0.04, 0.43] (0.30 / 0.05) | +0.30 |
| **All LLM judges pooled** | | | | | | | | | |
| all judges, all cuts |  | (a) paid content | (i) other LLM judges | S [c1], all judge runs | 76 / 67 | +0.330 (+0.351 / +0.309) | +0.021 (+0.022 / +0.019) | **0.06** [0.00, 0.13] (0.06 / 0.06) | +0.09 |
|  |  | (a) paid content | (ii) random-error runs | K, all judge runs | 42 / 35 | +0.564 (+0.579 / +0.549) | +0.148 (+0.109 / +0.187) | **0.26** [0.19, 0.36] (0.19 / 0.34) | +0.69 |
| all judges, 35% cut |  | (a) paid content | (i) other LLM judges | S [c1] | 38 / 31 | +0.303 (+0.299 / +0.307) | +0.032 (+0.032 / +0.031) | **0.11** [-0.00, 0.22] (0.11 / 0.10) | +0.07 |
|  |  | (a) paid content | (ii) random-error runs | K | 16 / 14 | +0.625 (+0.591 / +0.659) | +0.227 (+0.172 / +0.282) | **0.36** [0.22, 0.51] (0.29 / 0.43) | +0.72 |
|  |  | (b) refusal where it rejects right | (i) other LLM judges | X, all 44 pairs [c5; home: b05] | 43 / 44 (22 / 22 pairs) | +0.904 (+0.764 / +1.044) | +0.241 (+0.211 / +0.271) | **0.27** [0.21, 0.33] (0.28 / 0.26) | +0.68 |
|  |  | (b) refusal where it rejects right | (ii) random-error runs | D [b02] | 42 / 44 | +0.509 (+0.427 / +0.592) | +0.119 (+0.110 / +0.128) | **0.23** [0.19, 0.34] (0.26 / 0.22) | +1.98 |
|  |  | (c) answering where it pays wrong | (i) other LLM judges | X, all NQ pairs [v05] | 17 / 17 (10 / 10 pairs) | +0.642 (+0.526 / +0.758) | +0.147 (+0.093 / +0.202) | **0.23** [0.09, 0.40] (0.18 / 0.27) | +0.21 |
|  |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02] | 20 / 17 | +0.363 (+0.338 / +0.387) | +0.058 (+0.013 / +0.103) | **0.16** [-0.07, 0.40] (0.04 / 0.26) | +0.26 |
| all judges, P(Yes) > 0.5 |  | (b) refusal where it rejects right | (i) other LLM judges | X, all pairs | 19 / 18 (11 / 8 pairs) | +0.143 (+0.150 / +0.135) | +0.007 (+0.005 / +0.008) | **0.05** [-0.13, 0.26] (0.03 / 0.06) | +0.01 |
|  |  | (c) answering where it pays wrong | (i) other LLM judges | X, all pairs | 21 / 13 (18 / 8 pairs) | +0.293 (+0.289 / +0.296) | +0.104 (+0.111 / +0.096) | **0.35** [0.17, 0.55] (0.38 / 0.32) | +0.30 |
| **Judge-specific refusal (X, 35% cut) split into between- and within-category parts [b05]** | | | | | | | | | |
| categories = PopQA relation type (PopQA pairs) |  | (b) refusal | (i) other LLM judges | between part = X raw - X within | 6 / 6 pairs | +0.123 (+0.137 / +0.110) | +0.205 (+0.183 / +0.228) | **1.66** [0.83, 4.69] (1.34 / 2.07) |  |
|  |  | (b) refusal | (i) other LLM judges | within part = X with category fixed effects | 6 / 6 pairs | +0.785 (+0.714 / +0.856) | +0.177 (+0.117 / +0.238) | **0.23** [0.10, 0.34] (0.16 / 0.28) |  |
| categories = answer type (5 classes) |  | (b) refusal | (i) other LLM judges | between part = X raw - X within | 22 / 22 pairs | +0.028 (+0.036 / +0.020) | +0.066 (+0.056 / +0.077) | **2.36** [0.40, 14.99] (1.57 / 3.76) n.i. |  |
|  |  | (b) refusal | (i) other LLM judges | within part = X with category fixed effects | 22 / 22 pairs | +0.876 (+0.728 / +1.024) | +0.175 (+0.155 / +0.194) | **0.20** [0.14, 0.26] (0.21 / 0.19) |  |
| categories = question-embedding cluster (8) |  | (b) refusal | (i) other LLM judges | between part = X raw - X within | 22 / 22 pairs | +0.051 (+0.055 / +0.047) | +0.087 (+0.073 / +0.100) | **1.71** [0.78, 6.27] (1.34 / 2.15) |  |
|  |  | (b) refusal | (i) other LLM judges | within part = X with category fixed effects | 22 / 22 pairs | +0.853 (+0.709 / +0.997) | +0.154 (+0.138 / +0.171) | **0.18** [0.12, 0.25] (0.19 / 0.17) |  |
| categories = TF-IDF cluster (16) |  | (b) refusal | (i) other LLM judges | between part = X raw - X within | 22 / 22 pairs | +0.053 (+0.070 / +0.036) | +0.082 (+0.058 / +0.106) | **1.55** [-7.32, 12.33] (0.83 / 2.97) n.i. |  |
|  |  | (b) refusal | (i) other LLM judges | within part = X with category fixed effects | 22 / 22 pairs | +0.851 (+0.694 / +1.008) | +0.159 (+0.153 / +0.165) | **0.19** [0.12, 0.24] (0.22 / 0.16) |  |

## Notes (failures, nulls and what drives the differences)

1. **Item-tied errors, new-question sets.** For hash35, fnH80 and plscrM the grader's own rule on new questions (hash table, coin, hash rule)
   cannot be learned, so those rows are null by construction and test the estimator: T 0.01, 0.04, -0.05 (CIs include 0). The nearest-neighbour
   proxies (new questions whose 10 nearest training questions are in the set more often than its base rate; c78: a majority) give T 0.16 (fnH80, c78 proxy) and 0.19 [0.03, 0.38]
   (plscrM), but the same D on proxies of 300 random training sets of the same size gives as much: observed vs placebo mean +0.053 vs +0.074 and
   +0.043 vs +0.055 (fnH80; rank p 0.57, 0.59), +0.069 vs +0.005 and +0.106 vs +0.066 (plscrM; rank p 0.22, 0.31) (`f17_proxy_null_out.txt`).
   So no item-tied error shows transfer beyond what any random set shows. The first proxy (all training neighbours, as c78) was confounded for
   plscrM (its set lies inside the questions with a right answer, so the proxy marked easier questions; `f16_probe_scr_out.txt`); the
   eligible-neighbour proxy (`f03b`) fixes that.
2. **Estimator floor of D.** On 20 random halves of the items (no relation to training), D gives +0.00 to +0.05 (refusal, home) and -0.01 to
   +0.04 (new) for the high-abstention arms, and +-0.01 for low-refusal arms (hash35, t50 judges) (`f17b_coin_null_out.txt`). Hence the t50
   refusal D rows and the uncorrected hash35 answering D (T 0.29 [0.13, 0.61] with a null-by-construction new effect of +0.010) are not
   interpretable; subtracting D on 5 placebo hash tables leaves +0.005 (T 0.19 [-0.07, 0.78]). Correcting the 35%-cut rows for the floor moves
   T by <= 0.05 except plrealM (0.26 -> about 0.21).
3. **Where existing estimators disagree** (both are given; all reproduced exactly from raw files):
   - Judge-specific refusal: X (only items where two judges disagree, symmetric) 0.27 vs D with the other judges' runs as reference 0.17 [0.10, 0.32]
     (all of the judge's rejections, regression over other judges' runs; it fails where runs do not overlap in refusal: Llama-OLMo runs refuse
     0-6% of OLMo-accepted right answers at home vs 17-33% for the other judges, so OLMo has no D(i)) vs L02's estimator 0.22 [0.19, 0.29].
   - All-pattern refusal: D 0.23 [0.19, 0.34] vs L02 (agreement-stratified log-OR shift from the single nearest reference configuration) 0.29
     [0.25, 0.34]; per 100 new questions D gives 1.98 and L02 3.7 (Llama: 2.95 vs 5.5 for the Qwen judge) because D extrapolates a regression
     fitted mostly on low-abstention reference runs, L02 uses one configuration at the judge's abstention.
   - Planted arms: b02's D (strict-right units, all random-error runs) vs c78's map (sure-right units, key + false-reject runs, local-linear in
     abstention): fnB80 0.77 vs 0.87; plrealM 0.26 vs 0.34; the scrambled copy measured on Mistral's rejections (c78's contrast) is negative in both
     (-0.19 / -0.43), driven by the Llama plscrM runs (0.76 abstention on new questions, beyond most reference runs; one of them in D's support).
   - Content vs random-error runs: K with scored answers only vs b02 as printed (unscored answers counted as unpaid): e.g. Qwen judge 35% cut
     0.32 / 0.54 vs 0.22 / 0.61. Own-judge runs' answers are scored less often than reference runs' answers for M, G, O (0.79-0.96 vs
     0.92-0.99; `f14_coverage_out.txt`), which biases b02's version down for those judges.
   - Judge-specific content: S (6.1%, c1 main; 0.06 [0.00, 0.13] here with 7 later runs) vs c1's DiD against key-trained runs 0.092 [0.026, 0.160].
4. **Not estimable.** 50% cut, reference (i): only the Qwen judge was trained there. Answering at the 35% cut: NQ-open only (on TriviaQA / PopQA no
   judge pair has >= 5 wrong untrained new-question answers paid by one judge only; v05). Granite (R): NQ-open only, 2 runs per student. hash35
   content vs random-error runs and long80 changed-answer form on Qwen: no run inside the reference range at home. Mistral at P(Yes) > 0.5:
   X = 0.000 on new questions in every draw (no refusal on the disagreement items), +0.025 at home. Content S needs all four judges' scores (R: five).
5. **Runs added after the stats / bothsigns caches** (t50 runs of M, G, O on PopQA and M on NQ, written Nov 15 21:02) are included; with
   them the t50 refusal crossover on new questions is +0.007 (c5 printed +0.021 without them).

## The same cells with other estimators
| Grader error | Tied to | Effect | Reference | Estimator | Runs q / l | Home: pooled (q / l) | New: pooled (q / l) | **Transfer new / home**: pooled [95% CI] (q / l) | Per 100 new q, pooled |
|---|---|---|---|---|---|---|---|---|---|
| **Same cells, other estimators (computed here from raw files with the same bootstrap unless marked)** | | | | | | | | | |
| Qwen3.5-9B judge (Q), 35% cut |  | (b) refusal where it rejects right | (i) other LLM judges | D with the other judges' runs as reference | 7 / 6 | +0.386 (+0.254 / +0.518) | +0.048 (+0.077 / +0.019) | **0.12** [-0.03, 0.32] (0.30 / 0.04) | +0.41 |
| Qwen3.5-9B judge (Q), 35% cut |  | (c) answering where it pays wrong | (i) other LLM judges | D with the other judges' runs as reference | 5 / 6 | +0.244 (+0.176 / +0.312) | -0.026 (-0.050 / -0.002) | **-0.11** [-0.50, 0.28] (-0.28 / -0.01) | -0.12 |
| Mistral-Nemo judge (M), 35% cut |  | (b) refusal where it rejects right | (i) other LLM judges | D with the other judges' runs as reference | 6 / 6 | +0.288 (+0.266 / +0.311) | +0.109 (+0.125 / +0.094) | **0.38** [0.19, 0.71] (0.47 / 0.30) | +1.59 |
| Mistral-Nemo judge (M), 35% cut |  | (c) answering where it pays wrong | (i) other LLM judges | D with the other judges' runs as reference | 5 / 3 | +0.305 (+0.350 / +0.260) | +0.136 (+0.085 / +0.187) | **0.45** [0.11, 1.20] (0.24 / 0.72) | +0.54 |
| Gemma-4-12B judge (G), 35% cut |  | (b) refusal where it rejects right | (i) other LLM judges | D with the other judges' runs as reference | 6 / 3 | +0.335 (+0.132 / +0.538) | +0.069 (+0.031 / +0.107) | **0.21** [0.06, 0.46] (0.24 / 0.20) | +1.03 |
| Gemma-4-12B judge (G), 35% cut |  | (c) answering where it pays wrong | (i) other LLM judges | D with the other judges' runs as reference | 3 / 3 | +0.244 (+0.160 / +0.329) | +0.098 (+0.072 / +0.125) | **0.40** [-0.09, 1.02] (0.45 / 0.38) | +0.30 |
| OLMo-2-13B judge (O), 35% cut |  | (b) refusal where it rejects right | (i) other LLM judges | D with the other judges' runs as reference | 5 / 0 | n/a (+0.333 / n/a) | n/a (+0.059 / n/a) | **n/a** (0.18 / n/a) | n/a |
| OLMo-2-13B judge (O), 35% cut |  | (c) answering where it pays wrong | (i) other LLM judges | D with the other judges' runs as reference | 3 / 0 | n/a (+0.326 / n/a) | n/a (+0.093 / n/a) | **n/a** (0.29 / n/a) | n/a |
| Granite-3.3-8B judge (R), 35% cut |  | (b) refusal where it rejects right | (i) other LLM judges | D with the other judges' runs as reference | 2 / 2 | +0.346 (+0.138 / +0.554) | -0.034 (-0.032 / -0.037) | **-0.10** [-0.36, 0.20] (-0.23 / -0.07) | -0.35 |
| Granite-3.3-8B judge (R), 35% cut |  | (c) answering where it pays wrong | (i) other LLM judges | D with the other judges' runs as reference | 2 / 2 | +0.275 (+0.232 / +0.318) | -0.001 (-0.074 / +0.072) | **-0.00** [-0.30, 0.29] (-0.32 / 0.23) | -0.00 |
| all judges, 35% cut |  | (b) refusal where it rejects right | (i) other LLM judges | D with the other judges' runs as reference | 26 / 17 | +0.344 (+0.235 / +0.453) | +0.060 (+0.066 / +0.055) | **0.17** [0.10, 0.32] (0.28 / 0.12) | +0.81 |
| all judges, 35% cut |  | (c) answering where it pays wrong | (i) other LLM judges | D with the other judges' runs as reference | 18 / 14 | +0.279 (+0.253 / +0.305) | +0.053 (+0.029 / +0.076) | **0.19** [0.03, 0.40] (0.12 / 0.25) | +0.20 |
| all judges, all cuts |  | (b) refusal where it rejects right | (i) other LLM judges | D with the other judges' runs as reference | 47 / 32 | +0.195 (+0.172 / +0.217) | +0.029 (+0.035 / +0.023) | **0.15** [-0.12, 0.28] (0.20 / 0.11) | +0.46 |
| all judges, all cuts |  | (c) answering where it pays wrong | (i) other LLM judges | D with the other judges' runs as reference | 35 / 25 | +0.177 (+0.149 / +0.206) | +0.036 (+0.022 / +0.049) | **0.20** [0.05, 0.39] (0.15 / 0.24) | +0.20 |
| hash35 |  | (c) answering where it pays wrong | (ii) random-error runs | D [b02 E_dec], no placebo-table correction | 6 / 6 | +0.036 (+0.042 / +0.031) | +0.010 (+0.013 / +0.008) | **0.29** [0.13, 0.61] (0.31 / 0.26) | +0.10 |
| long80 |  | (a) paid content = form | (ii) random-error runs | K on changed answers only [b02 kindc] | 0 / 2 | n/a (n/a / +0.845) | n/a (n/a / +0.564) | **n/a** (n/a / 0.67) | n/a |
| fnH80 |  | (b) refusal where it rejects right | (ii) random-error runs | D; new set = nearest eligible neighbours proxy | 2 / 3 | +0.308 (+0.257 / +0.359) | -0.004 (-0.057 / +0.048) | **-0.01** [-0.20, 0.33] (-0.22 / 0.13) | +0.10 |
| plscrM |  | (b) refusal where it rejects right | (ii) random-error runs | D; new set = Mistral's own rejections [c78 contrast set] | 4 / 1 | +0.443 (+0.380 / +0.506) | -0.085 (-0.015 / -0.156) | **-0.19** [-0.31, 0.02] (-0.04 / -0.31) | -2.41 |
| **L02 / r02 'located refusals per 100 questions' estimator (re-derived in f18, home side added, crossed bootstrap f19; 35% cut)** | | | | | | | | | |
| Qwen3.5-9B judge (Q), 35% cut | | (b) refusal where it rejects right | (i) other LLM judges | excess refusals per 100 questions (MH log-OR, agreement strata) | 18 / 18 | 11.3 (7.7 / 14.9) | 1.8 (1.3 / 2.2) | **0.16** [0.10, 0.24] (0.17 / 0.15) | 1.8 |
| Mistral-Nemo judge (M), 35% cut | | (b) refusal where it rejects right | (i) other LLM judges | excess refusals per 100 questions (MH log-OR, agreement strata) | 8 / 8 | 9.1 (4.6 / 13.6) | 2.1 (2.0 / 2.3) | **0.23** [0.15, 0.38] (0.43 / 0.17) | 2.1 |
| Gemma-4-12B judge (G), 35% cut | | (b) refusal where it rejects right | (i) other LLM judges | excess refusals per 100 questions (MH log-OR, agreement strata) | 8 / 8 | 11.4 (6.1 / 16.7) | 3.2 (1.4 / 4.9) | **0.28** [0.20, 0.37] (0.23 / 0.29) | 3.2 |
| OLMo-2-13B judge (O), 35% cut | | (b) refusal where it rejects right | (i) other LLM judges | excess refusals per 100 questions (MH log-OR, agreement strata) | 7 / 8 | 10.0 (6.6 / 13.4) | 2.2 (1.2 / 3.3) | **0.23** [0.15, 0.34] (0.18 / 0.25) | 2.2 |
| all four judges, 35% cut | | (b) refusal where it rejects right | (i) other LLM judges | excess refusals per 100 questions (MH log-OR, agreement strata) | 41 / 42 | 10.4 (6.3 / 14.6) | 2.3 (1.5 / 3.2) | **0.22** [0.19, 0.29] (0.23 / 0.22) | 2.3 |
| Qwen3.5-9B judge (Q), 35% cut | | (b) refusal where it rejects right | (ii) nearest key + random-noise configuration | excess refusals per 100 questions (MH log-OR, agreement strata) | 18 / 18 | 13.7 (9.3 / 18.1) | 3.7 (1.9 / 5.5) | **0.27** [0.20, 0.37] (0.20 / 0.31) | 3.7 |
| Mistral-Nemo judge (M), 35% cut | | (b) refusal where it rejects right | (ii) nearest key + random-noise configuration | excess refusals per 100 questions (MH log-OR, agreement strata) | 8 / 8 | 11.5 (6.6 / 16.5) | 3.1 (2.6 / 3.7) | **0.27** [0.19, 0.40] (0.40 / 0.22) | 3.1 |
| Gemma-4-12B judge (G), 35% cut | | (b) refusal where it rejects right | (ii) nearest key + random-noise configuration | excess refusals per 100 questions (MH log-OR, agreement strata) | 8 / 8 | 14.2 (7.8 / 20.6) | 5.1 (1.7 / 8.6) | **0.36** [0.28, 0.45] (0.22 / 0.42) | 5.1 |
| OLMo-2-13B judge (O), 35% cut | | (b) refusal where it rejects right | (ii) nearest key + random-noise configuration | excess refusals per 100 questions (MH log-OR, agreement strata) | 7 / 8 | 10.8 (7.6 / 14.1) | 2.6 (1.6 / 3.6) | **0.24** [0.14, 0.35] (0.20 / 0.26) | 2.6 |
| all four judges, 35% cut | | (b) refusal where it rejects right | (ii) nearest key + random-noise configuration | excess refusals per 100 questions (MH log-OR, agreement strata) | 41 / 42 | 12.6 (7.8 / 17.3) | 3.7 (1.9 / 5.4) | **0.29** [0.25, 0.34] (0.25 / 0.31) | 3.7 |
| **Existing printed estimators for the same cells (point estimates reproduced exactly from raw files, f11-f15; intervals as printed by the source)** | | | | | | | | | |
| Qwen3.5-9B judge (Q), 35% cut | | (a) paid content | (ii) random-error runs | b02 as printed: E_kindc (unscored answers counted as unpaid) | 18 / 18 | q +0.559 / l +0.563 | q +0.121 / l +0.342 | q 0.22 [0.01, 0.46] / l 0.61 [0.19, 1.04] (no pooled value printed) | |
| Qwen3.5-9B judge (Q), 50% cut | | (a) paid content | (ii) random-error runs | b02 as printed: E_kindc (unscored answers counted as unpaid) | 18 / 18 | q +0.615 / l +0.527 | q +0.063 / l +0.150 | q 0.10 [-0.02, 0.25] / l 0.28 [0.06, 0.50] (no pooled value printed) | |
| Qwen3.5-9B judge (Q), P(Yes) > 0.5 | | (a) paid content | (ii) random-error runs | b02 as printed: E_kindc (unscored answers counted as unpaid) | 9 / 9 | q +0.435 / l +0.401 | q +0.007 / l +0.072 | q 0.02 [-0.35, 0.35] / l 0.18 [-0.04, 0.47] (no pooled value printed) | |
| Mistral-Nemo judge (M), 35% cut | | (a) paid content | (ii) random-error runs | b02 as printed: E_kindc (unscored answers counted as unpaid) | 8 / 8 | q +0.572 / l +0.711 | q +0.106 / l +0.231 | q 0.18 [-0.07, 0.51] / l 0.32 [-0.01, 0.61] (no pooled value printed) | |
| Mistral-Nemo judge (M), P(Yes) > 0.5 | | (a) paid content | (ii) random-error runs | b02 as printed: E_kindc (unscored answers counted as unpaid) | 1 / 1 | q +0.520 / l +0.442 | q +0.053 / l +0.079 | q 0.10 [-0.23, 0.45] / l 0.18 [-0.06, 0.68] (no pooled value printed) | |
| Gemma-4-12B judge (G), 35% cut | | (a) paid content | (ii) random-error runs | b02 as printed: E_kindc (unscored answers counted as unpaid) | 8 / 8 | q +0.442 / l +0.511 | q +0.136 / l +0.146 | q 0.31 [0.07, 0.93] / l 0.28 [-0.29, 0.74] (no pooled value printed) | |
| Gemma-4-12B judge (G), P(Yes) > 0.5 | | (a) paid content | (ii) random-error runs | b02 as printed: E_kindc (unscored answers counted as unpaid) | 3 / 3 | q +0.382 / l +0.368 | q -0.068 / l +0.140 | q -0.18 [-0.63, 0.23] / l 0.38 [0.11, 0.77] (no pooled value printed) | |
| OLMo-2-13B judge (O), 35% cut | | (a) paid content | (ii) random-error runs | b02 as printed: E_kindc (unscored answers counted as unpaid) | 8 / 8 | q +0.559 / l +0.556 | q +0.209 / l +0.456 | q 0.37 [0.12, 0.64] / l 0.82 [0.33, 0.99] (no pooled value printed) | |
| OLMo-2-13B judge (O), P(Yes) > 0.5 | | (a) paid content | (ii) random-error runs | b02 as printed: E_kindc (unscored answers counted as unpaid) | 3 / 3 | q +0.460 / l +0.307 | q +0.109 / l +0.017 | q 0.24 [0.04, 0.48] / l 0.06 [-0.17, 0.60] (no pooled value printed) | |
| Granite-3.3-8B judge (R), 35% cut | | (a) paid content | (ii) random-error runs | b02 as printed: E_kindc (unscored answers counted as unpaid) | 2 / 2 | q +0.414 / l +0.395 | q +0.078 / l +0.181 | q 0.19 [-0.27, 0.79] / l 0.46 [-0.02, 1.19] (no pooled value printed) | |
| hash35 | | (a) paid content | (ii) random-error runs | b02 as printed: E_kindc (unscored answers counted as unpaid) | 6 / 6 | q n/a / l +0.609 | q +0.009 / l +0.002 | q n/a / l 0.00 [-0.18, 0.11] (no pooled value printed) | |
| all judges, all cuts | | (a) paid content | (i) other LLM judges, minus the same contrast in key-trained runs | c1 DiD | 74 / 67 | +0.334 (+0.348 / +0.320) | +0.031 (+0.032 / +0.029) | **0.092** [0.026, 0.160] (0.093 / 0.091) | |
| fnB80 | | (b) refusal where it rejects right | (ii) key + random false-reject runs, local-linear in abstention | c78 map (sure-right units) | 3 / 3 | +0.446 (+0.359 / +0.534) | +0.394 (+0.282 / +0.505) | **0.87** [0.72, 1.09] (0.79 / 0.95) | |
| fnH80 | | (b) refusal where it rejects right | (ii) same as above | c78 map, new set = NN proxy | 2 / 3 | +0.244 (+0.195 / +0.293) | +0.031 (+0.040 / +0.023) | **0.14** [-0.16, 0.52] (0.20 / 0.08) | |
| plrealM | | (b) refusal where it rejects right | (ii) same as above | c78 map (home = own set, new = Mistral's rejections; ratio computed here, f15) | 4 / 4 | +0.413 (+0.388 / +0.437) | +0.141 (+0.126 / +0.156) | **0.34** (0.32 / 0.36), no interval | |
| plscrM | | (b) refusal where it rejects right | (ii) same as above | c78 map, new set = Mistral's rejections (ratio computed here, f15) | 4 / 3 | +0.413 (+0.285 / +0.541) | -0.176 (-0.005 / -0.348) | **-0.43** (-0.02 / -0.64), no interval | |
| judgeMa35 | | (b) refusal where it rejects right | (ii) same as above | c78 map, home and new = Mistral's rejections (f15) | 3 / 3 | +0.317 (+0.252 / +0.381) | +0.139 (+0.117 / +0.161) | **0.44** (0.46 / 0.42), no interval | |

## Verification of the reused cells (own code, raw files only)

| Existing cell (source) | Printed | Recomputed here | Script -> output |
|---|---|---|---|
| Caches: run sets, answers, codes, greedy answers, Sonnet levels, P(Yes) (bothsigns b01) | - | identical on all common runs (0 mismatches); 7 newer t50 runs added | f02_check_cache.py -> f02_check_cache_out.txt |
| Crossover X, new, 35% cut (stats/c5) | +0.211 / +0.271 / +0.241, 43 of 44 > 0 | same | f11_verify_X.py -> f11_verify_X_out.txt |
| Crossover X, home (fix_boundary/b05) | +0.764 / +1.044 / +0.904 | same | f11 |
| Crossover X, t50 new (c5 robustness) | +0.032 / +0.010 / +0.021 | same (c5 run set) | f11 |
| FA crossover, new, 35% cut (verify_map/v05) | -0.093 / -0.202 / -0.147 | same | f11 |
| FA crossover, t50 (fix_bothdir/b03) | -0.114 / -0.096 / -0.105 | same | f11 |
| Content S home / new (stats/c1 main; DiD) | +0.344/+0.019, +0.313/+0.021; DiD +0.348/+0.032, +0.320/+0.029 | same (rows 74/69, 67/64) | f12_verify_S.py -> f12_verify_S_out.txt |
| E_dec / E_kind / E_kindc, every arm of the table (bothsigns/b02) | b02b_table_out.txt | same, after (a) adding the set-file-neutral random graders on PopQA to the reference pool and (b) rebuilding b02's norm-only judge lookup | f13_verify_D.py -> f13_verify_D_out.txt |
| Planted arms (stats/c78): fnB80, fnH80, plrealM, plscrM | e.g. fnB80 home +0.359/+0.534, new +0.282/+0.505 | same | f15_verify_c78.py -> f15_verify_c78_out.txt |
| Between / within category split (fix_boundary/b05) | prop between 1.66, within 0.23; at5 2.36 / 0.20; emb8 1.71 / 0.18; tf16 1.55 / 0.19 | same | f20_engine.py (B cells) |
| Located refusals per 100 new (regimes/r02 -> fix_ledger/L02, 35% cut) | Q 1.9 / 1.3, M 2.6 / 2.0, G 1.7 / 1.4, O 1.6 / 1.2 (Qwen); 5.5 / 2.2, 3.7 / 2.3, 8.6 / 4.9, 3.6 / 3.3 (Llama) | same, same placebo configurations | f18_verify_L02.py -> f18_verify_L02_out.txt |

No printed cell failed to reproduce. Caveats found while verifying: b02's content cells count unscored answers as unpaid (note 3); c78's
"hidden half carries a little" (+0.031) is within the placebo-set null (note 1); D has a small floor at low refusal (note 2).
