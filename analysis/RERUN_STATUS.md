# Rerun status of the analysis scripts

Each script below was run from a fresh copy of this repository at a different location, with the run outputs unpacked. The main caches
(`analysis/stats/cache/`) were rebuilt from the run outputs first; the scripts that write the paper's tables ran last. The files each script
wrote were compared byte by byte with the files in this repository.

- scripts: 297; ran to the end: 259; need arguments (see the docstring): 35; need data that is not released: 1; stopped with an error: 0; stopped at the time limit: 0; not run because they need a GPU or a network service: 2
- files written again: 375; identical to the released file: 368; different: 7 (listed at the end with the reason)
- figures drawn again: 4 (PDF files carry their creation time, so they are not compared byte by byte)

| script | status | seconds |
|---|---|---|
| `analysis/answers/a0_cache.py` | needs arguments | 0 |
| `analysis/answers/a14_splits.py` | ok | 1 |
| `analysis/answers/a19_curve.py` | ok | 247 |
| `analysis/answers/a19c_perdataset.py` | ok | 0 |
| `analysis/answers/a1_jcache.py` | needs arguments | 0 |
| `analysis/answers/a3_selection.py` | ok | 207 |
| `analysis/answers/a4_seeds.py` | ok | 21 |
| `analysis/answers/a6_sonnet.py` | ok | 65 |
| `analysis/answers/au.py` | ok | 0 |
| `analysis/arch_falsereject/c04_tradeoff.py` | ok | 84 |
| `analysis/arch_falsereject/c05_specific_counts.py` | ok | 1 |
| `analysis/arch_falsereject/t01_thresholds.py` | ok | 0 |
| `analysis/arch_forecast/v02_law_checks.py` | ok | 20 |
| `analysis/arch_forecast/v03b_visibility_a35.py` | ok | 9 |
| `analysis/arch_v4a/a01_pool_oor.py` | ok | 1 |
| `analysis/arch_v4b/a02_credit_check.py` | ok | 1 |
| `analysis/arch_v4b/a03_fa_mirror.py` | ok | 11 |
| `analysis/arch_v4b/a04_known_refusals.py` | ok | 0 |
| `analysis/arch_v4b/a05_onepass.py` | ok | 0 |
| `analysis/arch_when/w04b_typefp_robust.py` | ok | 0 |
| `analysis/arch_when/w07_firstpass.py` | ok | 1 |
| `analysis/bothsigns/b00_common.py` | ok | 0 |
| `analysis/bothsigns/b01_cache.py` | needs arguments | 0 |
| `analysis/bothsigns/b01b_judgelookup.py` | ok | 3 |
| `analysis/bothsigns/b02_effects.py` | ok | 153 |
| `analysis/bothsigns/b02b_table.py` | ok | 7 |
| `analysis/bothsigns/b03_readability.py` | needs arguments | 1 |
| `analysis/bothsigns/b04_relation.py` | ok | 179 |
| `analysis/bothsigns/b05_details.py` | ok | 30 |
| `analysis/dose/d00_common.py` | ok | 0 |
| `analysis/dose/d01_cache.py` | needs arguments | 0 |
| `analysis/dose/d01b_capcache.py` | ok | 3 |
| `analysis/dose/d03_predictors.py` | needs arguments | 1 |
| `analysis/dose/d04_dose.py` | needs arguments | 1 |
| `analysis/dose/d04_lib.py` | ok | 0 |
| `analysis/dose/d04b_tables.py` | ok | 1 |
| `analysis/dose/d05_place.py` | ok | 2 |
| `analysis/dose/d06_fig.py` | ok | 1 |
| `analysis/dynamics/d02_order_check.py` | ok | 2 |
| `analysis/dynamics/d04_jump.py` | ok | 2 |
| `analysis/dynamics/d06_final.py` | ok | 74 |
| `analysis/dynamics/d07_hash.py` | ok | 33 |
| `analysis/dynamics/d07b_hash_support.py` | ok | 0 |
| `analysis/dynamics/d09_visit_collapse.py` | ok | 1 |
| `analysis/dynamics/d10_visit_table.py` | ok | 1 |
| `analysis/dynamics/d13_fk00.py` | ok | 1 |
| `analysis/dynamics/d15_judges.py` | ok | 2 |
| `analysis/dynamics/d17_diversity.py` | ok | 4 |
| `analysis/dynamics/d21_fig_visits.py` | ok | 1 |
| `analysis/dynamics/dyn.py` | ok | 0 |
| `analysis/extra/apitest/frontier_A.py` | not run (needs a GPU, a network service or a subprocess) |  |
| `analysis/extra/apitest/frontier_H.py` | not run (needs a GPU, a network service or a subprocess) |  |
| `analysis/extra/better/build_vote_keys.py` | ok | 5 |
| `analysis/extra/common/rc.py` | ok | 0 |
| `analysis/extra/confirm/thresholds.py` | ok | 4 |
| `analysis/extra/final/build_fn_files.py` | ok | 0 |
| `analysis/extra/final/build_nf_queue.py` | needs arguments | 0 |
| `analysis/extra/final/build_scramble.py` | ok | 4 |
| `analysis/extra/final/fn_score.py` | ok | 10 |
| `analysis/extra/final/scramble_score.py` | ok | 22 |
| `analysis/extra/final/thresholds_acc.py` | ok | 5 |
| `analysis/extra/forced/f01_forced.py` | ok | 24 |
| `analysis/extra/frontier/build_pairs.py` | ok | 0 |
| `analysis/extra/frontier/check.py` | ok | 0 |
| `analysis/extra/human_check/score_human_check.py` | ok | 0 |
| `analysis/extra/story/B_refusal_map/bm.py` | ok | 0 |
| `analysis/extra/story/B_refusal_map/bs.py` | ok | 0 |
| `analysis/fam_fix/e13_probe_crossover.py` | ok | 63 |
| `analysis/fam_fix/e15_count_dose.py` | ok | 10 |
| `analysis/fam_fix/e16_caricature.py` | ok | 3 |
| `analysis/fam_fix/e17_mirror.py` | ok | 0 |
| `analysis/fam_fix/e5_capitals.py` | ok | 91 |
| `analysis/fam_fix/e5_capitals_single.py` | ok | 12 |
| `analysis/fam_fix/e5b_single_alias.py` | ok | 1 |
| `analysis/fam_fix/e67_abstention.py` | ok | 7 |
| `analysis/fam_fix/e7_capitals.py` | ok | 2 |
| `analysis/fam_fix/e8v_forecast.py` | ok | 0 |
| `analysis/fam_fix/e8v_score.py` | needs arguments | 0 |
| `analysis/fam_fix/e9b_forced.py` | ok | 0 |
| `analysis/fam_ref/r05_audit_matched.py` | ok | 6 |
| `analysis/fam_ref/r06_reallocation.py` | ok | 1 |
| `analysis/fix_ladder/ladder_score.py` | needs arguments | 0 |
| `analysis/fix_ladder/ladder_score_acc.py` | needs arguments | 0 |
| `analysis/fix_ledger/L01_collapse.py` | ok | 25 |
| `analysis/fix_ledger/L02_magnitudes.py` | ok | 29 |
| `analysis/fix_ledger/L03_seeds.py` | ok | 0 |
| `analysis/fix_ledger/L04_law_nocollapse.py` | ok | 0 |
| `analysis/fix_ledger/L05_other_families.py` | ok | 0 |
| `analysis/fix_planted/p04_build_qacc.py` | ok | 2 |
| `analysis/fix_planted/p05_score_qacc.py` | ok | 1 |
| `analysis/fix_sc/f0_samples.py` | ok | 0 |
| `analysis/fix_sc/f1_scm.py` | ok | 82 |
| `analysis/fix_sc/f1b_checks.py` | ok | 0 |
| `analysis/fix_sc/f3_table.py` | ok | 1 |
| `analysis/general/g00_inventory.py` | ok | 1 |
| `analysis/general/g01_cache.py` | ok | 341 |
| `analysis/general/g02_scores.py` | ok | 85 |
| `analysis/general/g03_leniency.py` | ok | 24 |
| `analysis/general/g04_hash.py` | ok | 52 |
| `analysis/general/g05_ref_untrained.py` | ok | 16 |
| `analysis/general/g07_crossover.py` | ok | 8 |
| `analysis/general/g08_lift.py` | ok | 15 |
| `analysis/general/g09_readable.py` | ok | 58 |
| `analysis/general/g11_misc.py` | ok | 54 |
| `analysis/general/gcommon.py` | ok | 0 |
| `analysis/grpo/g00_configs.py` | ok | 0 |
| `analysis/grpo/g01_base.py` | needs arguments | 0 |
| `analysis/grpo/g02_theory.py` | ok | 13 |
| `analysis/grpo/g03_runs.py` | needs arguments | 0 |
| `analysis/grpo/g05_model.py` | needs arguments | 0 |
| `analysis/grpo/g06_eval.py` | ok | 4 |
| `analysis/grpo/g07_counterfactual.py` | needs arguments | 0 |
| `analysis/grpo/g08_credit.py` | ok | 1 |
| `analysis/grpo/g09_kappa.py` | needs arguments | 0 |
| `analysis/grpo/g11_type.py` | needs arguments | 0 |
| `analysis/judges/j0_cache.py` | ok | 0 |
| `analysis/judges/j10_specific.py` | ok | 1 |
| `analysis/judges/j11_frontier.py` | ok | 0 |
| `analysis/judges/j13_wrongspec.py` | ok | 0 |
| `analysis/judges/j14_did.py` | ok | 20 |
| `analysis/judges/j14b_cells.py` | ok | 0 |
| `analysis/judges/j14c_excess.py` | ok | 2 |
| `analysis/judges/j17_beta.py` | ok | 0 |
| `analysis/judges/j1_decomp.py` | ok | 2 |
| `analysis/judges/j1b_summary.py` | ok | 0 |
| `analysis/judges/j2_props.py` | ok | 3 |
| `analysis/judges/j3_predict.py` | ok | 1 |
| `analysis/judges/j4_agree.py` | ok | 0 |
| `analysis/knowledge/k0_feat.py` | ok | 0 |
| `analysis/knowledge/k10_explain.py` | ok | 57 |
| `analysis/knowledge/k11_home_abroad.py` | ok | 47 |
| `analysis/knowledge/k13_robust.py` | needs arguments | 1 |
| `analysis/knowledge/k14_splits.py` | ok | 1 |
| `analysis/knowledge/k16_confirm2.py` | ok | 4 |
| `analysis/knowledge/k17_decodable.py` | ok | 13 |
| `analysis/knowledge/k19_cells.py` | ok | 1 |
| `analysis/knowledge/k20_confirm_bend.py` | ok | 0 |
| `analysis/knowledge/k2_decomp.py` | ok | 0 |
| `analysis/knowledge/k2b_known.py` | ok | 0 |
| `analysis/knowledge/k6_forecast.py` | ok | 36 |
| `analysis/knowledge/k6b_identity.py` | ok | 1 |
| `analysis/knowledge/k8_confirm.py` | ok | 2 |
| `analysis/knowledge/k9_verdict_vs_probe.py` | ok | 24 |
| `analysis/long_answers/r2_long80.py` | ok | 2 |
| `analysis/meta/m00_configs.py` | ok | 0 |
| `analysis/meta/m01_build.py` | needs arguments | 0 |
| `analysis/meta/m02_table.py` | ok | 19 |
| `analysis/meta/m03_law.py` | ok | 2 |
| `analysis/meta/m04_premium.py` | ok | 2 |
| `analysis/meta/m05_onecurve.py` | ok | 2 |
| `analysis/meta/m06_hack.py` | ok | 3 |
| `analysis/meta/m07_refusal.py` | ok | 3 |
| `analysis/meta/m07b_refusal_quad.py` | ok | 4 |
| `analysis/meta/m08_predict.py` | ok | 86 |
| `analysis/meta/m10_drive.py` | ok | 11 |
| `analysis/meta/m11_drive_law.py` | ok | 1 |
| `analysis/meta/m12_selectivity.py` | ok | 4 |
| `analysis/meta/m13_transfer_law.py` | ok | 16 |
| `analysis/meta/m14_keynumbers.py` | ok | 3 |
| `analysis/meta/m15_audit_check.py` | ok | 32 |
| `analysis/meta/m16_true_hack.py` | ok | 60 |
| `analysis/meta/m17_hidden_vis.py` | ok | 12 |
| `analysis/meta/m18_fa_alignment.py` | ok | 9 |
| `analysis/regimes/r02_costs.py` | ok | 43 |
| `analysis/regimes/r02c_frontier.py` | ok | 4 |
| `analysis/regimes/r03a_readable.py` | ok | 47 |
| `analysis/regimes/r03b_travel.py` | ok | 48 |
| `analysis/regimes/r04_visits.py` | ok | 5 |
| `analysis/regimes/rg.py` | ok | 0 |
| `analysis/rev_F1/f1_nokl.py` | ok | 1 |
| `analysis/rev_F2/f1_checks.py` | ok | 9 |
| `analysis/rev_F2/f1b_checks.py` | ok | 13 |
| `analysis/rev_F2/f2_nokl.py` | ok | 9 |
| `analysis/rev_F3/z1_zero_param.py` | ok | 0 |
| `analysis/rev_F3/z3_oppoint.py` | ok | 2 |
| `analysis/rev_F4/r01_nokl.py` | ok | 4 |
| `analysis/rev_F5/f1_c5_repro.py` | ok | 16 |
| `analysis/rev_F5/f2_magnitude.py` | ok | 12 |
| `analysis/rev_F5/f3_popularity.py` | ok | 2 |
| `analysis/rev_F6/f1_crossover_reltype.py` | ok | 0 |
| `analysis/rev_F6/f2_popularity.py` | ok | 0 |
| `analysis/rev_R1/r1_crossover_split.py` | ok | 0 |
| `analysis/rev_R1/r3b_when_audited.py` | ok | 17 |
| `analysis/rev_R3/r1_capitals_v1v2.py` | ok | 1 |
| `analysis/rev_R3/r2_known_refusals.py` | ok | 0 |
| `analysis/rev_R4/r4_fa_crossover.py` | ok | 10 |
| `analysis/rev_R5/r5_c6_regress.py` | ok | 1 |
| `analysis/rev_R5/r5_kl.py` | ok | 3 |
| `analysis/rev_R5/r5_onepass.py` | ok | 0 |
| `analysis/rev_R6/k1_popqa_types.py` | ok | 0 |
| `analysis/selfcons/s0_build.py` | ok | 0 |
| `analysis/selfcons/s1_probe.py` | ok | 2 |
| `analysis/selfcons/s2_riskcov.py` | ok | 5 |
| `analysis/selfcons/s3_overlap.py` | ok | 1 |
| `analysis/selfcons/s4_scm.py` | ok | 2 |
| `analysis/selfcons/s5_home.py` | ok | 1 |
| `analysis/selfcons/s6_rate.py` | ok | 0 |
| `analysis/selfcons/s7_cells.py` | ok | 0 |
| `analysis/selfcons/s9_compose.py` | ok | 0 |
| `analysis/selfcons/sc_lib.py` | ok | 0 |
| `analysis/selfcons/sc_map.py` | ok | 0 |
| `analysis/stats/c10_capitals.py` | ok | 4 |
| `analysis/stats/c1_content.py` | ok | 74 |
| `analysis/stats/c1b_neardup.py` | ok | 11 |
| `analysis/stats/c2_visits.py` | ok | 32 |
| `analysis/stats/c3_noise.py` | ok | 12 |
| `analysis/stats/c4_law.py` | ok | 20 |
| `analysis/stats/c5_crossover.py` | ok | 26 |
| `analysis/stats/c6_readable.py` | ok | 86 |
| `analysis/stats/c78_causal.py` | ok | 18 |
| `analysis/stats/c9_cluster.py` | ok | 11 |
| `analysis/stats/s00_common.py` | ok | 0 |
| `analysis/stats/s01_cache.py` | ok | 105 |
| `analysis/stats/s02_judgefinal.py` | needs arguments | 0 |
| `analysis/stats/s03_probes.py` | needs arguments | 1 |
| `analysis/stats/s99_summary.py` | ok | 1 |
| `analysis/supp_final/aud.py` | ok | 0 |
| `analysis/supp_final/r2_long80.py` | ok | 2 |
| `analysis/supp_final/s14_sf_step0.py` | needs data that is not released | 1 |
| `analysis/supp_final/s16_fix_returns.py` | ok | 2 |
| `analysis/supp_final/s18_fix_ceiling.py` | ok | 48 |
| `analysis/supp_final/s18b_checks.py` | ok | 6 |
| `analysis/supp_final/s1_judge_knows.py` | ok | 1 |
| `analysis/supp_final/s20_fix_confirm.py` | ok | 12 |
| `analysis/supp_final/s20b_check_raw.py` | ok | 4 |
| `analysis/supp_final/s21_cut_and_calibration.py` | ok | 5 |
| `analysis/supp_final/s22_transitions.py` | ok | 4 |
| `analysis/supp_final/s22b_transition_checks.py` | ok | 65 |
| `analysis/supp_final/s23_small_checks.py` | ok | 3 |
| `analysis/supp_final/s3_planted_counts.py` | ok | 10 |
| `analysis/supp_final/s4_sc_reference.py` | ok | 7 |
| `analysis/supp_final/s5_return_by_cut.py` | ok | 2 |
| `analysis/supp_final/s6_jury_offline.py` | ok | 2 |
| `analysis/supp_final/s7_kinds_heldout.py` | ok | 1 |
| `analysis/supp_final/s8_fresh_kinds.py` | ok | 13 |
| `analysis/supp_final/s9_dose_planted.py` | ok | 11 |
| `analysis/verify_grpo/v00_cfg.py` | ok | 2 |
| `analysis/verify_grpo/v01_base.py` | needs arguments | 0 |
| `analysis/verify_grpo/v02_alpha.py` | needs arguments | 0 |
| `analysis/verify_grpo/v04_solve.py` | needs arguments | 0 |
| `analysis/verify_grpo/v05_eval.py` | ok | 11 |
| `analysis/verify_grpo/v06_cf.py` | needs arguments | 0 |
| `analysis/verify_grpo/v07_force.py` | ok | 132 |
| `analysis/verify_grpo/v08_visits.py` | ok | 0 |
| `analysis/verify_grpo/v09_drgrpo.py` | ok | 1 |
| `analysis/verify_map/v00_common.py` | ok | 0 |
| `analysis/verify_map/v01_cache.py` | needs arguments | 1 |
| `analysis/verify_map/v02_probe.py` | needs arguments | 1 |
| `analysis/verify_map/v02b_predictors.py` | needs arguments | 2 |
| `analysis/verify_map/v04_dose.py` | ok | 8 |
| `analysis/verify_map/v05_crossover.py` | ok | 11 |
| `analysis/verify_map/v06_homenew.py` | ok | 6 |
| `analysis/verify_map/v07_spec.py` | ok | 16 |
| `analysis/verify_map/v08_figure.py` | needs arguments | 0 |
| `analysis/verify_map/v10_t50.py` | ok | 0 |
| `analysis/verify_map/v11_plot.py` | ok | 1 |
| `analysis/verify_rate/v0_configs.py` | ok | 0 |
| `analysis/verify_rate/v10_recipe.py` | ok | 1 |
| `analysis/verify_rate/v1_build.py` | needs arguments | 0 |
| `analysis/verify_rate/v2_table.py` | ok | 1 |
| `analysis/verify_rate/v6_idk.py` | ok | 15 |
| `analysis/verify_rate/v7_worth.py` | ok | 2 |
| `analysis/verify_rate/v8_offcurve.py` | ok | 6 |
| `analysis/verify_rate/vr.py` | ok | 0 |
| `analysis/verify_what/v00_leak.py` | ok | 1 |
| `analysis/verify_what/v01_cache.py` | needs arguments | 0 |
| `analysis/verify_what/v02_configs.py` | ok | 1 |
| `analysis/verify_what/v10_knowledge.py` | ok | 75 |
| `analysis/verify_what/v20_sched.py` | ok | 19 |
| `analysis/verify_what/v20b_r998.py` | ok | 2 |
| `analysis/verify_what/v21_hash_jump.py` | ok | 14 |
| `analysis/verify_what/v22b_jumpwin.py` | ok | 0 |
| `analysis/verify_what/v30_judge.py` | needs arguments | 0 |
| `analysis/verify_what/v31_specific.py` | ok | 95 |
| `analysis/verify_what/v31b_placebo.py` | ok | 0 |
| `analysis/verify_what/v32_qboot.py` | ok | 73 |
| `analysis/verify_what/v33_when_what.py` | ok | 19 |
| `analysis/verify_what/v40_hash_curves.py` | ok | 17 |
| `analysis/verify_where/v0_cache.py` | needs arguments | 0 |
| `analysis/verify_where/v1_runs.py` | needs arguments | 0 |
| `analysis/verify_where/v2_judge_errors.py` | ok | 4 |
| `analysis/verify_where/v3_fingerprint.py` | ok | 26 |
| `analysis/verify_where/v4_probes.py` | needs arguments | 1 |
| `analysis/verify_where/v5_readable.py` | ok | 62 |
| `analysis/verify_where/v6_forecast.py` | ok | 46 |
| `analysis/verify_where/v7_visibility.py` | ok | 67 |
| `analysis/verify_where/v7b_fit.py` | ok | 0 |
| `paper/check_final.py` | ok | 1 |
| `paper/fig1_data.py` | ok | 2 |
| `paper/gen_app_tables.py` | ok | 0 |
| `paper/gen_forced_tables.py` | ok | 0 |
| `paper/gen_main_table.py` | ok | 0 |
| `paper/gen_tables.py` | ok | 0 |
| `paper/gen_trans_tables.py` | ok | 0 |
| `paper/make_fig_ladder.py` | ok | 1 |
| `paper/make_figs_final.py` | ok | 1 |
| `paper/final_numbers.py` | ok | 5 |

## Files that differ from the released file after the rerun

The released outputs are the ones the paper quotes; they were written by a rerun of every script on the final set of runs.

- `analysis/answers/a14_out.txt`: the 'base' column of the two Sonnet rows changes in the third decimal when the script is run again; nothing else differs
- `analysis/general/g05_rows.pkl`: cache; its rows are stored in the order the run folders were read
- `analysis/meta/cache/configs.json`: table of run settings; it stores the file paths of the machine it was written on
- `analysis/verify_grpo/cache/cfg_v.json`: table of run settings; it stores the file paths of the machine it was written on
- `analysis/verify_map/v04_dose_out.txt`: prints its running time
- `analysis/verify_map/v05_crossover_out.txt`: prints its running time
- `analysis/verify_map/v07_spec_out.txt`: the bootstrap intervals move in the third decimal when it is run again; the point estimates are identical
