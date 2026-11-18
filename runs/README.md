# Run outputs

This folder is filled by the archives on the release page (see the top-level README).

- `run1/rl/<run>/eval_rl_ck0.jsonl`, `eval_rl_ck120.jsonl`: the model's answers before and after training, one JSON object per question and sample
  (`idx` question, `s` sample number with -1 for the greedy answer, `text` the reply, `ans` the normalised answer). `train_log.jsonl`: one line per training step.
  `score_rl_<judge>.jsonl`: a judge's probability for each answer, where it was scored.
- `run1/base/<student>_<set>_n32_s1.jsonl`: 32 samples per question from the untrained student; `score_<judge>_...`: the judges' probabilities for them
  (`py` = probability of "correct").
- `run1/adeval/probe2/<run>.jsonl`: answers of a trained model to the capitals probe; `fresh/`: answers to the fresh questions;
  `fprobe/`, `fheld_*/`: answers with "I don't know" blocked (rows with `s` = -2).
