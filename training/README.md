# Training

- `code/`: `st_grpo.py`, `st_grpo_b.py` and `st_grpo_c.py` are the trainer (GRPO with LoRA, vLLM for sampling) at three versions; each version adds
  options and, with the new options unset, computes every reward as the version before; the job scripts name the version a run used.
  `st_common.py` holds the prompts,
  answer normalisation and the answer-key match, `st_gen.py` samples and decodes, `st_gen_forced.py` decodes with "I don't know" blocked,
  `st_judge.py` scores answers with an open judge, `st_api.py` calls the Sonnet judge, `st_hidden.py` extracts hidden states.
- `jobs/`: the job scripts as they ran on our cloud machines (`acl_st_89_worker16.sh` is the queue worker that trained most runs,
  `acl_st_90_adeval.sh` evaluates kept adapters, `acl_st_92_forced.sh` runs forced decoding). They install `vllm==0.30.0` and `peft==0.21.2`.
  Names of our cloud project, storage bucket and secrets are replaced by `YOUR_GCP_PROJECT`, `YOUR_BUCKET`, `HF_TOKEN_SECRET` and `ANTHROPIC_KEY_SECRET`;
  the scripts document what ran and need adapting to another machine.
- `queues/`: one file per run with its settings as `KEY=VALUE` pairs (grader, error rates, data files, seed, output name).
- `configs/run_configs.tsv`: one line per run folder in `runs/run1/rl/` with its settings. A few early runs were configured inside a job script
  and have an empty line here.
