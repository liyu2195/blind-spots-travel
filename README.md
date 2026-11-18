# Blind Spots Travel, Knowledge Stays Home: What RL Inherits from an LLM Judge

Liang Yue and Eric A. Vance · University of Colorado Boulder · 2026

[Manuscript source](paper/main.tex) · [arXiv submission archive](https://github.com/liyu2195/blind-spots-travel/releases/download/v1.0/arxiv_submission.tar.gz)

We train 7–8B language models with reinforcement learning (GRPO with LoRA) to answer closed-book questions or say "I don't know",
and we change only the grader that gives the reward: an answer key, LLM judges of the student's size, a much stronger judge, graders
whose errors we place by hand, and the student's own 32-sample vote. The paper asks what a trained model takes from its grader to
questions it never trained on. This repository holds what is needed to check that answer: the training code, the configuration and
the answers of every run, the judges' scores, the audit labels, the capitals probe and the scripts that compute every number in the paper.

## What is here

| Folder | Content |
|---|---|
| `paper/` | LaTeX source of the paper, the scripts that write its tables and figures, and `check_v13.py`, which recomputes the numbers of the main text and compares them with the sentences of the paper |
| `analysis/` | The analysis scripts with the outputs they wrote (`*_out.txt`, `*.json`). `INDEX.md` says which file each number of the paper comes from |
| `analysis/fix_ledger/ledger.md` | The ledger of the 95 claims we tested during analysis, with the ones that failed |
| `data/` | Question sets (training and held-out), the capitals probe (`pqq_1500.jsonl`), the fresh questions, the own-vote keys and the rejection sets of the designed graders |
| `training/` | Training and evaluation code (`code/`), the job scripts as they ran (`jobs/`), the queue files that configured the runs (`queues/`) and one configuration line per run (`configs/run_configs.tsv`) |
| `runs/` | Not in git. Answers and logs of every run, answers of the untrained students with the judges' scores, and the probe answers. Download from the release page (below) |

## Getting the run outputs

The large files are attached to the [release page](../../releases) of this repository. Download them and unpack each one in the top folder of the repository:

```
tar xzf runs_rl.tar.gz          # runs/run1/rl/<run>/: answers after training (eval_rl_ck120.jsonl), before (eval_rl_ck0.jsonl), training log
tar xzf runs_untrained.tar.gz   # runs/run1/base/: 32 samples per question from the untrained students, judge scores, hidden states
tar xzf runs_probes.tar.gz      # runs/run1/adeval/: answers of the kept adapters to the capitals probe, the fresh questions, forced decoding
tar xzf analysis_caches.tar.gz  # optional: the caches the analysis scripts build from the files above (saves hours)
```

The caches are Python pickles. Load pickles only from a source you trust; every cache can be rebuilt from the run outputs with the
builder script in its folder (for example `analysis/stats/s01_cache.py`).

## Checking the numbers

```
pip install -r requirements.txt
cd paper
python v13_numbers.py        # main table, capitals refused, wrong answers per 100 (reads the raw answers of the runs)
python gen_tables.py && python gen_main_table.py && python gen_app_tables.py && python gen_trans_tables.py && python gen_forced_tables.py
python fig1_data.py && python make_figs_v13.py && python make_fig_ladder.py
python check_v13.py          # every number of the main text against the files it comes from; ends with "FAILURES: 0"
pdflatex main && pdflatex main     # main.bbl is included
```

Each analysis script is run from its own folder (`cd analysis/supp_v11 && python s22_transitions.py`). The scripts find the top
folder of the repository by the file `.repo_root`, so the repository can live anywhere. `analysis/RERUN_STATUS.md` lists, for every
script, whether we reran it from a fresh copy of this repository and whether it wrote the same files again.

### What we checked before releasing

From a fresh copy of this repository at another location, with the archives unpacked:

1. Before packaging, the authors rebuilt the main caches from the run outputs with `analysis/stats/s01_cache.py` and verified that they matched byte for byte. For this release, private storage paths in eight cache files were updated with a placeholder.
2. The ten scripts in `paper/`, run after everything else, write the same tables and number files again, and `check_v13.py` ends with `FAILURES: 0`.
3. Of the 297 scripts in `analysis/` and `paper/`, 259 ran to the end, 35 need arguments, 2 call the Anthropic API and were not run, and 1 needs the responses that are not released.
   They wrote 375 files again: 298 are identical to the released file and 77 differ.

The comparison above records the original release. The arXiv appendix uses updated estimates, including the held-out result for 27 cells.
Most differences arose because the rerun included runs or audit labels added after the original analysis. A few scripts also print their running time.
See [the rerun record](analysis/RERUN_STATUS.md) for a file-by-file account.

## Reading a run name

`<student>_<recipe><grader>_idkc<credit>_s<seed>_<dataset>`, for example `llama8_FK20judgeQa35_idkc05_s0_pq`.

- Students: `qwen7` (Qwen2.5-7B-Instruct), `llama8` (Llama-3.1-8B-Instruct); smaller students `phi4m`, `smol3`, `granite`.
- Datasets: `pq` PopQA, `h1` TriviaQA, `nq` NQ-open (750 training questions and about 750 held-out questions each).
- `FK20`: the recipe of the paper (GRPO, 120 steps, 32 questions with 8 samples each, KL weight 0.2). `idkc05`: "I don't know" earns 0.5.
- Graders: `sim00fn00` answer key; `sim00fnXX` answer key with random rejections; `judge<J><cut>` an LLM judge (`Q` Qwen, `M` Mistral, `G` Gemma,
  `O` OLMo) at its default threshold (`t50`), at the cut that accepts as many answers as the answer key (`acc`), or at the cut that accepts
  50% or 35% of answers (`a50`, `a35`); `judgeQRt50` the Qwen judge shown the reference; `apiS55` Sonnet; `frzVOTE<k>` the student's own vote
  with a cut of k of 32; `frzMAJ35` a jury of five judges; `pl...` graders whose verdicts we set by hand (Appendix E of the paper).

The exact settings of a run are its line in `training/configs/run_configs.tsv`. Appendix N of the paper lists every experiment.

## What is not here

- The LoRA adapters. The answers of every run are here; the weights are not.
- The model responses of the one experiment with harmful requests (Appendix N, "Open-ended requests with a refusal option"). Its scorer and its results are in `analysis/supp_v11/s14_sf_step0.*`.
- Records of individual crowd annotators. `analysis/extra/human_check/` has the three labels per item without annotator identifiers, which is what the reported agreement is computed from.

## Data sources and licences

The questions come from TriviaQA (Joshi et al., 2017), PopQA (Mallen et al., 2023) and NQ-open (Kwiatkowski et al., 2019; Lee et al., 2019),
and remain under the licences of those datasets (NQ-open: CC BY-SA 3.0). The code is under the MIT licence (`LICENSE`). Everything we produced
(run outputs, judge scores, audit labels, the capitals probe, tables) is under CC BY 4.0 (`LICENSE-DATA.md`).

## Citation

```
@article{yue2026blindspots,
  title  = {Blind Spots Travel, Knowledge Stays Home: What RL Inherits from an LLM Judge},
  author = {Yue, Liang and Vance, Eric A.},
  year   = {2026},
  url    = {https://github.com/liyu2195/blind-spots-travel}
}
```

Questions: liyu2195@colorado.edu
