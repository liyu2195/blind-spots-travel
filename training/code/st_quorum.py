"""Group-quorum gate for judge rewards (st_grpo_q.py --quorum M). Pure Python, testable without a GPU.
Idea: standard GRPO pays a single judge-accepted rollout in a group, so one lucky false accept is reinforced
("judge-pass@k >= 1 becomes pass@1"). The gate pays a question's accepted rollouts only if the group shows a quorum:
  mode "same": at least M accepted rollouts give the SAME answer; only the rollouts with that answer are paid, every other attempted
               answer of the group (accepted or not) gets 0;
  mode "any":  at least M attempted rollouts are accepted (whatever their answers); then accepted rollouts keep their pay.
If the quorum fails, every attempted answer of the group gets 0. "I don't know" replies and malformed replies are not touched
(they keep the credit / 0 they already have). Ties between answers with the same number of accepted rollouts: the first one in the group."""
from collections import Counter


def quorum_gate(rew, answers, idk, valid, k, m, mode="same", paid=0.5):
    """rew: list of rewards per rollout (after the IDK credit and the format rule); answers: normalised answer strings ("" = none);
    idk / valid: booleans per rollout; k: rollouts per question; m: quorum. Returns (new rewards, stats)."""
    out = list(rew); n_groups = len(rew) // k; passed = 0; acc_total = 0; removed = 0
    for g in range(n_groups):
        idx = [i for i in range(g * k, (g + 1) * k) if answers[i] and valid[i] and not idk[i]]       # attempted answers
        acc = [i for i in idx if rew[i] > paid]
        acc_total += len(acc)
        if mode == "any":
            ok = len(acc) >= m; keep = set(acc) if ok else set()
        else:
            cnt = Counter(answers[i] for i in acc); best = None
            for i in acc:                                                 # first answer (in group order) with the highest count
                if best is None or cnt[answers[i]] > cnt[best]: best = answers[i]
            ok = best is not None and cnt[best] >= m; keep = {i for i in acc if answers[i] == best} if ok else set()
        passed += ok
        for i in idx:
            if i not in keep:
                if rew[i] > paid: removed += 1
                out[i] = 0.0
    return out, {"q_pass": passed / max(1, n_groups), "q_removed": removed / max(1, acc_total)}
