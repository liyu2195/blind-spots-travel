"""How far is each question from "I don't know" in the UNTRAINED student?
For every question (IDK prompt) this writes: the greedy reply and its log-probability; the greedy reply when the first token of
"I don't know" is not allowed as the first token (= the student's best real answer) and its log-probability; the log-probability
of the IDK first token and of the best other first token; and the log-probability of the whole canonical reply "I don't know.".
Use: a "one-number" abstention rule (add a constant to the IDK first-token logit and decode greedily) needs no training; this
file lets us trace its hallucination-abstention curve and compare its abstention set with RL-trained students'."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_new", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lora", default=None)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.student); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    if a.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.lora)
    model.eval()
    gc = model.generation_config.eos_token_id
    eos = set([tok.eos_token_id] + ([gc] if isinstance(gc, int) else list(gc or [])))
    chat = lambda q: tok.apply_chat_template([{"role": "user", "content": C.TQ_PROMPT_IDK.format(q=q)}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    idk_ids = tok("I don't know.", add_special_tokens=False).input_ids
    idk1 = idk_ids[0]
    print("IDK reply tokens:", [tok.decode([t]) for t in idk_ids], "first token id", idk1, flush=True)

    class BanFirst(LogitsProcessor):          # the IDK first token may not open the reply
        def __init__(self, start):
            self.start = start

        def __call__(self, input_ids, scores):
            if input_ids.shape[1] == self.start:
                scores = scores.clone(); scores[:, idk1] = -float("inf")
            return scores

    def reply(gen, n_prompt, lp1):
        """texts and summed log-probabilities of generated replies; the first token is scored with the unconstrained lp1."""
        seq = gen.sequences[:, n_prompt:]
        tr = model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True).float()
        res = []
        for i in range(seq.shape[0]):
            row = seq[i].tolist()
            cut = next((j for j, t in enumerate(row) if t in eos), None)
            n = len(row) if cut is None else cut + 1
            lp = lp1[i, row[0]].item() + sum(tr[i, j].item() for j in range(1, n))
            res.append((tok.decode(row[:n], skip_special_tokens=True), lp, n, cut is not None, row[0]))
        return res

    probs = C.load_jsonl(a.data)
    if a.limit:
        probs = probs[:a.limit]
    out = open(a.out, "w")
    for b in range(0, len(probs), a.batch):
        batch = probs[b:b + a.batch]
        enc = tok([chat(p["question"]) for p in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        n_prompt = enc["input_ids"].shape[1]
        with torch.no_grad():
            lp1 = torch.log_softmax(model(**enc).logits[:, -1, :].float(), -1)
            kw = dict(do_sample=False, max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id, output_scores=True,
                      return_dict_in_generate=True, use_cache=True)
            g1 = reply(model.generate(**enc, **kw), n_prompt, lp1)
            g2 = reply(model.generate(**enc, logits_processor=LogitsProcessorList([BanFirst(n_prompt)]), **kw), n_prompt, lp1)
            # whole canonical IDK reply, teacher-forced
            ids = torch.tensor([idk_ids] * len(batch), device=enc["input_ids"].device)
            full = torch.cat([enc["input_ids"], ids], 1)
            mask = torch.cat([enc["attention_mask"], torch.ones_like(ids)], 1)
            lg = torch.log_softmax(model(input_ids=full, attention_mask=mask).logits[:, n_prompt - 1:-1, :].float(), -1)
            lp_idk = lg.gather(2, ids.unsqueeze(-1)).squeeze(-1).sum(1)
            other = lp1.clone(); other[:, idk1] = -float("inf")
            top_other, top_id = other.max(1)
            top5 = torch.topk(lp1, 5, dim=1)
        for i, p in enumerate(batch):
            out.write(json.dumps(dict(
                idx=p["idx"], greedy=g1[i][0], lp_greedy=round(g1[i][1], 4), greedy_ended=g1[i][3],
                ans=g2[i][0], lp_ans=round(g2[i][1], 4), ans_ntok=g2[i][2], ans_ended=g2[i][3],
                lp1_idk=round(lp1[i, idk1].item(), 4), lp1_top_other=round(top_other[i].item(), 4), top_other_tok=tok.decode([top_id[i].item()]),
                lp_idk_seq=round(lp_idk[i].item(), 4),
                top5=[[tok.decode([t]), round(v, 3)] for t, v in zip(top5.indices[i].tolist(), top5.values[i].tolist())])) + "\n")
        out.flush()
        print(f"{min(b + a.batch, len(probs))}/{len(probs)}", flush=True)
    out.close()
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
