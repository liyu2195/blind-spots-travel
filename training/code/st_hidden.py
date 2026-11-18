"""Hidden states of the UNTRAINED student at the last prompt token (the position that decides the
first reply token), for a few layers, for every question (IDK prompt). Used to test whether what RL students learn about when to
abstain is a simple read-out of the untrained model's internal state. Output: npz with idx, layers, h (float16, N x layers x d)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--every", type=int, default=4, help="keep every n-th layer, plus the last")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import numpy as np, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.student); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    model.eval()
    chat = lambda q: tok.apply_chat_template([{"role": "user", "content": C.TQ_PROMPT_IDK.format(q=q)}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    probs = C.load_jsonl(a.data)
    if a.limit:
        probs = probs[:a.limit]
    H, layers = [], None
    for b in range(0, len(probs), a.batch):
        batch = probs[b:b + a.batch]
        enc = tok([chat(p["question"]) for p in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states      # embeddings + one entry per layer
        if layers is None:
            n = len(hs) - 1
            layers = sorted(set(list(range(a.every, n + 1, a.every)) + [n]))
            print("layers kept:", layers, "of", n, "width", hs[-1].shape[-1], flush=True)
        H.append(torch.stack([hs[l][:, -1, :].float() for l in layers], 1).cpu().numpy().astype("float16"))
        print(f"{min(b + a.batch, len(probs))}/{len(probs)}", flush=True)
    np.savez_compressed(a.out, idx=np.array([p["idx"] for p in probs]), layers=np.array(layers), h=np.concatenate(H, 0))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
