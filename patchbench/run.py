"""Run experiments for one model and append results to a JSONL file (resumable).

Example:
    python -m patchbench.run --model Qwen/Qwen2.5-3B-Instruct --experiments main,oracle,length,crosslang \
        --preset standard --out results/
"""
import argparse
import json
import os
import random
import time

from .data import score
from .experiments import build_jobs


def load_model(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return tok, model


def chat(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)


def generate(tok, model, prompts, max_new_tokens):
    import torch
    texts = [chat(tok, p) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    n_in = enc["input_ids"].shape[1]
    lens = enc["attention_mask"].sum(1).tolist()
    return [tok.decode(o[n_in:], skip_special_tokens=True).strip() for o in out], lens


class FakeModel:
    """Offline stand-in used for testing the pipeline: answers correctly with a probability that
    decreases in French and in the middle of the context, and otherwise returns a trap value."""
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def answer(self, job):
        it = job["item"]
        p = 0.9 if job["ctx_lang"] == "en" else 0.75
        if job["exp"] != "oracle" and job["pos"] in (6, 9, 13):
            p -= 0.1 if job["ctx_lang"] == "en" else 0.2
        if job["cond"] == "closed_book":
            p = 0.0
        if self.rng.random() < p:
            v = it["answer"]
        else:
            v = self.rng.choice(it["distractor_values"] or [it["previous"]])
        word = "Answer" if job["q_lang"] == "en" else "Réponse"
        return f"{word}: {v:g}".replace(".", ",") if job["q_lang"] == "fr" else f"{word}: {v:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face model id, or 'fake' for an offline test")
    ap.add_argument("--experiments", default="main,oracle,length,crosslang")
    ap.add_argument("--preset", default="standard", choices=["smoke", "standard", "full"])
    ap.add_argument("--n_main", type=int); ap.add_argument("--n_oracle", type=int)
    ap.add_argument("--n_length", type=int); ap.add_argument("--n_crosslang", type=int)
    ap.add_argument("--out", default="results")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    fake = args.model == "fake"
    if fake:
        tok = model = None
        ntok = lambda text: len(text) // 4          # rough proxy, only for offline tests
        fm = FakeModel(args.seed)
    else:
        import torch
        torch.manual_seed(args.seed)
        tok, model = load_model(args.model)
        ntok = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])

    overrides = {k[2:]: v for k, v in vars(args).items() if k.startswith("n_") and v is not None}
    jobs, cfg = build_jobs(args.experiments.split(","), args.preset, ntok, overrides)

    os.makedirs(args.out, exist_ok=True)
    tag = args.model.split("/")[-1]
    path = os.path.join(args.out, f"{tag}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {json.loads(line)["job_id"] for line in f if line.strip()}
    todo = [j for j in jobs if j["job_id"] not in done]
    todo.sort(key=lambda j: len(j["prompt"]))          # similar lengths in a batch
    print(f"{len(jobs)} jobs, {len(done)} already done, {len(todo)} to run -> {path}")
    with open(os.path.join(args.out, f"{tag}.config.json"), "w") as f:
        json.dump({**vars(args), "resolved": cfg}, f, indent=1)

    bs, t0, k = args.batch_size, time.time(), 0
    with open(path, "a") as f:
        while k < len(todo):
            batch = todo[k:k + bs]
            if fake:
                outs, lens = [fm.answer(j) for j in batch], [ntok(j["prompt"]) for j in batch]
            else:
                import torch
                try:
                    outs, lens = generate(tok, model, [j["prompt"] for j in batch], args.max_new_tokens)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if bs == 1:
                        raise
                    bs = max(1, bs // 2)
                    print(f"OOM, batch size -> {bs}")
                    continue
            for j, o, L in zip(batch, outs, lens):
                rec = {k2: v for k2, v in j.items() if k2 not in ("prompt", "item")}
                it = j["item"]
                rec.update({"model": args.model, "output": o, "prompt_tokens": L, "answer": it["answer"],
                            "previous": it["previous"], "target": it["target"], **score(it, o)})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            k += len(batch)
            if k % (bs * 25) < bs or k == len(todo):
                el = time.time() - t0
                print(f"{k}/{len(todo)}  {el/60:.1f} min  ~{el/k*(len(todo)-k)/60:.1f} min left", flush=True)


if __name__ == "__main__":
    main()
