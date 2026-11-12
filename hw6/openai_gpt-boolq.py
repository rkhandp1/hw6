"""
openai_gpt-boolq.py

Few-shot (8-shot) prompting on BoolQ.
  * 8 demonstrations randomly sampled from the BoolQ TRAIN split
  * balanced: 4 'yes' + 4 'no'
  * interleaved (yes, no, yes, no, ...) to avoid recency / label-imbalance bias
  * evaluated on N (default 30) random instances from the BoolQ VALIDATION split

Backends
  --backend openai : OpenAI API. Default model gpt-4.1-mini (Chat Completions endpoint).
                     The assignment specifies gpt-3.5-turbo-instruct (Completions endpoint), but OpenAI
                     retired it on 2026-09-28; pass --model gpt-3.5-turbo-instruct to reproduce the error.
  --backend hf     : local Hugging Face instruction-tuned model used as a substitute
                     (default Qwen/Qwen2.5-7B-Instruct), prompted with the same raw text prompt.

Usage
  export OPENAI_API_KEY=<your key>
  python openai_gpt-boolq.py --backend openai                      # gpt-4.1-mini
  python openai_gpt-boolq.py --backend openai --model gpt-3.5-turbo-instruct
  python openai_gpt-boolq.py --backend hf --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import os
import random

from datasets import load_dataset


# ---------------------------------------------------------------- prompt construction
def format_example(ex, with_answer=True):
    question = ex["question"].strip()
    if not question.endswith("?"):
        question += "?"
    text = f"Passage: {ex['passage'].strip()}\nQuestion: {question}\nAnswer:"
    if with_answer:
        text += " yes" if ex["answer"] else " no"
    return text


def sample_balanced_demos(train, k=8, seed=42):
    """Randomly pick k/2 'yes' and k/2 'no' demos and interleave them."""
    rng = random.Random(seed)
    yes_idx = [i for i, a in enumerate(train["answer"]) if a]
    no_idx = [i for i, a in enumerate(train["answer"]) if not a]
    yes = rng.sample(yes_idx, k // 2)
    no = rng.sample(no_idx, k // 2)
    # interleave; randomly choose which label goes first
    first, second = (yes, no) if rng.random() < 0.5 else (no, yes)
    order = [i for pair in zip(first, second) for i in pair]
    return [train[i] for i in order]


def build_prompt(demos, query):
    header = "Answer each question about the passage with yes or no.\n\n"
    shots = "\n\n".join(format_example(d) for d in demos)
    return header + shots + "\n\n" + format_example(query, with_answer=False)


def parse_answer(text):
    """Map the model's continuation to True (yes) / False (no) / None (unparseable)."""
    words = text.strip().lower().split()
    if not words:
        return None
    w = words[0].strip(".,!:;\"'")
    if w.startswith("yes"):
        return True
    if w.startswith("no"):
        return False
    return None


# ---------------------------------------------------------------- backends
class OpenAIBackend:
    """Legacy completion models (e.g. gpt-3.5-turbo-instruct) use the Completions endpoint;
    current chat models (e.g. gpt-4.1-mini) use Chat Completions with the whole few-shot
    prompt sent as a single user message."""

    def __init__(self, model="gpt-4.1-mini"):
        from openai import OpenAI
        self.client = OpenAI()                # reads OPENAI_API_KEY from the environment
        self.model = model
        self.legacy = model.endswith("-instruct") or model in ("davinci-002", "babbage-002")

    def complete(self, prompt):
        if self.legacy:
            resp = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=3,
                temperature=0,
                stop=["\n"],
            )
            return resp.choices[0].text
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=3,
            temperature=0,
            stop=["\n"],
        )
        return resp.choices[0].message.content or ""


class HFBackend:
    def __init__(self, model="Qwen/Qwen2.5-7B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype).to(device)
        self.model.eval()

    def complete(self, prompt):
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,                                  # temperature 0 (greedy)
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["openai", "hf"], default="openai")
    p.add_argument("--model", default=None,
                   help="default: gpt-4.1-mini (openai) / Qwen/Qwen2.5-7B-Instruct (hf)")
    p.add_argument("--num_eval", type=int, default=30)
    p.add_argument("--num_shots", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show_prompt", action="store_true", help="print the first full prompt")
    args = p.parse_args()

    dataset = load_dataset("google/boolq")
    train, val = dataset["train"], dataset["validation"]

    demos = sample_balanced_demos(train, k=args.num_shots, seed=args.seed)
    print("Demonstration labels (in order):", ["yes" if d["answer"] else "no" for d in demos])

    rng = random.Random(args.seed + 1)
    eval_idx = rng.sample(range(len(val)), args.num_eval)
    eval_set = [val[i] for i in eval_idx]

    if args.backend == "openai":
        backend = OpenAIBackend(args.model or "gpt-4.1-mini")
    else:
        backend = HFBackend(args.model or "Qwen/Qwen2.5-7B-Instruct")

    correct, unparsed = 0, 0
    for n, ex in enumerate(eval_set, 1):
        prompt = build_prompt(demos, ex)
        if args.show_prompt and n == 1:
            print("=" * 30, "PROMPT", "=" * 30)
            print(prompt)
            print("=" * 68)
        raw = backend.complete(prompt)
        pred = parse_answer(raw)
        gold = ex["answer"]
        if pred is None:
            unparsed += 1
        correct += int(pred == gold)
        print(f"[{n:2d}] gold={'yes' if gold else 'no':3s} "
              f"pred={raw.strip()!r:10s} {'✓' if pred == gold else '✗'}")

    acc = correct / len(eval_set)
    print(f"\nAccuracy: {correct}/{len(eval_set)} = {acc:.3f}  (unparseable outputs: {unparsed})")


if __name__ == "__main__":
    main()
