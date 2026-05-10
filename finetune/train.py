"""
LoRA fine-tune Qwen2.5-7B-Instruct on brain-engagement-optimised podcast pairs.

Each training example is weighted by its TRIBE score delta (refined − original),
normalised to [0.1, 1.0] across the dataset.  This means pairs where TRIBE
confirmed the refinement helped are learned from more heavily than pairs where
the aggregate score did not improve, without throwing any data away.

Requires peft:
    ./venv/bin/pip install peft

Run from the hackathon root:
    cd /root/amd-developer-hackathon
    ./venv/bin/python -m finetune.train

Resumes automatically from the latest checkpoint in finetune/checkpoints/.
"""

import sys
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    raise SystemExit(
        "peft is not installed.\n"
        "Fix: ./venv/bin/pip install peft"
    )

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

from finetune.config import TrainConfig

_SYSTEM_PROMPT = (
    "You are a writing engine. Output ONLY the final text. "
    "Do not show reasoning, drafts, checks, or explanations."
)


# ── dataset ───────────────────────────────────────────────────────────────────

class EngagementDataset(Dataset):
    """
    Tokenised (prompt → refined_text) pairs weighted by TRIBE score delta.

    Weight normalisation: delta values are linearly mapped to [0.1, 1.0] across
    the full dataset.  The worst pair gets weight 0.1 (still learned from, but
    minimally); the best gets 1.0.  Missing delta defaults to 0.0 before
    normalisation so legacy pairs without a delta field are treated as neutral.
    """

    def __init__(self, pairs_file: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        raw = Path(pairs_file).read_text(encoding="utf-8").strip().splitlines()
        pairs = [json.loads(line) for line in raw if line.strip()]
        print(f"Loaded {len(pairs)} pairs from {pairs_file}")

        # Compute weight normalisation bounds from all valid pairs
        deltas = [p.get("delta", 0.0) for p in pairs]
        min_d, max_d = min(deltas), max(deltas)
        d_range = max(max_d - min_d, 1e-6)
        print(f"Delta range: {min_d:.2f} → {max_d:.2f}  (normalising to [0.1, 1.0])")

        skipped = 0
        for pair in pairs:
            refined = pair.get("refined_text", "").strip()
            prompt = pair.get("prompt", "").strip()
            if not refined or not prompt:
                skipped += 1
                continue

            delta = pair.get("delta", 0.0)
            weight = 0.1 + 0.9 * (delta - min_d) / d_range

            enc = self._encode(prompt, refined, weight)
            if enc is not None:
                self.examples.append(enc)
            else:
                skipped += 1

        print(f"Dataset: {len(self.examples)} examples ({skipped} skipped / too long).")

    def _encode(self, user_prompt: str, response: str, weight: float):
        prompt_messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": response}
        ]

        prompt_str = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_str = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        full_enc = self.tokenizer(
            full_str,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = full_enc["input_ids"]

        # Skip examples where the response was entirely truncated away
        if len(input_ids) - len(prompt_ids) < 20:
            return None

        labels = list(input_ids)
        for idx in range(min(len(prompt_ids), len(labels))):
            labels[idx] = -100  # mask prompt tokens from loss

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(full_enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weight": torch.tensor(weight, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ── collator ──────────────────────────────────────────────────────────────────

@dataclass
class WeightedDataCollator:
    """
    Wraps DataCollatorForSeq2Seq and passes `weight` through separately so
    the padding logic never sees it (it's a scalar, not a sequence).
    """
    base: DataCollatorForSeq2Seq

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [dict(f) for f in features]  # don't mutate dataset examples
        weights = torch.stack([f.pop("weight") for f in features])
        batch = self.base(features)
        batch["weight"] = weights
        return batch


# ── trainer ───────────────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """
    Computes per-example cross-entropy loss and scales each by its delta weight
    before averaging.  High-delta examples (refinement confirmed by TRIBE) pull
    the gradient harder than low-delta or negative-delta examples.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight")          # (batch,)  float32
        labels  = inputs["labels"]              # (batch, seq_len)
        outputs = model(**inputs)
        logits  = outputs.logits                # (batch, seq_len, vocab_size)

        # Shift by one for causal LM: predict token[t] from token[t-1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())             # (batch, seq_len-1)

        # Average over non-masked (non-prompt) tokens per example
        valid = (shift_labels != -100).float()
        per_example_loss = (per_token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

        # Scale by normalised delta weight and average over the batch
        weighted_loss = (per_example_loss * weights.to(per_example_loss.device)).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss


# ── utilities ─────────────────────────────────────────────────────────────────

def _latest_checkpoint(output_dir: str) -> Optional[str]:
    p = Path(output_dir)
    if not p.exists():
        return None
    ckpts = sorted(
        p.glob("checkpoint-*"),
        key=lambda x: int(x.name.split("-")[-1]),
    )
    return str(ckpts[-1]) if ckpts else None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = TrainConfig()

    if not Path(cfg.data_file).exists():
        raise SystemExit(
            f"No data file at {cfg.data_file}.\n"
            "Run generate_data.py first:\n"
            "  ./venv/bin/python -m finetune.generate_data"
        )

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building dataset ...")
    dataset = EngagementDataset(cfg.data_file, tokenizer, cfg.max_seq_length)

    if len(dataset) == 0:
        raise SystemExit(
            "Dataset is empty after filtering.\n"
            "Generate more pairs: ./venv/bin/python -m finetune.generate_data"
        )

    print(f"\nLoading base model {cfg.base_model_id} ...")
    # No torch.compile — it breaks LoRA gradient flow
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.enable_input_require_grads()

    print("Applying LoRA ...")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    resume_ckpt = _latest_checkpoint(cfg.output_dir)
    if resume_ckpt:
        print(f"Resuming from {resume_ckpt}")
    else:
        print("Starting fresh training run.")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
    )

    collator = WeightedDataCollator(
        base=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("\nStarting training ...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_dir = Path(cfg.output_dir) / "final"
    print(f"\nSaving LoRA adapter to {final_dir} ...")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print("Done.")


if __name__ == "__main__":
    main()
