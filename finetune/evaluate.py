"""
Post-training evaluation: compare fine-tuned model vs base model on TRIBE scores.

Loads the LoRA adapter from finetune/checkpoints/final/, generates podcast
scripts with both models for a set of held-out topics, scores each with TRIBE,
and prints a comparison table showing brain engagement improvement.

Run from the hackathon root:
    cd /root/amd-developer-hackathon
    ./venv/bin/python -m finetune.evaluate

Options via CLI (edit the constants below or pass as env vars):
    EVAL_TOPICS  — comma-separated list of topics (overrides default list)
    ADAPTER_DIR  — path to LoRA adapter dir (default: finetune/checkpoints/final)
"""

import sys
import os
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except ImportError:
    raise SystemExit("peft is not installed.  ./venv/bin/pip install peft")

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
BRAINSKRIBBL_DIR = os.path.join(REPO_ROOT, "brainskribbl")
TRIBEV2_DIR = os.path.join(REPO_ROOT, "tribev2")

sys.path.insert(0, TRIBEV2_DIR)      # fixes tribev2 namespace-package shadowing
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, BRAINSKRIBBL_DIR)

from config import Config
from modalities.text.params import TextParams
from scoring.tribe_runner import TRIBERunner
from scoring.roi_extractor import ROIExtractor
from finetune.config import TrainConfig

# ── defaults ──────────────────────────────────────────────────────────────────

EVAL_TOPICS = os.environ.get("EVAL_TOPICS", "").split(",") if os.environ.get("EVAL_TOPICS") else [
    "why aerobic exercise has such a powerful effect on mood",
    "the role of dopamine in motivation and reward learning",
    "how transformer-based language models predict the next word",
    "why biodiversity matters for ecosystem resilience",
    "the psychology of habit formation and behaviour change",
]

_SYSTEM_PROMPT = (
    "You are a writing engine. Output ONLY the final text. "
    "Do not show reasoning, drafts, checks, or explanations."
)


# ── generation helpers ────────────────────────────────────────────────────────

def _build_prompt(topic: str, params: TextParams) -> str:
    """Must match TextGenerator.generate_from_topic exactly."""
    return (
        f"Write an engaging educational podcast script about: {topic}\n\n"
        "Requirements:\n"
        "- Total length: approximately 700 words\n"
        "- Use 4 well-developed paragraphs\n"
        "- Each paragraph should be moderately detailed (roughly 120-220 words)\n"
        f"- Average sentence length: {params.sentence_length} words\n"
        f"- Tone: {params.emotional_tone}\n"
        f"- Complexity level: {params.vocab_complexity}\n"
        "- Focus on clarity, flow, and listener engagement\n"
        "- Include interesting explanations, examples, or analogies where appropriate\n"
        "- Avoid overly short paragraphs or fragmented ideas\n"
        "- Avoid excessive repetition or unnecessary filler\n"
        "- Write in a natural spoken style suitable for narration\n"
        "- Do not include section titles, bullet points, stage directions, or speaker labels\n"
        "- Output only the podcast narration text\n"
    )


def _generate(model, tokenizer, topic: str, params: TextParams) -> str:
    prompt = _build_prompt(topic, params)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=1000)
    output_ids = generated[0][len(inputs.input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def _score(text: str, tribe_runner: TRIBERunner, roi_extractor: ROIExtractor) -> float:
    preds, segments, _ = tribe_runner.run_text(text)
    _, abs_scores = roi_extractor.extract_absolute_score(preds, segments)
    return float(abs_scores.mean())


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    train_cfg = TrainConfig()
    adapter_dir = os.environ.get(
        "ADAPTER_DIR", str(Path(train_cfg.output_dir) / "final")
    )

    if not Path(adapter_dir).exists():
        raise SystemExit(
            f"Adapter not found at {adapter_dir}.\n"
            "Run train.py first:  ./venv/bin/python -m finetune.train"
        )

    params = TextParams.default()

    print("Loading TRIBE scorer ...")
    app_config = Config()
    tribe_runner = TRIBERunner(app_config)
    roi_extractor = ROIExtractor()

    print(f"\nLoading base model {train_cfg.base_model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        train_cfg.base_model_id, trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        train_cfg.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    base_model.eval()

    print(f"Loading fine-tuned adapter from {adapter_dir} ...")
    ft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    ft_model.eval()

    results = []
    print(f"\nEvaluating {len(EVAL_TOPICS)} topics ...\n")

    for topic in EVAL_TOPICS:
        print(f"Topic: {topic!r}")

        print("  base  → generating ...", flush=True)
        base_text = _generate(base_model, tokenizer, topic, params)
        base_score = _score(base_text, tribe_runner, roi_extractor)
        print(f"  base  score: {base_score:.1f}")

        print("  tuned → generating ...", flush=True)
        ft_text = _generate(ft_model, tokenizer, topic, params)
        ft_score = _score(ft_text, tribe_runner, roi_extractor)
        print(f"  tuned score: {ft_score:.1f}  (Δ {ft_score - base_score:+.1f})\n")

        results.append({
            "topic": topic,
            "base_score": base_score,
            "ft_score": ft_score,
            "delta": ft_score - base_score,
        })

    # ── summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"{'Topic':<40} {'Base':>6} {'Tuned':>6} {'Δ':>6}")
    print("-" * 68)
    for r in results:
        short = r["topic"][:38] + ".." if len(r["topic"]) > 40 else r["topic"]
        print(f"{short:<40} {r['base_score']:>6.1f} {r['ft_score']:>6.1f} "
              f"{r['delta']:>+6.1f}")
    print("=" * 68)

    mean_base = sum(r["base_score"] for r in results) / len(results)
    mean_ft = sum(r["ft_score"] for r in results) / len(results)
    print(f"{'MEAN':<40} {mean_base:>6.1f} {mean_ft:>6.1f} "
          f"{mean_ft - mean_base:>+6.1f}")

    # Save results for the demo
    out_file = Path(_HERE) / "data" / "eval_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
