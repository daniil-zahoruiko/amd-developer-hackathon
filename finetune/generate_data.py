"""
Generate (original, refined) text pairs for LoRA fine-tuning.

Each pair is produced by:
  1. Generating a podcast script with TextGenerator (Qwen2.5-7B).
  2. Running one TRIBE-guided optimisation iteration to get a higher-scoring
     refined version of that script.
  3. Saving the pair immediately to JSONL so no work is lost on interruption.

Progress is tracked in a JSON checkpoint file; re-running this script
automatically skips already-completed topics.

Run from the hackathon root:
    cd /root/amd-developer-hackathon
    ./venv/bin/python -m finetune.generate_data
"""

import sys
import os
import json
import random
import time
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
BRAINSKRIBBL_DIR = os.path.join(REPO_ROOT, "brainskribbl")
TRIBEV2_DIR = os.path.join(REPO_ROOT, "tribev2")

# tribev2 is an editable install whose source lives at REPO_ROOT/tribev2/tribev2/.
# When running from REPO_ROOT the outer tribev2/ folder would be seen as a
# namespace package, shadowing the real package.  Inserting its parent first
# ensures Python resolves the inner tribev2/__init__.py correctly.
sys.path.insert(0, TRIBEV2_DIR)      # fixes tribev2 namespace-package shadowing
sys.path.insert(0, REPO_ROOT)        # enables `import finetune.*`
sys.path.insert(0, BRAINSKRIBBL_DIR)  # enables `from config import Config` etc.

from config import Config                              # brainskribbl config
from modalities.text.params import TextParams
from modalities.text.generator import TextGenerator
from scoring.tribe_runner import TRIBERunner
from scoring.roi_extractor import ROIExtractor
from optimization.optimizer import Optimizer

from finetune.config import DataGenConfig
from finetune.topics import TOPICS

# ── helpers ───────────────────────────────────────────────────────────────────

def _build_prompt(topic: str, params: TextParams) -> str:
    """Replicates the exact prompt from TextGenerator.generate_from_topic."""
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


def _varied_params(rng: random.Random) -> TextParams:
    """Sample TextParams for diversity; structure fixed to prose for audio-natural text."""
    return TextParams(
        sentence_length=rng.choice([12, 14, 16, 18]),
        vocab_complexity=rng.choice([7, 8, 9, 10]),
        emotional_tone=rng.choice(["neutral", "warm", "urgent"]),
        structure="prose",
    )


def _load_completed(progress_file: str) -> set:
    p = Path(progress_file)
    if p.exists():
        return set(json.loads(p.read_text()).get("completed", []))
    return set()


def _save_completed(progress_file: str, completed: set) -> None:
    Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
    Path(progress_file).write_text(
        json.dumps({"completed": sorted(completed)}, indent=2)
    )


def _append_pair(pairs_file: str, pair: dict) -> None:
    Path(pairs_file).parent.mkdir(parents=True, exist_ok=True)
    with open(pairs_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = DataGenConfig()
    rng = random.Random(42)

    completed = _load_completed(cfg.progress_file)
    remaining = [t for t in TOPICS if t not in completed]
    rng.shuffle(remaining)
    to_process = remaining[: max(0, cfg.n_pairs - len(completed))]

    print(f"Progress: {len(completed)} done, {len(to_process)} to go "
          f"(target {cfg.n_pairs} pairs).")

    if not to_process:
        print("All pairs already generated. Run train.py next.")
        return

    print("\nLoading models — this takes a few minutes on first run...")
    app_config = Config()
    tribe_runner = TRIBERunner(app_config)
    roi_extractor = ROIExtractor()
    generator = TextGenerator()
    optimizer = Optimizer(generator)
    print("Models loaded.\n")

    for i, topic in enumerate(to_process, 1):
        t0 = time.time()
        print(f"[{i}/{len(to_process)}] {topic!r}")

        try:
            params = _varied_params(rng)
            prompt = _build_prompt(topic, params)

            print("  → generating draft ...", flush=True)
            original_text = generator.generate_from_topic(topic, params)

            print("  → running TRIBE + LLM refinement ...", flush=True)
            # run_one_iteration scores the original to find weak sentences, then
            # refines — the score it returns is from the original, not the refined text.
            refined_text, (seq_ids, scores, _), original_score = (
                optimizer.run_one_iteration(
                    original_text, params, tribe_runner, roi_extractor
                )
            )

            if not refined_text.strip():
                print("  ! refined text is empty — skipping.")
                continue

            print("  → scoring refined text with TRIBE ...", flush=True)
            refined_preds, refined_segments, _ = tribe_runner.run_text(refined_text)
            _, refined_abs_scores = roi_extractor.extract_absolute_score(
                refined_preds, refined_segments
            )
            refined_score = float(refined_abs_scores.mean())

            delta = refined_score - original_score

            pair = {
                "topic": topic,
                "params": {
                    "sentence_length": params.sentence_length,
                    "vocab_complexity": params.vocab_complexity,
                    "emotional_tone": params.emotional_tone,
                    "structure": params.structure,
                },
                "prompt": prompt,
                "original_text": original_text,
                "refined_text": refined_text,
                "original_score": original_score,
                "refined_score": refined_score,
                "delta": delta,
                "seq_scores": scores.tolist() if hasattr(scores, "tolist") else list(scores),
            }
            _append_pair(cfg.pairs_file, pair)
            completed.add(topic)
            _save_completed(cfg.progress_file, completed)

            elapsed = time.time() - t0
            print(f"  ✓ saved  |  {original_score:.1f} → {refined_score:.1f} "
                  f"(Δ {delta:+.1f})  |  {elapsed:.0f}s elapsed")

        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved — re-run to resume.")
            return
        except Exception as exc:
            print(f"  ! error on {topic!r}: {exc} — skipping.")
            continue

    total = len(completed)
    print(f"\nDone. {total} pair(s) in {cfg.pairs_file}")
    if total < 10:
        print("Warning: fewer than 10 pairs — consider running more topics "
              "before fine-tuning.")


if __name__ == "__main__":
    main()
