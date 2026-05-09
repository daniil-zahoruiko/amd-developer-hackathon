from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from modalities.text.params import TextParams
    from modalities.text.generator import TextGenerator
    from scoring.tribe_runner import TRIBERunner
    from scoring.roi_extractor import ROIExtractor


SYSTEM_PROMPT = """\
You are a precise text editor improving engagement of content that will be \
read aloud. The text has been scored by a brain simulation model (TRIBE v2) \
that measures predicted neural engagement when a person hears it spoken.

Sentences that score low when heard tend to share these problems:
- Passive voice or heavy nominalization ("it has been observed that")
- Abstract language with no concrete anchor (names, numbers, physical details)
- Predictable or clichéd phrasing that lets the mind drift
- Over-hedged language that drains stakes ("it is worth noting that")
- Syntactically complex clauses that are hard to track aurally

Your task: rewrite ONLY the sentences listed below. Return the complete \
original text with those sentences replaced. Do not change any other sentence. \
Do not add new paragraphs. Preserve meaning.\
"""


class Optimizer:
    def __init__(self, generator: "TextGenerator"):
        self.generator = generator

    def suggest_refinement_instruction(
        self,
        seq_ids: np.ndarray,
        scores: np.ndarray,
        seq_to_sentence: dict,
    ) -> str:
        if len(scores) == 0:
            return "Improve overall clarity and engagement."

        # Flag the bottom quartile; always flag at least one sentence
        threshold = float(np.percentile(scores, 25))
        weak_mask = scores <= threshold
        if not weak_mask.any():
            weak_mask[scores.argmin()] = True

        lines = []
        for seq_id, score, is_weak in zip(seq_ids, scores, weak_mask):
            if not is_weak:
                continue
            sentence_text = seq_to_sentence.get(int(seq_id), f"[sequence {seq_id}]")
            lines.append(f'[score {score:.1f}/100] "{sentence_text}"')

        if not lines:
            return "Improve overall clarity and engagement."

        return (
            "Rewrite the following low-engagement sequences:\n\n"
            + "\n\n".join(lines)
        )

    def run_one_iteration(
        self,
        text: str,
        params: "TextParams",
        tribe_runner: "TRIBERunner",
        roi_extractor: "ROIExtractor",
    ) -> tuple[str, tuple]:
        preds, segments, df = tribe_runner.run_text(text)
        seq_ids, scores = roi_extractor.extract_segment_scores(preds, segments)
        words = df[df["type"] == "Word"]
        def _seq_text(group):
            lst = list(group["text"])
            n = len(lst)
            for period in range(1, n + 1):
                if n % period == 0 and lst[:period] * (n // period) == lst:
                    return " ".join(lst[:period])
            return " ".join(lst)
        seq_to_sentence = words.groupby("sequence_id").apply(_seq_text).to_dict()
        instruction = self.suggest_refinement_instruction(seq_ids, scores, seq_to_sentence)
        print(f"Refining with {instruction}")
        refined_text = self.generator.refine(text, params, instruction, SYSTEM_PROMPT)
        return refined_text, (seq_ids, scores, seq_to_sentence)
