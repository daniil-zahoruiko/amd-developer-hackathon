from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modalities.text.params import TextParams
    from modalities.text.generator import TextGenerator
    from modalities.text.preprocessor import TextPreprocessor
    from scoring.tribe_runner import TRIBERunner
    from scoring.roi_extractor import ROIExtractor


class Optimizer:
    def __init__(self, generator: "TextGenerator"):
        self.generator = generator

    def suggest_refinement_instruction(self, segment_scores: list[dict]) -> str:
        """
        TODO: Inspect segment_scores, identify the weakest segments (lowest
        engagement_score), and produce a plain-English instruction string
        describing what to improve — e.g.:
          "Shorten sentences in paragraph 2. Use more concrete language in paragraph 3."
        Consider flagging the bottom quartile of segments and describing changes
        that typically improve neural engagement (concrete nouns, active voice,
        varied sentence rhythm, sensory language).
        This instruction is passed directly to generator.refine().
        """
        # TODO: replace with real weak-segment analysis
        if not segment_scores:
            return "Improve overall clarity and engagement."
        weakest = min(segment_scores, key=lambda s: s["engagement_score"])
        return (
            f"[MOCK] Improve segment {weakest['segment_idx']} "
            f"(score: {weakest['engagement_score']:.1f}). "
            "Make it clearer and more engaging."
        )

    def run_one_iteration(
        self,
        text: str,
        params: "TextParams",
        preprocessor: "TextPreprocessor",
        tribe_runner: "TRIBERunner",
        roi_extractor: "ROIExtractor",
    ) -> tuple[str, list[dict], float]:
        """
        Runs one full refinement cycle. Implement the stubs it calls to make
        the real system work.
          1. Preprocess current text
          2. Run TRIBE v2 -> voxel activations
          3. Extract ROI scores per segment
          4. Build refinement instruction from weak segments
          5. Generate refined text
        Returns (refined_text, segment_scores, mean_score).
        """
        preprocessed = preprocessor.preprocess(text)
        activations = tribe_runner.run(preprocessed)
        segment_scores = roi_extractor.extract_segment_scores(activations)
        mean = roi_extractor.mean_score(segment_scores)
        instruction = self.suggest_refinement_instruction(segment_scores)
        refined_text = self.generator.refine(text, params, instruction)
        return refined_text, segment_scores, mean
