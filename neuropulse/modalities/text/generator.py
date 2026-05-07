from modalities.text.params import TextParams


class TextGenerator:
    def generate_from_topic(self, topic: str, params: TextParams) -> str:
        """
        TODO: Call an LLM (e.g. Qwen2.5-7B via vLLM) to generate a short piece
        of text (2-4 paragraphs) about `topic`, following the style constraints
        in `params`. Use params.emotional_tone, params.sentence_length, and
        params.structure to shape the output prompt.
        """
        # TODO: replace with real LLM call
        return (
            f"This is a generated draft about '{topic}'.\n\n"
            f"[MOCK — replace with LLM call. tone={params.emotional_tone}, "
            f"sentence_length={params.sentence_length}]"
        )

    def refine(self, text: str, params: TextParams, instruction: str) -> str:
        """
        TODO: Call the LLM with `text` and `instruction` (a plain-English
        description of what to improve, produced by the optimizer) to produce
        a refined version that preserves meaning but improves neural engagement.
        The system prompt should instruct the model to keep the same topic and
        approximate length while applying the changes described in `instruction`.
        Returns the refined text string.
        """
        # TODO: replace with real LLM call
        return f"{text}\n\n[MOCK REFINEMENT — instruction: {instruction}]"
