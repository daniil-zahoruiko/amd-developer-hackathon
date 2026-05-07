class TextPreprocessor:
    def __init__(self, config):
        self.config = config

    def preprocess(self, text: str) -> dict:
        """
        TODO: Format `text` into the input dict TRIBE v2 expects for text stimuli.
        Refer to the TRIBE v2 model card for the exact input format (tokenization,
        stimulus metadata, subject conditioning fields, etc.).
        The returned dict is passed directly to TRIBERunner.run().
        """
        # TODO: replace with real TRIBE v2 input formatting
        return {"text": text, "segments": self._split_segments(text), "mock": True}

    def _split_segments(self, text: str) -> list[str]:
        """
        Splits text into segments of roughly config.SEGMENT_CHARS characters,
        breaking on sentence boundaries (". ") where possible.
        """
        limit = self.config.SEGMENT_CHARS
        sentences = text.split(". ")
        segments: list[str] = []
        current = ""

        for i, sentence in enumerate(sentences):
            # Re-attach the period stripped by split (except for the last sentence)
            piece = sentence if i == len(sentences) - 1 else sentence + ". "
            if current and len(current) + len(piece) > limit:
                segments.append(current.strip())
                current = piece
            else:
                current += piece

        if current.strip():
            segments.append(current.strip())

        return segments or [text]
