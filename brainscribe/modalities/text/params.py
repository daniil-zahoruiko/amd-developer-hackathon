from dataclasses import dataclass


@dataclass
class TextParams:
    sentence_length: int = 14       # target avg words per sentence, range 6-24
    vocab_complexity: int = 8       # Flesch-Kincaid grade level, range 6-14
    emotional_tone: str = "neutral" # choices: neutral, warm, urgent, playful
    structure: str = "prose"        # choices: prose, bullets, numbered

    @classmethod
    def default(cls) -> "TextParams":
        return cls()

    @classmethod
    def choices(cls) -> dict:
        return {
            "sentence_length": (6, 24),
            "vocab_complexity": (6, 14),
            "emotional_tone": ["neutral", "warm", "urgent", "playful"],
            "structure": ["prose", "bullets", "numbered"],
        }
