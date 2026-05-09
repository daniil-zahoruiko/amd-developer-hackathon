from dataclasses import dataclass, field


@dataclass
class VideoSceneParams:
    animation_speed: float = 1.0    # seconds per Write animation, range 0.5–2.5
    words_per_frame: int = 8        # words shown per text chunk, range 4–16
    pause_after_sec: float = 1.0    # silence after scene ends, range 0.3–3.0
    visual_elements: int = 2        # number of decorative shapes alongside text, range 1–5
    emphasis_words: list[str] = field(default_factory=list)  # highlighted in accent colour

    @classmethod
    def default(cls) -> "VideoSceneParams":
        return cls()

    @classmethod
    def choices(cls) -> dict:
        return {
            "animation_speed": (0.5, 2.5),
            "words_per_frame": (4, 16),
            "pause_after_sec": (0.3, 3.0),
            "visual_elements": (1, 5),
        }
