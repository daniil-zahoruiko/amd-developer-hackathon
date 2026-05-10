from dataclasses import dataclass


@dataclass
class Config:
    TRIBE_MODEL_PATH: str = "path/to/tribe_v2"
    DEVICE: str = "cuda"          # ROCm uses same API — no change needed
    N_SUBJECTS: int = 30          # virtual subjects per TRIBE v2 scoring pass
    SEGMENT_CHARS: int = 300      # characters per scored text segment
    APP_TITLE: str = "BrainSkribbl"
    APP_DESCRIPTION: str = "Refine any text using TRIBE v2 brain simulation as a feedback signal."
