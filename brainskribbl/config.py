from dataclasses import dataclass


@dataclass
class Config:
    DEVICE: str = "cuda"          # ROCm uses same API — no change needed
    SEGMENT_CHARS: int = 300      # characters per scored text segment
    APP_TITLE: str = "BrainSkribbl"
    APP_DESCRIPTION: str = "Refine any text using TRIBE v2 brain simulation as a feedback signal."
