import os
from dataclasses import dataclass, field
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DataGenConfig:
    n_pairs: int = 50
    pairs_file: str = os.path.join(_HERE, "data", "pairs.jsonl")
    progress_file: str = os.path.join(_HERE, "data", "progress.json")


@dataclass
class TrainConfig:
    data_file: str = os.path.join(_HERE, "data", "pairs.jsonl")
    output_dir: str = os.path.join(_HERE, "checkpoints")
    # Must match the model used by TextGenerator in brainskribbl/modalities/text/generator.py
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct"

    # LoRA — r=8 is a good default for instruction following; covers all projection layers
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training — small dataset so more epochs + small effective batch size
    num_epochs: int = 5
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4  # effective batch = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    save_steps: int = 20
    logging_steps: int = 5
    bf16: bool = True
