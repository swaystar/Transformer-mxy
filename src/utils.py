import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch


@dataclass
class TrainConfig:
    dataset: str = "tiny_shakespeare"  # tiny_shakespeare | wikitext2
    vocab: str = "byte"  # byte | char
    seq_len: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    n_encoder_layers: int = 4  # 编码器层数（用于编码器-解码器）
    n_decoder_layers: int = 4  # 解码器层数（用于编码器-解码器）
    ffn_hidden: int = 1024
    dropout: float = 0.1

    # 模型架构选项
    model_type: str = "lm"  # "lm" (语言模型) 或 "seq2seq" (编码器-解码器)
    attention_type: str = "standard"  # "standard" (标准注意力) 或 "linear" (线性注意力)
    pos_encoding_type: str = "absolute"  # "absolute" (绝对位置编码) 或 "relative" (相对位置编码)
    max_relative_position: int = 128  # 最大相对位置距离（仅用于相对位置编码）

    lr: float = 3e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"  # adamw | adam
    scheduler: str = "cosine"  # cosine | onecycle | steplr | none
    warmup_steps: int = 500

    batch_size: int = 64
    max_steps: int = 5000
    eval_interval: int = 200
    seed: int = 3407

    results_dir: str = "results/exp_default"
    device: Optional[str] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(pref: Optional[str] = None) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class SmoothedValue:
    def __init__(self):
        self.values = []

    def update(self, v: float) -> None:
        self.values.append(float(v))
        if len(self.values) > 1000:
            self.values = self.values[-1000:]

    @property
    def avg(self) -> float:
        if not self.values:
            return 0.0
        return float(np.mean(self.values))


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def linear_warmup_cosine(step: int, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


