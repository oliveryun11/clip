import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vision import VisionTransformer
from model.text import TextTransformer

class CLIP(nn.Module):
    def __init__(
            self,
            image_size: int = 224,
            patch_size: int = 32,
            width: int = 512,
            layers: int = 6,
            heads: int = 8,
            vocab_size: int = 49408,
            max_seq_len: int = 77,
            temperature: float = 0.07,
            dropout: float = 0.1
            ) -> None:
        super().__init__()

        self.vision_transformer = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=width,
            num_heads=heads,
            num_layers=layers,
            dropout=dropout
        )

        self.termperature = nn.Parameter(torch.ones([]) * temperature)
