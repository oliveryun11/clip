import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import VisionTransformer
from .text import TextTransformer
from config import CLIP_CONFIG

class CLIP(nn.Module):
    def __init__(
            self,
            image_size = CLIP_CONFIG['image_size'],
            patch_size = CLIP_CONFIG['patch_size'],
            width = CLIP_CONFIG['width'],
            layers = CLIP_CONFIG['layers'],
            heads = CLIP_CONFIG['heads'],
            vocab_size = CLIP_CONFIG['vocab_size'],
            max_seq_len = CLIP_CONFIG['max_seq_len'],
            temperature = CLIP_CONFIG['temperature'],
            dropout = CLIP_CONFIG['dropout']
    ):
        super().__init__()

        self.vision_transformer = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=width,
            num_heads=heads,
            num_layers=layers,
            dropout=dropout
        )

        self.text_transformer = TextTransformer(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=width,
            num_heads=heads,
            num_layers=layers,
            dropout=dropout
        )

        self.image_projection = nn.Linear(width, width)
        self.text_projection = nn.Linear(width, width)
        self.temperature = nn.Parameter(torch.ones([]) * temperature)

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        image_features = self.vision_transformer(image)
        text_features = self.text_transformer(text)

        image_features = self.image_projection(image_features)
        text_features = self.text_projection(text_features)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits = torch.matmul(image_features, text_features.transpose(-2, -1)) / self.temperature

        return logits

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        labels = torch.arange(len(logits), device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return (loss_i + loss_t) / 2

if __name__ == "__main__":
    # Initialize parameters
    batch_size = 2
    image_size = 224
    patch_size = 32
    width = 512
    vocab_size = 49408
    max_seq_len = 77

    # Create model
    model = CLIP(
        image_size=image_size,
        patch_size=patch_size,
        width=width,
        layers=6,
        heads=8,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len
    )

    # Create sample inputs
    images = torch.randn(batch_size, 3, image_size, image_size)
    texts = torch.randint(0, vocab_size, (batch_size, max_seq_len))

    with torch.no_grad():
        # Test full model
        logits = model(images, texts)
        print("\nFull Model Test:")
        print(f"Images shape: {images.shape}")
        print(f"Texts shape: {texts.shape}")
        print(f"Logits shape: {logits.shape}")  # Should be [2, 2]

        # Test individual components
        print("\nComponent Tests:")

        # 1. Vision Transformer
        image_features = model.vision_transformer(images)
        print("\nVision Transformer:")
        print(f"Output shape: {image_features.shape}")  # Should be [2, 512]

        # 2. Text Transformer
        text_features = model.text_transformer(texts)
        print("\nText Transformer:")
        print(f"Output shape: {text_features.shape}")  # Should be [2, 512]

        # 3. Projections
        image_proj = model.image_projection(image_features)
        text_proj = model.text_projection(text_features)
        print("\nProjection Layers:")
        print(f"Image projection shape: {image_proj.shape}")  # Should be [2, 512]
        print(f"Text projection shape: {text_proj.shape}")  # Should be [2, 512]

        # Verify shapes are correct
        expected_shapes = {
            "Images": (batch_size, 3, image_size, image_size),
            "Texts": (batch_size, max_seq_len),
            "Vision Features": (batch_size, width),
            "Text Features": (batch_size, width),
            "Image Projection": (batch_size, width),
            "Text Projection": (batch_size, width),
            "Logits": (batch_size, batch_size)
        }

        print("\nShape Verification:")
        all_correct = True
        actual_shapes = {
            "Images": images.shape,
            "Texts": texts.shape,
            "Vision Features": image_features.shape,
            "Text Features": text_features.shape,
            "Image Projection": image_proj.shape,
            "Text Projection": text_proj.shape,
            "Logits": logits.shape
        }

        for name, expected in expected_shapes.items():
            actual = actual_shapes[name]
            is_correct = actual == expected
            print(f"{name:<20} - Expected: {str(expected):<20} Got: {str(actual):<20} {'✓' if is_correct else '✗'}")
            all_correct = all_correct and is_correct

        print(f"\nAll shapes correct: {'✓' if all_correct else '✗'}")

        # Test loss computation
        loss = model.contrastive_loss(logits)
        print(f"\nContrastive Loss: {loss.item()}")