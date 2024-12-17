import torch
import torch.nn as nn

from .vision import TransformerBlock
from config import TEXT_CONFIG

class TextTransformer(nn.Module):
    def __init__(
            self,
            vocab_size = TEXT_CONFIG['vocab_size'],
            max_seq_len = TEXT_CONFIG['max_seq_len'],
            embed_dim = TEXT_CONFIG['embed_dim'],
            num_layers = TEXT_CONFIG['num_layers'],
            num_heads = TEXT_CONFIG['num_heads'],
            dropout = TEXT_CONFIG['dropout'],
            mlp_ratio = TEXT_CONFIG['mlp_ratio'],
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoding(
            max_seq_len = max_seq_len,
            embed_dim = embed_dim,
            dropout = dropout,
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim = embed_dim,
                num_heads = num_heads,
                mlp_ratio = mlp_ratio,
                dropout = dropout,
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)

        return x[:, 0]

class PositionalEncoding(nn.Module):
    def __init__(
            self,
            max_seq_len: int,
            embed_dim: int,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, max_seq_len + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, seq_len)
        """
        batch_size = x.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embeddings
        x = self.dropout(x)

        return x

if __name__ == "__main__":
    batch_size = 2
    vocab_size = 10000
    max_seq_len = 77
    embed_dim = 512

    model = TextTransformer(
        vocab_size = vocab_size,
        max_seq_len = max_seq_len,
        embed_dim = embed_dim,
        num_layers = 6,
        num_heads = 8,
        mlp_ratio = 4.0,
    )

    text = torch.randint(0, vocab_size, (batch_size, max_seq_len))

    with torch.no_grad():
        # Test full model
        output = model(text)
        print("\nFull Model Test:")
        print(f"Input shape: {text.shape}")
        print(f"Output shape: {output.shape}")  # Should be [2, 512]
        
        # Test individual components
        print("\nComponent Tests:")
        
        # 1. Token Embedding
        token_embed = model.token_embedding(text)
        print("\nToken Embedding:")
        print(f"Output shape: {token_embed.shape}")  # Should be [2, 77, 512]
        
        # 2. Position Encoding (with CLS token)
        pos_embed = model.pos_embedding(token_embed)
        print("\nPosition Encoding (with CLS token):")
        print(f"Output shape: {pos_embed.shape}")  # Should be [2, 78, 512]
        
        # 3. First Transformer Block
        block_out = model.transformer_blocks[0](pos_embed)
        print("\nTransformer Block:")
        print(f"Output shape: {block_out.shape}")  # Should be [2, 78, 512]
        
        # 4. Final Layer Norm
        normalized = model.norm(block_out)
        print("\nFinal Layer Norm:")
        print(f"Output shape: {normalized.shape}")  # Should be [2, 78, 512]
        
        # Verify shapes are correct
        expected_shapes = {
            "Input": (batch_size, max_seq_len),
            "Token Embedding": (batch_size, max_seq_len, embed_dim),
            "Position Encoding": (batch_size, max_seq_len + 1, embed_dim),  # +1 for CLS token
            "Transformer Block": (batch_size, max_seq_len + 1, embed_dim),
            "Final Output": (batch_size, embed_dim)  # CLS token only
        }
        
        print("\nShape Verification:")
        all_correct = True
        actual_shapes = {
            "Input": text.shape,
            "Token Embedding": token_embed.shape,
            "Position Encoding": pos_embed.shape,
            "Transformer Block": block_out.shape,
            "Final Output": output.shape
        }
        
        for name, expected in expected_shapes.items():
            actual = actual_shapes[name]
            is_correct = actual == expected
            print(f"{name:<20} - Expected: {str(expected):<20} Got: {str(actual):<20} {'✓' if is_correct else '✗'}")
            all_correct = all_correct and is_correct
        
        print(f"\nAll shapes correct: {'✓' if all_correct else '✗'}")