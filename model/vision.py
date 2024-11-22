import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class VisionTransformer(nn.Module):
    """
    complete vision transformer for image processing.

    flow:
    1. image -> patches -> patch embeddings
    2. add positional encoding and cls token
    3. transformer encoder layers
    4. layer normalize and return cls token
    """
    def __init__(
            self,
            image_size = 224,
            patch_size = 32,
            in_channels = 3,
            embed_dim = 512,
            num_heads = 8,
            num_layers = 6,
            mlp_ratio = 4.0,
            dropout = 0.1
    ):
        """
        Args:
            image_size: height and width of the input image
            patch_size: height and width of each patch
            in_channels: number of channels in the input image
            embed_dim: dimension of the embedding vector
            num_heads: number of attention heads
            num_layers: number of transformer layers
            mlp_ratio: ratio of the dimension of the hidden layer to the embedding dimension
            dropout: dropout rate
        """
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_size = image_size,
            patch_size = patch_size,
            in_channels = in_channels,
            embed_dim = embed_dim
        )

        self.pos_embed = PositionalEncoding(
            num_patches = self.patch_embed.num_patches,
            embed_dim = embed_dim,
            dropout = dropout
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim = embed_dim,
                num_heads = num_heads,
                mlp_ratio = mlp_ratio,
                dropout = dropout
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, height, width)
        """
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        
        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        return x[:, 0]

class PatchEmbedding(nn.Module):
    """
    patch embedding module takes patches (blocks) of an input image and then
    converts each patch into a one-dimensional vector.
    This prepares the image for input into the transformer.
    """
    def __init__(
            self, 
            image_size = 224, 
            patch_size = 32, 
            in_channels = 3, 
            embed_dim = 512
    ):
        """
        Args:
            image_size: height and width of the input image
            patch_size: height and width of each patch
            in_channels: number of channels in the input image
            embed_dim: dimension of the embedding vector
        """
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size = patch_size,
            stride = patch_size,
        )

    def forward(self, x):
        """
        Args:
            x: Input image tensor with shape (batch_size, in_channels, height, width)
        """
        x = self.projection(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        return x

class PositionalEncoding(nn.Module):
    """
    adds positional information and a CLS token to patch embeddings.

    The CLS token is a special learnable vector prepended to the sequence.
    It is used to capture the global image context.

    Positional embeddings are added to give the transformer information about 
    the spatial position of each patch. The transformer model does not inherently
    understand spatial information.
    """
    def __init__(
            self,
            num_patches,
            embed_dim = 512,
            dropout = 0.1
    ):
        """
        Args:
            embed_dim: dimension of the embedding vector
            dropout: dropout rate
        """
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: output from the patch embedding module of shape (batch_size, num_patches, embed_dim)
        """
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim = 1)
        x += self.pos_embeddings
        x = self.dropout(x)

        return x

class TransformerEncoder(nn.Module):
    """
    multi-head self attention mechanism

    allows the model to jointly attend to information from different representation
    subspaces at different positions. Each head can learn a different represntation.
    """
    def __init__(
            self,
            embed_dim = 512,
            num_heads = 8,
            dropout = 0.1
    ):
        """
        Args:
            embed_dim: dimension of the embedding vector
            num_heads: number of attention heads
            num_layers: number of transformer layers
            dropout: dropout rate
        """
        super().__init__()

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, num_patches + 1, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(
            t,
            'b n (h d) -> b h n d',
            h = self.num_heads
        ), qkv)

        scale = self.head_dim ** -0.5
        attention = torch.matmul(q, k.transpose(-2, -1)) * scale
        attention = F.softmax(attention, dim = -1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)

        return out

class MLPBlock(nn.Module):
    """
    multilayer perceptron block following the attention layer.

    allows each token to process information independently.
    typically expands dimension, applies non-linearity, then contracts dimension.
    """
    def __init__(
            self,
            embed_dim = 512,
            mlp_ratio = 4.0,
            dropout = 0.1
    ):
        """
        Args:
            embed_dim: dimension of the embedding vector
            mlp_ratio: ratio of the expanded dimension to the original dimension
            dropout: dropout rate
        """
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, num_patches + 1, embed_dim)
        """
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    full transformer block combining attention and mlp layers.

    architecture:
    input -> layernorm -> attention -> residual connection -> layernorm -> mlp -> residual connection -> output

    residual connections (adding input of a layer to its output) help in stabilizing the learning process
    and prevent overfitting.
    """
    def __init__(
            self,
            embed_dim = 512,
            num_heads = 8,
            mlp_ratio = 4.0,
            dropout = 0.1
    ):
        """
        Args:
            embed_dim: dimension of the embedding vector
            num_heads: number of attention heads
            mlp_ratio: ratio of the expanded dimension to the original dimension
            dropout: dropout rate
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = TransformerEncoder(
            embed_dim = embed_dim,
            num_heads = num_heads,
            dropout = dropout
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(
            embed_dim = embed_dim,
            mlp_ratio = mlp_ratio,
            dropout = dropout
        )

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, num_patches + 1, embed_dim)
        """
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

if __name__ == "__main__":
    model = VisionTransformer()
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        # Test full model
        output = model(image)
        print("\nFull Model Test:")
        print(f"Input shape: {image.shape}")
        print(f"Output shape: {output.shape}")  # Should be [2, 512]
        
        # Test individual components
        print("\nComponent Tests:")
        
        # 1. Patch Embedding
        patch_embed = model.patch_embed(image)
        print("\nPatch Embedding:")
        print(f"Output shape: {patch_embed.shape}")  # Should be [2, 49, 512]
        print(f"Number of patches: {model.patch_embed.num_patches}")  # Should be 49
        
        # 2. Position Encoding
        pos_embed = model.pos_embed(patch_embed)
        print("\nPosition Encoding (with CLS token):")
        print(f"Output shape: {pos_embed.shape}")  # Should be [2, 50, 512]
        
        # 3. First Transformer Block
        block_out = model.transformer_blocks[0](pos_embed)
        print("\nTransformer Block:")
        print(f"Output shape: {block_out.shape}")  # Should be [2, 50, 512]
        
        # 4. Final Layer Norm
        normalized = model.norm(block_out)
        print("\nFinal Layer Norm:")
        print(f"Output shape: {normalized.shape}")  # Should be [2, 50, 512]
        
        # Verify shapes are correct
        expected_shapes = {
            "Input": (batch_size, 3, 224, 224),
            "Patch Embedding": (batch_size, 49, 512),
            "Position Encoding": (batch_size, 50, 512),  # +1 for CLS token
            "Transformer Block": (batch_size, 50, 512),
            "Final Output": (batch_size, 512)  # CLS token only
        }
        
        print("\nShape Verification:")
        all_correct = True
        actual_shapes = {
            "Input": image.shape,
            "Patch Embedding": patch_embed.shape,
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