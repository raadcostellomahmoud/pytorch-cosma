import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64, swin: bool=False):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"Image size {img_size} must be divisible by patch size {patch_size}")
        if not isinstance(img_size, int) or img_size <= 0:
            raise ValueError(f"Invalid img_size: {img_size}")
        
        self.patch_size = patch_size
        self.swin=swin
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                  kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        
        if self.swin:
            self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        else:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1, embed_dim))
    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x)        # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        if not self.swin:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            
        x += self.positional_embedding
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=64, num_layers=4, num_heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim)
        self.encoder = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        return self.classifier(x[:, 0])  # Classify using CLS token


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias"""
    def __init__(self, dim: int, num_heads: int, window_size: int, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        self._init_relative_positions()

    def _init_relative_positions(self):
        coords = torch.stack(torch.meshgrid(
            [torch.arange(self.window_size),
             torch.arange(self.window_size)]), dim=0).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1
        ).permute(2, 0, 1)
        attn += relative_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class SwinBlock(nn.Module):
    """Swin Transformer Block with shifted window attention"""
    def __init__(self, dim: int, num_heads: int, window_size: int, shifted: bool = False, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.shifted = shifted
        self.window_size = window_size

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows

    def window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = int(windows.shape[0] / (H * W / self.window_size ** 2))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int(N ** 0.5)  # Now N=64 for CIFAR-10 -> H=W=8
    
        assert H * W == N, f"Cannot reshape {N} patches into square grid"
        assert H >= self.window_size and W >= self.window_size, (
            f"Window size {self.window_size} exceeds feature map dimensions {H}x{W}"
        )
        x = x.view(B, H, W, C)  # [B, 8, 8, C]
        
        
        if self.shifted:
            x = torch.roll(x, shifts=(-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
            
        x = self.window_partition(x)
        x = x.view(-1, self.window_size * self.window_size, x.shape[-1])
        x = self.attn(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        
        x = self.window_reverse(x, H, W)
        if self.shifted:
            x = torch.roll(x, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))
            
        return x.view(x.shape[0], H * W, -1)

class PatchMerging(nn.Module):
    """Downsample feature resolution by 2x"""
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))