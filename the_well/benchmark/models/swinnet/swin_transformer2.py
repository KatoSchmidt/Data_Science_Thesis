import torch
from torch import nn
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 


# SwinU-net Model
class SwinTransformerUnet(nn.Module):
    """
    U-Net style architecture with Swin Transformer blocks.

    Components:
    - Patch embedding for input
    - 4-stage encoder with Swin blocks and PatchMerging
    - Bottleneck with 2 Swin blocks (no resolution change)
    - 4-stage decoder with PatchExpanding and skip connections
    - Final patch expanding and Conv2d projection to output fields
    """

    def __init__(
        self,
        img_size=(128, 384),
        patch_size=4,
        in_chans=16,
        num_output_fields=4,  #velocity_x, velocity_y, pressure, temperature
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        num_bottleneck_blocks=2
    ):
        super().__init__()

        self.num_output_fields = num_output_fields
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Initial patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        # Calculate resolution after patch embed
        self.patches_resolution = self.patch_embed.grid_size

        # Drop path rates per block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Encoder
        self.encoder_layers = nn.ModuleList()
        input_dim = embed_dim
        resolution = self.patches_resolution
        for i_layer in range(self.num_layers):
            layer = EncoderStage(
                dim=input_dim,
                input_resolution=resolution,
                depth=depths[i_layer],
                downsample=True,
                block_class=Block,
                block_kwargs=dict(
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    input_resolution=resolution,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                ),
                patch_merging_class=PatchMerging
            )
            self.encoder_layers.append(layer)
            resolution = (resolution[0] // 2, resolution[1] // 2)
            input_dim *= 2


        # Bottleneck 
        self.bottleneck = nn.Sequential(*[
                Block(
                    dim=input_dim,
                    num_heads=num_heads[-1],
                    window_size=window_size,
                    input_resolution=resolution,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    norm_layer=norm_layer
                ) for _ in range(num_bottleneck_blocks)])
    
        # Decoder 
        self.decoder_layers = nn.ModuleList()
        for i_layer in reversed(range(self.num_layers)):
            dim_in = input_dim          
            input_dim = input_dim // 2 
            dim_skip = input_dim       
            dim_out = input_dim  
 
            layer = DecoderStage(
                dim_in=dim_in,
                dim_skip=dim_skip,
                dim_out=dim_out,
                input_resolution=resolution,
                depth=depths[i_layer],
                block_class=Block,
                patch_expand_class=PatchExpanding,
                block_kwargs=dict(
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    input_resolution=resolution,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    norm_layer=norm_layer,
                ),
            )
            self.decoder_layers.append(layer)

            # Update voor volgende iteratie
            input_dim = dim_out
            resolution = (resolution[0] * 2, resolution[1] * 2)


        # Final upsampling + projection
        self.final_upsample1 = FinalPatchExpanding(input_resolution = resolution, dim = input_dim)
        resolution = (resolution[0] * 2, resolution[1] * 2)
        self.final_upsample2 = FinalPatchExpanding(input_resolution = resolution, dim = input_dim)
        self.final_projection = LinearProjection(in_channels=embed_dim, out_channels=num_output_fields)

        #Final Skip Connection
        self.skip_conv = nn.Conv2d(embed_dim + in_chans, embed_dim, kernel_size=1)

    def forward(self, x):
        """
        x: [B, in_chans, H, W]
        returns: [B, num_output_fields, H, W]
        """
        skips = []
        # print(f"[Input] x: {x.shape}")
        skips.append(x)

        # 1. Patch Partition
        x = self.patch_embed(x)
        B, L, C = x.shape
        H, W = self.patches_resolution
        # print(f"[PatchEmbed] x: {x.shape} | Grid: {self.patches_resolution}")

        # 2. Encoder
       
        for i, layer in enumerate(self.encoder_layers):
            skips.append(x)
            # print(f"[Encoder Stage {i + 1} input] x: {x.shape}, H: {H}, W: {W}")
            x, H, W = layer(x, H, W)
            # print(f"[Encoder Stage {i + 1} output] x: {x.shape}, H: {H}, W: {W}")
        
        # print("skips shape list: ", [x.shape for x in skips])

        # # 3. Bottleneck
        # print(f"[Bottleneck Shape] x: {x.shape}, H: {H}, W: {W}")
        for block in self.bottleneck:
            block.H = H
            block.W = W
            x = block(x, mask_matrix=None)
        # print(f"[After Bottleneck] x: {x.shape}, H: {H}, W: {W}")

        # 4. Decoder
        for i, layer in enumerate(self.decoder_layers):
            skip = skips[-(i+1)]
            # print("skip shape: ", skip.shape)
            # print(f"[Decoder Stage {i + 1} input]  x: {x.shape}, skip: {skip.shape}, H: {H}, W: {W}")
        
            x, H, W = layer(x, skip, H, W)
            # print(f"[Decoder Stage {i + 1} output]  x: {x.shape}, H: {H}, W: {W}")

        # 5. Final upsample
        # print(f"[Before Final Upsample] x: {x.shape}, H: {H}, W: {W} ")
        x = self.final_upsample1(x)
        H, W = H * 2, W * 2
        # print(f"[After final_upsample_1] x: {x.shape}, H: {H}, W: {W}")
        x = self.final_upsample2(x)
        H, W = H * 2, W * 2
        # print(f"[After final_upsample_2] x: {x.shape}, H: {H}, W: {W}")

        # x: [B, H*W, C] na final_upsample2
        x_img = skips[0]  # [B, in_chans, 128, 384]
        x_img = rearrange(x_img, 'b c h w -> b (h w) c', h=H, w=W)
        # print(" x skip: ", x_img.shape, " x.shape", x.shape)

        # eventueel eerst matchen in_channels met embed_dim via conv:
        x = torch.cat([x, x_img], dim=2)  # concat op kanaaldimensie
        # print("x.shape after cat", x.shape)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        x = self.skip_conv(x)
        # print("x.shape after skip", x.shape)

        x = self.final_projection(x, H, W)
        # print(f"[After final_projection] x: {x.shape}, H: {H}, W: {W}")
        return x


# Encoder Downsampling Stage, Decoder Upsampling Stage 
class EncoderStage(nn.Module):
    """
    One stage of the Swin-Unet encoder:
    - Applies a sequence of Swin Transformer blocks
    - Optionally performs Patch Merging (i.e., downsampling)

    Args:
        dim (int): Input feature dimension
        input_resolution (tuple): Resolution of the feature map (H, W)
        depth (int): Number of Swin Transformer blocks
        downsample (bool): Whether to apply PatchMerging after the blocks
        block_class (nn.Module): The SwinTransformer block implementation
        **block_kwargs: Additional args to be passed to each block
    """
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        downsample=True,
        block_class=None,
        block_kwargs=None,
        patch_merging_class=None
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            block_class(
                dim=dim,
                drop_path=block_kwargs["drop_path"][i] if isinstance(block_kwargs["drop_path"], list) else block_kwargs["drop_path"],
                **{k: v for k, v in block_kwargs.items() if k != "drop_path"}
            ) for i in range(depth)])

        # print("patch_merging_class(input_resolution, dim): ", downsample)
        # self.downsample = patch_merging_class(input_resolution, dim) if downsample else None
        self.downsample = patch_merging_class(dim, 2 * dim) if downsample else None


    def forward(self, x, H, W, attn_mask=None):
        """
        x: [B, H*W, C]
        H, W: spatial dimensions
        attn_mask: attention mask for shifted windows
        """
        for blk in self.blocks:
            blk.H = H  # Needed inside Swin block
            blk.W = W
            x = blk(x, mask_matrix = attn_mask)
        
        # print("encoder downsample: ", self.downsample)
        if self.downsample is not None:
            # x = self.downsample(x)
            # H, W = H // 2, W // 2
            x, H, W = self.downsample(x, H, W)

        return x, H, W

class DecoderStage(nn.Module):
    """
    One stage of the Swin-Unet decoder:
    - Applies PatchExpanding (2× upsampling)
    - Concatenates with skip connection
    - Reduces concatenated channels via Linear
    - Applies multiple Swin Transformer blocks

    Args:
        dim_in (int): Input channels from previous decoder output
        dim_skip (int): Channels from corresponding encoder skip connection
        dim_out (int): Desired output dimension after fusion
        input_resolution (tuple[int, int]): Resolution before upsampling (H, W)
        depth (int): Number of Swin Transformer blocks
        block_class (nn.Module): Swin Transformer block class
        patch_expand_class (nn.Module): PatchExpanding class
        block_kwargs (dict): Arguments for Swin blocks
    """

    def __init__(
        self,
        dim_in,
        dim_skip,
        dim_out,
        input_resolution,
        depth,
        block_class,
        patch_expand_class,
        block_kwargs=None,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.upsample = patch_expand_class(input_resolution, dim_in)
        self.concat_linear = nn.Linear(dim_in, dim_out)
        # print("dim_in + dim_skip: ", dim_in, dim_skip)
        # print("dim_out: ", dim_out)
        self.norm = norm_layer(dim_out)
        self.blocks = nn.ModuleList([
            block_class(dim=dim_out, **block_kwargs) for _ in range(depth)
        ])

    def forward(self, x, skip, H, W, attn_mask=None):
        # print("shape x before patch expanding: ", x.shape)
        x = self.upsample(x)  # [B, L_up, C]
        H, W = H * 2, W * 2
        x = x.reshape(x.size(0), H * W , -1)  # expliciet L = H*2 × W*2
        # print("shape x after patch expanding: ", x.shape)
 
        x = torch.cat([x, skip], dim=-1)
        # print("X.shape after concat: ", x.shape)

        x = self.concat_linear(x)
        x = self.norm(x)

        for blk in self.blocks:
            blk.H = H
            blk.W = W
            x = blk(x, mask_matrix=attn_mask)

        return x, H, W


# Patch Partitioning + Linear Embedding, Patch Merging, Patch Expanding, Final Patch Expanding + Linear Projection
class PatchEmbed(nn.Module):
    """ Patch Partitioning + Linear Embedding """
    def __init__(self, img_size, patch_size, in_chans,  embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        ## Patchsize 
        self.patch_size = to_2tuple(patch_size)
        
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)  # shape: [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # shape: [B, num_patches, embed_dim]
        x = self.norm(x)
        return x

# class PatchMerging(nn.Module):
#     """ Patch Merging """
#     def __init__(self, input_resolution, dim):
#         super().__init__()
#         self.input_resolution = input_resolution  # [H, W]
#         self.dim = dim

#         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # channel reduction
#         #self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=2, stride=2)
#         self.norm = nn.LayerNorm(4 * dim)

#     def forward(self, x):
#         """
#         x: [B, H*W, C]
#         """
#         B, L, C = x.shape
#         H, W = self.input_resolution
#         # print( B, L, C, H, W)
#         assert L == H * W, "Input feature has wrong size"

#         x = x.reshape(B, H, W, C)

#         # Split in 2x2 patches
#         x0 = x[:, 0::2, 0::2, :]  # top-left
#         x1 = x[:, 1::2, 0::2, :]  # bottom-left
#         x2 = x[:, 0::2, 1::2, :]  # top-right
#         x3 = x[:, 1::2, 1::2, :]  # bottom-right

#         # Concatenate along channel dim
#         x = torch.cat([x0, x1, x2, x3], dim=-1)  # shape: [B, H/2, W/2, 4*C]
#         x = x.reshape(B, -1, 4 * C)  # flatten again
#         x = self.norm(x.contiguous())
#         x = self.reduction(x)  # [B, H*W/4, 2*C]
#         return x
    
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, H, W):
        B, L, C = x.shape

        x = x.transpose(1, 2).view(B, C, H, W)         # → [B, C, H, W]
        x = self.conv(x)                               # downsampling
        H, W = H // 2, W // 2
        x = x.flatten(2).transpose(1, 2)               # → [B, H*W, C']
        x = self.norm(x)
        return x, H, W

    
# class PatchExpanding(nn.Module):
#     """ Patch Expanding """
#     def __init__(self, input_resolution, dim, expand_ratio=2, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
        
#         self.expand = nn.Linear(dim, expand_ratio * dim, bias=False)
#         self.norm = norm_layer(dim // expand_ratio)

#     def forward(self, x):
#         B, L, C = x.shape
#         H, W = self.input_resolution
#         assert L == H * W, "wrong input shape"
#         x = self.expand(x)  # [B, L, 2*C]
#         x = x.view(B, H, W, -1)
#         x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1 w p2) c', p1=2, p2=2, c=C // 2)
#         x = self.norm(x)
#         return x
    
class PatchExpanding(nn.Module):
    """ Patch Expanding """
    def __init__(self, input_resolution, dim, expand_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        
        # self.expand = nn.Linear(dim, expand_ratio * dim, bias=False)
        self.expand = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)

        self.norm = norm_layer(dim // expand_ratio)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "wrong input shape"

        # Reshape naar beeldvorm voor conv
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        
        x = self.expand(x)  # [B, C//2, H*2, W*2]
        
        H, W = H * 2, W * 2
        x = x.flatten(2).transpose(1, 2)  # terug naar [B, L, C]
        x = self.norm(x)

        return x

    
# class FinalPatchExpanding(nn.Module):
#     """ Patch Expanding """
#     def __init__(self, input_resolution, dim, expand_ratio=4, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
#         self.expand = nn.Linear(dim, expand_ratio * dim, bias=False)
#         #self.expand = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)

#         self.norm = norm_layer(dim)

#     def forward(self, x):
#         B, L, C = x.shape
#         H, W = self.input_resolution
#         # print(B, L, C, H, W)

#         assert L == H * W, "wrong input shape"
#         # print("dim: ", self.dim)

#         x = self.expand(x)  # [B, L, 2*C]
#         x = x.view(B, H, W, -1)
#         # print("x.shape: ", x.shape)
#         x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1 w p2) c', p1=2, p2=2, c=C )
#         # print("x.shape before norm: ", x.shape)
#         x = self.norm(x)
#         return x
    
class FinalPatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim, expand_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.norm = norm_layer(dim)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        # print(B , L , C, H , W)
        assert L == H * W, "wrong input shape"

        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.expand(x)
        H, W = H * 2, W * 2
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x



class LinearProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.output_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x, H, W):
        """
        x: [B, H*W, C]
        returns: [B, out_channels, H, W]
        """

        if x.dim() == 3: 
            B, L, C = x.shape
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.output_proj(x)
        return x

        

# SWIN TRANSFORMER BLOCK -> Block,  window_partition,  window_reverse, Mlp, WindowAttention 
class Block(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, input_resolution=(128, 384), shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if isinstance(window_size, int):
            self.window_size = min(window_size, input_resolution[0], input_resolution[1])
        else:
            self.window_size = (
                min(window_size[0], input_resolution[0]),
                min(window_size[1], input_resolution[1])
            )
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # print("norm_layer(dim): ", dim)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.reshape(B, H * W, C)
        x = self.norm1(x.contiguous())
        x = x.reshape(B, H, W, C)  # terug naar (B, H, W, C) om verder te werken


        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.reshape(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, -1)
    return x
    
class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
        x: input features with shape of (num_windows*B, N, C)
        mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None """

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B_, num_heads, N, N]

        relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1)  # [num_heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)  # [B_, num_heads, N, N]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

