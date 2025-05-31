import timm, torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------
# resize the original position embedding from (24,24) to (24,32)
# -----------------------------------------------------------
def _resize_pos_embed(pos_embed, src_hw, dst_hw):
    """
    Resize the position embedding of shape (1, 1+H×W, C) 
    from src_hw to dst_hw.
    Both src_hw and dst_hw are tuples, e.g., (24,24) → (24,32).
    """
    cls, patch = pos_embed[:, :1], pos_embed[:, 1:]           # (1,1,C) + (1,N,C)
    patch = patch.reshape(1, *src_hw, -1).permute(0, 3, 1, 2) # (1,C,H,W)
    patch = F.interpolate(patch, dst_hw, mode='bilinear', align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, dst_hw[0] * dst_hw[1], -1)
    return torch.cat((cls, patch), dim=1)                     # (1,1+N',C)

# -----------------------------------------------------------
# model
# -----------------------------------------------------------
class RGBDRefineTransformerLarge(nn.Module):
    """
    ViT-Large (pretrained on ImageNet-21k) + 4-channel PatchEmbed
    1 keyframe + 1 target frame → CLS head → 6-DoF ∆ pose
    """
    def __init__(self,
                 img_size=(384, 512),
                 backbone='vit_small_patch16_224',
                 pretrained=True):
        super().__init__()

        # Load pretrained ViT backbone from timm
        vit = timm.create_model(backbone, pretrained=pretrained)

        # Patch size and token grid shape (H/patch, W/patch) = (24, 32)
        self.patch_size = vit.patch_embed.patch_size[0]       # int = 16
        self.grid_hw    = (img_size[0] // self.patch_size,    # (24,32)
                           img_size[1] // self.patch_size)

        # Replace patch embed to handle 4-channel input (RGB + depth)
        self.patch = timm.models.vision_transformer.PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=4,  # modified input channel count
            embed_dim=vit.embed_dim,
            norm_layer=vit.patch_embed.norm.__class__
        )

        # If pretrained, copy RGB weights and randomly init depth channel
        if pretrained:
            with torch.no_grad():
                self.patch.proj.weight[:, :3] = vit.patch_embed.proj.weight  # copy RGB weights
                nn.init.kaiming_normal_(self.patch.proj.weight[:, 3:4])      # init depth weights
                self.patch.proj.bias = vit.patch_embed.proj.bias             # copy bias

        # Embed dimension (C), e.g., 384 for vit_small
        self.embed_dim = vit.embed_dim                               # C
        self.cls_token = vit.cls_token                               # (1,1,C)

        # Resize and separate position embeddings
        resized_pe = _resize_pos_embed(
            vit.pos_embed,
            src_hw=(vit.patch_embed.img_size[0] // self.patch_size,
                    vit.patch_embed.img_size[1] // self.patch_size),  # (24,24)
            dst_hw=self.grid_hw                                       # (24,32)
        )
        self.pe_cls   = nn.Parameter(resized_pe[:, :1])   # (1,1,C) CLS token pos embed
        self.pe_patch = nn.Parameter(resized_pe[:, 1:])   # (1,N,C) patch tokens pos embed

        self.pos_drop = vit.pos_drop   # dropout layer before transformer
        self.blocks   = vit.blocks     # list of transformer blocks
        self.norm     = vit.norm       # final LayerNorm

        # Regression head for 6-DoF pose delta
        self.mlp_head = nn.Sequential(
            nn.Linear(self.embed_dim, 1024),  # (C → 1024)
            nn.GELU(),
            nn.Linear(1024, 6)               # output: (B,6)
        )

    # --------- helpers ---------
    def _prep(self, rgb, d):
        """Concatenate RGB and depth → patch embeddings: (B,4,H,W) → (B,N,C)"""
        x = torch.cat([rgb, d], 1)          # concat channel dim: (B,4,H,W)
        return self.patch(x)                # apply patch embed: (B,N,C)

    # --------- forward ---------
    def forward(self, rgb0, d0, rgbi, di, fid_emb=None):
        """
        Forward pass:
        - rgb0, d0: keyframe RGB and depth (B,3,H,W), (B,1,H,W)
        - rgbi, di: target frame RGB and depth (B,3,H,W), (B,1,H,W)
        - fid_emb: optional frame index embedding (B,C)
        """
        t0 = self._prep(rgb0, d0)           # (B,N,C)
        ti = self._prep(rgbi, di)           # (B,N,C)
        tokens = torch.cat([t0, ti], 1)     # concat views: (B,2N,C)
        B = tokens.size(0)

        cls = self.cls_token.expand(B, -1, -1)         # (B,1,C), shared global token
        x   = torch.cat((cls, tokens), 1)              # add to front: (B,1+2N,C)

        # ====== Add Positional Embedding ======
        pos = torch.cat((
          self.pe_cls.expand(B, -1, -1),          # (B,1,C)
          self.pe_patch.repeat(B, 2, 1)           # (B,2N,C)
        ), dim=1)                                 # (B,1+2N,C)
        x = x + pos
        if fid_emb is not None:
            x[:, 1:] += fid_emb[:, None, :]       # only patch tokens (B,2N,C)

        x = self.pos_drop(x)                      # apply dropout

        for blk in self.blocks:
            x = blk(x)                            # transformer blocks
        x = self.norm(x)                          # final norm

        delta = self.mlp_head(x[:, 0])            # regress pose from CLS token (B,6)
        return delta
