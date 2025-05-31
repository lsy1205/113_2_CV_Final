import timm
import torch
import torch.nn as nn
class RGBDRefineTransformer(nn.Module):
    def __init__(self, img_size=(384,512), patch=(16,16), dim=384, depth=6, heads=6):
        super().__init__()
        self.patch = timm.models.vision_transformer.PatchEmbed(
            img_size=img_size, patch_size=patch, in_chans=4, embed_dim=dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, depth)
        self.cls = nn.Parameter(torch.zeros(1,1,dim))
        self.mlp = nn.Sequential(nn.LayerNorm(dim),
                                 nn.Linear(dim,512), nn.GELU(),
                                 nn.Linear(512,6))
    def _prep(self, rgb, d):
        x = torch.cat([rgb,d],1)             # (B,4,H,W)
        tokens = self.patch(x)               # (B,N,dim)
        return tokens
    def forward(self, rgb0,d0,rgbi,di,fid_emb=None):
        t0 = self._prep(rgb0,d0)
        ti = self._prep(rgbi,di)
        tokens = torch.cat([t0,ti],1)
        B = tokens.size(0)
        cls = self.cls.expand(B,-1,-1)
        if fid_emb is not None:
            tokens = tokens + fid_emb[:,None,:]
        x = torch.cat([cls,tokens],1)        # (B,1+2N,dim)
        x = self.encoder(x)[:,0]             # CLS
        delta = self.mlp(x)                  # (B,6)
        return delta
