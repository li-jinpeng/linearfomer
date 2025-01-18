import torch
from torch import nn

from models import TimerBackbone


class Model(nn.Module):
    def __init__(self, ckpt_path: str):
        super().__init__()

        self.backbone = TimerBackbone.Model()
        # Decoder
        self.decoder = self.backbone.decoder
        self.proj = self.backbone.proj
        self.enc_embedding = self.backbone.patch_embedding

        print('loading model: ', ckpt_path)
        if ckpt_path.endswith('.pth'):
            self.backbone.load_state_dict(torch.load(ckpt_path))
        elif ckpt_path.endswith('.ckpt'):
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            sd = {k[6:]: v for k, v in sd.items()}
            self.backbone.load_state_dict(sd, strict=True)
        else:
            raise NotImplementedError

    def forward(self, x_enc):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc) # [B * M, N, D]

        # Transformer Blocks
        dec_out, attns = self.decoder(dec_in) # [B * M, N, D]
        dec_out = self.proj(dec_out) # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2) # [B, T, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means
        return dec_out
