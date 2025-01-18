from torch import nn

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_len = 96
        self.stride = 96
        self.d_model = 1024
        self.d_ff = 2048
        self.layers = 8
        self.n_heads = 8
        self.dropout = 0.1
        self.factor = 3
        self.activation = 'gelu'
        padding = 0

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, padding, self.dropout)

        # Decoder-only Transformer: Refer to issue: https://github.com/thuml/Large-Time-Series-Model/issues/23
        self.decoder = Encoder(
            [
                 EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, self.factor, attention_dropout=self.dropout,
                                      output_attention=True), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model)
        )

        # Prediction Head
        self.proj = nn.Linear(self.d_model, self.patch_len, bias=True)