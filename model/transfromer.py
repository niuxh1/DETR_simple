import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_encoder_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_decoder_layers
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # Flatten and permute to match Transformer input shape
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # Flatten and permute pos_embed similarly
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # Repeat query_embed for each batch
        mask = mask.flatten(1)  # Flatten mask to match the expected shape

        # Add position encoding to the source and target
        src = src + pos_embed  # Add position encoding to src
        tgt = torch.zeros_like(query_embed)  # Initialize target for the decoder

        tgt = tgt + query_embed  # Add position encoding to tgt

        memory = self.encoder(src, src_key_padding_mask=mask)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          )
        return hs


if __name__ == "__main__":
    transformer = Transformer().to('cuda')
    transformer.eval()
    # 输入数据
    src = torch.rand(1, 512, 1, 1).to('cuda')  # (batch_size, channels, height, width)
    mask = torch.randn(1, 1, 1) .to('cuda')> 0   # (batch_size, height, width)
    query_embed = torch.rand(100, 512) .to('cuda')  # (num_queries, d_model)
    pos_embed = torch.rand(1,512,1, 1) .to('cuda') # (height, width)

    # 修改 pos_embed 的形状，使其符合输入要求# (batch_size, height, width)

    # 输出结果
    out = transformer(src, mask, query_embed, pos_embed)
    print(out)
