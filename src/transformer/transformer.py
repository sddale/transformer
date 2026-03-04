import torch
from torch import nn
from .config import Config
from .block import Block


class Transformer(nn.Module):
    """A basic transformer."""

    def __init__(self, conf: Config, device=torch.device("cpu")) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(conf.d_vocab, conf.d_model, device=device)
        self.pos_embed = torch.nn.Embedding(conf.n_context, conf.d_model, device=device)
        self.blocks = nn.ModuleList(
            [Block(conf).to(device) for _ in range(conf.n_blocks)]
        )

        self.lm_head = nn.Linear(conf.d_model, conf.d_vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embed(x) + self.pos_embed(torch.arange(x.size(1), device=x.device))
        for block in self.blocks:
            out = block(out)
        out = self.lm_head(out)
        return out
