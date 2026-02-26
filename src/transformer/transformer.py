import torch
from torch import nn
from .config import Config
from .block import Block


class Transformer(nn.Module):
    """A basic transformer."""

    def __init__(self, conf: Config, device=torch.device("cpu")) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(conf.d_vocab, conf.d_model, device=device)
        self.blocks = nn.ModuleList(
            [Block(conf).to(device) for i in range(conf.n_blocks)]
        )

        self.lm_head = nn.Linear(conf.d_model, conf.d_vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embed(x)
        # print(f"{out.shape=}")
        for block in self.blocks:
            out = block(out)
        out = self.lm_head(out)
        return out
