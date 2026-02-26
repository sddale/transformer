import torch
from torch import nn
from .config import Config


class Attention(nn.Module):
    """Attention mechanism in its simplest form."""

    def __init__(self, conf: Config) -> None:
        super().__init__()
        self.WQ = nn.Linear(conf.d_model, conf.d_head, bias=conf.attn_bias)
        self.WK = nn.Linear(conf.d_model, conf.d_head, bias=conf.attn_bias)
        self.WV = nn.Linear(conf.d_model, conf.d_model, bias=conf.attn_bias)
        self.d_model = conf.d_model
        torch.nn.init.xavier_uniform_(self.WV.weight)
        torch.nn.init.zeros_(self.WV.bias)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We are currently not specifying constant input size/context window
        # so we must make our mask each forward call
        M = torch.full((x.shape[1], x.shape[1]), float("-inf"), device=x.device)
        M = torch.triu(M, diagonal=1)

        q = self.WQ(x)
        k = self.WK(x)
        v = self.WV(x)
        attn = self.softmax(q @ (k.transpose(-2, -1)) + M)
        # print(f"attn{q=}")
        # print(f"attn{k=}")
        # print(f"attn{x=}")
        # print(f"attn{v=}")
        # print(f"attn{attn=}")
        # print(f"attn{out=}")
        return attn @ v
