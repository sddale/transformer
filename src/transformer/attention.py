import torch
from torch import nn
from .config import Config


class Attention(nn.Module):
    """Attention mechanism in its simplest form."""

    def __init__(self, conf: Config) -> None:
        super().__init__()
        self.WQ = nn.Linear(conf.d_model, conf.d_head, bias=conf.attn_bias)
        self.WK = nn.Linear(conf.d_model, conf.d_head, bias=conf.attn_bias)
        self.WV = nn.Linear(conf.d_model, conf.d_head, bias=conf.attn_bias)
        self.WO = nn.Linear(conf.d_head, conf.d_model, bias=conf.attn_bias)

        for layer in [self.WQ, self.WK, self.WV, self.WO]:
            self.__init_weights(layer, conf.attn_bias)

        self.sqrt_head = conf.d_head**0.5

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We are currently not specifying constant input size/context window
        # so we must make our mask each forward call
        mask = torch.triu(
            torch.ones(x.shape[1], x.shape[1], device=x.device), diagonal=1
        ).bool()

        # Projections
        q = self.WQ(x)
        k = self.WK(x)
        v = self.WV(x)

        scores = (q @ k.transpose(-2, -1)) / self.sqrt_head
        scores = scores.masked_fill(mask, float("-inf"))

        attn = self.softmax(scores)
        context = attn @ v

        return self.WO(context)

    def __init_weights(self, M, head_bias):
        torch.nn.init.xavier_uniform_(M.weight)
        if head_bias:
            torch.nn.init.zeros_(M.bias)
