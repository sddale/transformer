import torch
from torch import nn
from .config import Config


class Attention(nn.Module):
    """Attention mechanism in its simplest form."""

    def __init__(self, conf: Config) -> None:
        super().__init__()
        self.WQ = nn.Linear(conf.d_model, conf.d_head, bias=conf.attn_bias)
        self.__init_weights(self.WQ, conf.attn_bias)

        self.WK = nn.Linear(conf.d_model, conf.d_head, bias=conf.attn_bias)
        self.__init_weights(self.WK, conf.attn_bias)

        self.WV = nn.Linear(conf.d_model, conf.d_model, bias=conf.attn_bias)
        self.__init_weights(self.WV, conf.attn_bias)

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
        return attn @ v

    def __init_weights(self, M, head_bias):
        torch.nn.init.xavier_uniform_(M.weight)
        if head_bias:
            torch.nn.init.zeros_(M.bias)
