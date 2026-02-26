import torch
from torch import nn
from .config import Config


class MultiLayerPerceptron(nn.Module):
    """A Basic multi layer perceptron"""

    def __init__(self, conf: Config) -> None:
        super().__init__()
        self.lin_up = nn.Linear(conf.d_model, conf.d_hidden)
        self.gelu = nn.GELU()
        self.hiddens = nn.ModuleList(
            [nn.Linear(conf.d_hidden, conf.d_hidden) for i in range(conf.n_hidden)]
        )
        self.lin_down = nn.Linear(conf.d_hidden, conf.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin_up(x)
        out = self.gelu(out)
        for lin in self.hiddens:
            out = lin(out)
            out = self.gelu(out)
        out = self.lin_down(out)
        return out
