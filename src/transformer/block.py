from torch import nn
from .config import Config
from .attention import Attention
from .multi_layer_perceptron import MultiLayerPerceptron


class Block(nn.Module):
    """A transformer block comprised of attention and mlp."""

    def __init__(self, conf: Config) -> None:
        super().__init__()
        self.attention = Attention(conf)
        self.mlp = MultiLayerPerceptron(conf)

    def forward(self, x):
        res = x + self.attention(x)
        return res + self.mlp(res)
