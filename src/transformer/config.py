from dataclasses import dataclass


@dataclass
class Config:
    """Aranges configuration data needed by various Transformer components"""

    d_model: int
    d_vocab: int
    d_hidden: int
    n_hidden: int  # number of linear layers in MLP
    d_head: int
    n_blocks: int  # number of transformer blocks per attention head
    n_context: int
    attn_bias: bool  # use bias in W linear layers
