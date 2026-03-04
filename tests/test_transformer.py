import torch
import pytest
from transformer.config import Config
from transformer.attention import Attention
from transformer.block import Block
from transformer.transformer import Transformer


@pytest.fixture
def config():
    """Provides a small, fast config for testing."""
    return Config(
        d_model=10,
        d_vocab=100,
        d_hidden=4 * 10,
        n_hidden=0,
        d_head=10,
        n_blocks=2,
        n_context=10,
        attn_bias=True,
    )


def test_attention_shapes(config):
    """Ensure attention preserves input shape."""
    attn = Attention(config)
    x = torch.randn(
        config.d_model, config.n_context, config.d_model
    )  # [Batch, Seq, Channels]
    y = attn(x)
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"


def test_transformer_block_forward(config):
    """Verify a single block passes data through without shape changes."""
    block = Block(config)
    x = torch.randn(config.d_model, config.n_context, config.d_model)
    y = block(x)
    assert y.shape == x.shape


def test_full_transformer_output_logits(config):
    """Check if the full model outputs the correct vocabulary distribution."""
    model = Transformer(config)
    idx = torch.randint(0, config.d_vocab, (1, 10))  # [Batch, Seq]
    logits = model(idx)

    # Logits should be [Batch, Seq, Vocab_size]
    assert logits.shape == (1, 10, config.d_vocab)


def test_model_overfit_batch(config):
    """A 'smoke test' to see if a single step of backprop works."""
    model = Transformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

    vals = torch.randint(0, config.d_vocab, (8, 8))
    xs = vals[:, :-1]
    ys = vals[:, 1:]

    # Forward + Backward
    logits = model(xs)
    loss = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), ys, ignore_index=-1
    )
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
    assert not torch.isnan(loss)
