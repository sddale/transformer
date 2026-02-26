import torch
from torch.utils.tensorboard import SummaryWriter
from transformer.transformer import Transformer
from transformer.utils import train, load_data, get_loaders, generate
from transformer.config import Config
from transformer.tokenizer import Tokenizer


def main():
    device = (
        torch.device("mps")
        if torch.mps.is_available()
        else torch.device("cuda")  # untested
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"{device=}")

    # Load kaggle dataset
    # Do not "force_remake" unless you have a few hours to wait
    data = " ".join(load_data(force_remake=False))

    # Create tokenizer
    tk = Tokenizer(data)

    # Setup some model hyperparameters
    conf = Config(
        # d_model=200,
        d_model=128,
        d_vocab=len(tk.vocab_dict.keys()),
        d_hidden=4 * 128,
        n_hidden=0,
        # d_head=100,
        d_head=128,
        n_blocks=2,
        n_context=300,
        attn_bias=True,
    )

    # Encode and split data
    data_encoded = tk.make_batches(batch_size=conf.n_context)

    # Convert to pytorch dataloader
    loader_tr = get_loaders(data_encoded, batch_size=32)

    # Instantiate transformer network
    trans = Transformer(conf, device).to(device)

    # Compile for speedup
    trans.compile(
        backend=("aot_eager" if device == torch.device("mps") else "inductor")
    )

    # Declare optimizer. Fused provides speedup
    optim = torch.optim.AdamW(trans.parameters(), lr=1e-4, fused=True)

    # Train & save
    train(
        model=trans,
        optim=optim,
        loader_tr=loader_tr,
        writer=SummaryWriter(),
        epochs=10,
    )
    torch.save(trans, "model.pt")
    # trans = torch.load("model.pt", weights_only=False)

    # Get some gibberish
    response = generate(trans, tk, "the chocolate was ", 128)
    print(response)


if __name__ == "__main__":
    main()
