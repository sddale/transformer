import torch
from torch.utils.tensorboard import SummaryWriter
from transformer.transformer import Transformer
from transformer.utils import train, load_data, get_loaders, generate
from transformer.config import Config
from transformer.tokenizer import Tokenizer
from pathlib import Path
import typer
from typing_extensions import Annotated
from collections import OrderedDict


def main(
    interactive: Annotated[
        bool, typer.Option(help="Launch interactive session.")
    ] = False,
    out_tokens: Annotated[
        int, typer.Option(help="Length of generated model output.")
    ] = 128,
    model_path: str = "data/model.pt",
    force_train: bool = False,
    force_fresh_data: bool = False,
):
    device = (
        torch.device("mps")
        if torch.mps.is_available()
        else torch.device("cuda")  # untested
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Using device {device}.")

    # Load kaggle dataset
    print("Loading dataset...")
    data = " ".join(load_data(force_remake=force_fresh_data))
    print("Dataset initialized.")

    # Create tokenizer
    tk = Tokenizer(data)

    # Setup some model hyperparameters
    conf = Config(
        d_model=128,
        d_vocab=len(tk.vocab_dict),
        d_hidden=4 * 128,
        n_hidden=0,
        d_head=128,
        n_blocks=2,
        n_context=256,
        attn_bias=True,
    )

    # Encode and split data
    data_encoded = tk.make_batches(batch_size=conf.n_context)

    # Convert to pytorch dataloader
    loader_tr = get_loaders(data_encoded, batch_size=32)

    # Instantiate transformer network
    trans = Transformer(conf, device).to(device)
    print(f"Parameter count: {sum(p.numel() for p in trans.parameters())}")

    # Declare optimizer. Fused provides speedup
    optim = torch.optim.AdamW(trans.parameters(), lr=1e-4, fused=True)

    # Compiler backend for speedups
    backend = "aot_eager" if device == torch.device("mps") else "inductor"

    # Train & save
    if not force_train and Path(model_path).is_file():
        # Handle & load compiled state dict
        state_dict = torch.load(model_path, weights_only=False)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("_orig_mod.", "")
            new_state_dict[name] = v

        trans.load_state_dict(new_state_dict)

        trans.compile(backend=backend)

        print("Loaded trained model.\n")
    else:
        print("Beginning training...")

        trans.compile(backend=backend)
        train(
            model=trans,
            optim=optim,
            loader_tr=loader_tr,
            writer=SummaryWriter(),
            epochs=1,
        )
        print("Training complete.\n")
        torch.save(trans.state_dict(), model_path)
        print("Model saved.")

    # Get generated string
    if interactive:
        exit_phrase = "Goodbye"
        print(f'Beginning interactive session. Send "{exit_phrase}" to exit.')
        while True:
            prompt = input(
                "Give me the first few words of a sentence to get me started...\n"
            )
            if prompt == exit_phrase:
                break
            if prompt[-1] != " ":
                prompt += " "
            print(generate(trans, tk, prompt, out_tokens) + "\n")
    else:
        print(generate(trans, tk, "the chocolate was ", out_tokens))


if __name__ == "__main__":
    typer.run(main)
