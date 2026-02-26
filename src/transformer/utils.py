import kagglehub
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from spellchecker import SpellChecker
from .tokenizer import Tokenizer


def load_data(force_remake=False, save_path="data/processed_data.npy"):
    """Utility function to load and spellcheck Amazon fine food reviews via kagglehub."""
    if not force_remake and Path(save_path).is_file():
        return np.load(save_path)

    if force_remake:
        print("Force remake option detected.")
        print("Warning: This may take a while...")

    # Download dataset from kaggle
    dir_data = Path(kagglehub.dataset_download("snap/amazon-fine-food-reviews"))
    csv_data = list(dir_data.rglob("*.csv"))
    assert len(csv_data) == 1
    csv_data = csv_data[0]
    assert csv_data.is_file()

    df_full = pd.read_csv(csv_data)
    df_full = df_full[::10]

    assert "Text" in df_full.columns
    assert len(df_full["Text"]) > 0

    data_unchecked = [Tokenizer.process_text(x) for x in df_full["Text"].to_numpy()]

    # Eliminate entire entry if spelling error found (corrections take too long)
    spell = SpellChecker()
    data_proc = np.ndarray([])
    n_rows = len(data_unchecked)
    allowed_punc = '-.,;:!?()"' + "".join(str(x) for x in range(10))
    for i, review in enumerate(data_unchecked):
        print(f"{i}/{n_rows}")

        if np.all(
            [word in spell or word in allowed_punc for word in review.split(" ")]
        ):
            data_proc = np.append(data_proc, review)

    np.save(save_path, data_proc)
    return data_proc


def get_loaders(data_encoded, batch_size=2056):
    """Utility function to wrap encoded data in a torch dataloader for training."""
    tData = torch.tensor(data_encoded)
    gen = torch.Generator(device=torch.device("cpu"))
    gen.manual_seed(12345)

    loader = DataLoader(
        TensorDataset(tData),
        batch_size=batch_size,
        generator=gen,
        shuffle=True,
        drop_last=(batch_size < len(tData)),
    )

    return loader


def train(
    model,
    optim,
    loader_tr,
    writer,
    epochs=10,
):
    """A basic training loop."""
    device = next(model.parameters()).device
    n_batches = torch.tensor(len(loader_tr), device=device)
    total_loss = torch.zeros(1, device=device)

    model.train()
    for epoch in range(epochs):
        total_loss.zero_()
        for (batch,) in loader_tr:
            # xs = batch[:-1, :]
            # ys = batch[1:, :]  # offset by one
            xs = batch[:, :-1]
            ys = batch[:, 1:]  # offset by one
            # print(xs.shape)
            # print(xs)
            xs = xs.to(device, dtype=torch.long)
            # print(f"{xs.shape=}")
            # print(f"{xs=}")
            # print(f"{ys=}")
            ys = ys.to(device, dtype=torch.long)
            y_hats = model(xs)

            # print(f"{y_hats=}")
            # print(f"{ys=}")
            # print(f"{y_hats.shape=}")
            # print(f"{ys.shape=}")
            loss = torch.nn.functional.cross_entropy(y_hats.transpose(1, 2), ys)
            # print(f"{loss=}")
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.detach()
        total_loss = total_loss / n_batches
        loss_sum = total_loss.cpu().numpy()
        writer.add_scalar("Loss", loss_sum, epoch)
        print(f"E: {epoch}/{epochs}, Loss: {loss_sum}")


def generate(model, tokenizer, prompt, len_output):
    prompt_tensor = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
        device=next(model.parameters()).device,
    ).unsqueeze(0)

    response = list()
    for _ in range(len_output):
        logits = model(prompt_tensor)
        logits = logits[:, -1, :]  # get final step
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        prompt_tensor = torch.cat((prompt_tensor, idx_next), dim=1)
        response.append(int(idx_next[0, 0]))

    return prompt + tokenizer.decode(response)
