import kagglehub
import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from symspellpy import SymSpell, Verbosity
from .tokenizer import Tokenizer
import importlib.resources


def load_data(force_remake=False, save_path="data/processed_data.npy"):
    """Utility function to load and spellcheck Amazon fine food reviews via kagglehub."""
    if not force_remake and Path(save_path).is_file():
        return np.load(save_path)

    if force_remake:
        print("Force remake dataset option detected.")
        print("Warning: This may take a while...")

    # Download dataset from kaggle
    dir_data = Path(kagglehub.dataset_download("snap/amazon-fine-food-reviews"))
    csv_data = list(dir_data.rglob("*.csv"))
    assert len(csv_data) == 1
    csv_data = csv_data[0]
    assert csv_data.is_file()

    df_full = pd.read_csv(csv_data)
    # df_full = df_full[::4]  # Reduce dataset size

    sym_spell = SymSpell(
        max_dictionary_edit_distance=2,  # typos up to 2 letters
        prefix_length=7,
    )

    with importlib.resources.path(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    ) as dictionary_path:
        sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)

    # Cache to store words we've already checked/corrected
    corrections = {}

    def correct_spelling(text):
        if pd.isna(text):
            return text

        # Process individual words matched by regex
        def replace_word(match):
            word = match.group(0)

            # Skip numbers
            if word.isdigit():
                return word

            word = word.lower()

            if word in corrections:  # Check cache
                corrected = corrections[word]
            else:  # use spellchecker & add to cache
                suggestions = sym_spell.lookup(
                    word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True
                )
                corrected = suggestions[0].term if suggestions else word
                corrections[word] = corrected

            return corrected

        # Use Regex to find whole words, process them, and keep punctuation intact
        return re.sub(r"\b[A-Za-z]+\b", replace_word, str(text))

    # Spell check & correct
    print("Starting correction process (1/2)...")
    df_full["Text"] = df_full["Text"].apply(correct_spelling)
    print(f"Found {df_full.shape[0]} data points after first pass.")

    good_words = {}

    # Second pass spell check
    # This time we eliminate an entire entry if a single word is unknown
    def is_clean(text):
        if pd.isna(text):
            return False
        text = Tokenizer.process_text(text)
        for word in str(text).split():
            if word in good_words and not good_words[word]:  # In cache and bad
                return False
            if word not in good_words:  # Not in cache
                suggestions = sym_spell.lookup(
                    word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=False
                )
                if not suggestions:  # Unknown word
                    good_words[word] = False
                    return False
                good_words[word] = True
        return True

    # Mask of rows where all words are spelled correctly
    print("Starting correction process (2/2)...")
    mask = df_full["Text"].apply(is_clean)
    df_full = df_full[mask]

    print(f"Found {df_full.shape[0]} data points after second pass.")

    assert "Text" in df_full.columns
    assert len(df_full["Text"]) > 0

    data = [Tokenizer.process_text(x) for x in df_full["Text"].to_numpy()]
    np.save(save_path, data)
    print("Processing complete!")
    return data


def get_loaders(data_encoded, batch_size=2056, seed=12345):
    """Utility function to wrap encoded data in a torch `DataLoader` for training."""
    tData = torch.from_numpy(np.array(data_encoded))
    gen = torch.Generator(device=torch.device("cpu"))
    if seed is not None:
        gen.manual_seed(seed)

    loader = DataLoader(
        TensorDataset(tData),
        batch_size=batch_size,
        generator=gen,
        shuffle=True,
        drop_last=(batch_size < len(tData)),
    )

    return loader


def train(model, optim, loader_tr, writer, epochs=10, write_interval=10):
    """A basic training loop."""
    device = next(model.parameters()).device
    n_batches = torch.tensor(len(loader_tr), device=device)
    total_loss = torch.zeros(1, device=device)

    model.train()
    for epoch in range(epochs):
        total_loss.zero_()
        for i, (batch,) in enumerate(loader_tr):
            xs = batch[:, :-1]  # e.g. [the quick brown]
            ys = batch[:, 1:]  # e.g. [quick brown fox]
            xs = xs.to(device, dtype=torch.long)
            ys = ys.to(device, dtype=torch.long)
            y_hats = model(xs)
            loss = torch.nn.functional.cross_entropy(y_hats.transpose(1, 2), ys)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_val = loss.detach().cpu().numpy()
            if i % write_interval == 0:
                print(f"B: {i + 1}/{n_batches}, Loss: {loss_val:.3f}")
                writer.add_scalar("Loss/Batch", loss_val, i)
            total_loss += loss.detach()

        total_loss = total_loss / n_batches
        loss_sum = total_loss.cpu().numpy()
        writer.add_scalar("Loss/sum", loss_sum, epoch)
        print(f"E: {epoch + 1}/{epochs}, Loss: {loss_sum}")


def generate(model, tokenizer, prompt, len_output):
    """Generate a response from `model` beginning with `prompt`."""
    model.eval()
    with torch.no_grad():
        prompt = prompt.lower()
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
