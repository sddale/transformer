import unicodedata
from collections import Counter
import numpy as np


class Tokenizer:
    """A collection of functions to prepare raw data for input to a transformer model."""

    def __init__(self, data):
        data_tokenized = self.tokenize(data)
        vocab_freq = self.analyze_vocab(data_tokenized)
        self.vocab_arr = [word for word, _ in vocab_freq.most_common()]
        self.vocab_dict = {word: i for i, word in enumerate(self.vocab_arr)}
        self.data_encoded = self.encode(data_tokenized)

    @staticmethod
    def process_text(
        text: str,
        allowed_punctuation: str = '-.,;:!?()"' + "".join(str(x) for x in range(10)),
        punctuation_convert: dict[str, str] = {
            "—": "-",
            "can't": "can not",
            "cant": "can not",
            "cannot": "can not",
            "n't": " not",
            "'ve": " have",
        },
        split_punc=True,
    ) -> str:
        """A suite of text processing calls adapted from github.com/mines-opt-ml/decoding-gpt."""
        # replace some special characters which unicode won't normalize properly
        for char, replacement in punctuation_convert.items():
            text = text.replace(char, replacement)

        # Normalize the string to decompose Unicode characters
        text = unicodedata.normalize("NFKD", text)

        # Encode to ASCII bytes, then decode back to string, ignoring errors
        text = text.encode("ascii", "ignore").decode("ascii")

        # remove newlines and tabs
        text = text.replace("\n", " ").replace("\t", " ").replace("br", "")

        # put spaces around allowed punctuation
        if split_punc:
            for char in allowed_punctuation:
                text = text.replace(char, f" {char} ")

        # remove leading and trailing spaces
        text = text.strip()

        # remove multiple spaces
        while "  " in text:
            text = text.replace("  ", " ")

        # remove all characters except (alphanumeric, allowed_punctuation, ' ')
        text = "".join(
            (
                char
                if (char.isalnum() or char in allowed_punctuation or char == " ")
                else ""
            )
            for char in text
        )

        text = text.lower()
        text = text.strip()

        return text

    def tokenize(
        self,
        text: str,
        process: bool = False,
    ) -> list[str]:
        """Tokenizes data."""
        if process:
            text = self.process_text(text)
        return text.split(" ")

    def make_batches(self, batch_size):
        # trim the last tokens to make sure the length is a multiple of n_context
        data = self.data_encoded[: -(len(self.data_encoded) % batch_size)]

        # split the text into examples of length n_context
        data: list[list[int]] = [
            data[i : i + batch_size] for i in range(0, len(data), batch_size)
        ]
        return data

    @staticmethod
    def analyze_vocab(tokenized_text: list[str]) -> Counter[str]:
        """Provides a Counter wrapper to tokenized data."""
        return Counter(tokenized_text)

    def encode(self, text: str | list[str]):
        """Encodes data using a pre-constructed vocab dictionary."""
        if isinstance(text, str):
            text = self.tokenize(text)
        return np.array(
            [self.vocab_dict[word] for word in text if word in self.vocab_dict]
        )

    def decode(self, encoded_text: list[int]) -> str:
        """Decodes data using a pre-constructed vocab list."""
        space_separated = " ".join(self.vocab_arr[i] for i in encoded_text)
        for punc in '-.,;:!?)"':
            space_separated = space_separated.replace(" " + punc, punc)
        for punc in "-(":
            space_separated = space_separated.replace(punc + " ", punc)
        return space_separated
