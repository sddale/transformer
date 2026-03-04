# transformer
Toy transformers implemented with vanilla pytorch.

## Getting started
This repo uses `uv` for dependency management and installation. See `uv` [install directions](https://docs.astral.sh/uv/getting-started/installation/) for your operating system if needed.

Install and setup Python:
```
uv python install
```

Install dependencies:
```
uv sync
```

Execute training:
```
uv run main.py
```

This script uses `typer` for command line arguments. Run
```
uv run main.py --help
```
for supported options.

To launch an live session and "interact" with the model, run:
```
uv run main.py --interactive
```