# arxiv-fetch

A CLI tool for downloading arxiv papers by URL or ID, and semantically searching your local collection.

## What it does

- **Download** papers as PDFs and index them with sentence-transformer embeddings
- **Search** your local collection using natural language queries (semantic similarity)
- **Find similar** papers to one you already have
- **Manage** the embedding model and download directory via config
- **Tab completion** for all commands and flags

## Installation

Requires Python 3.11+.

```bash
git clone <repo>
cd arxiv_fetch
pip install -e .
```

### Tab completion (optional)

```bash
arxiv-fetch completions install
# then open a new terminal, or:
source ~/.zshrc
```

## Commands

### `download`

Download a paper by arxiv URL or paper ID. The PDF is saved to your configured download directory and indexed for search.

```bash
arxiv-fetch download 2301.07041
arxiv-fetch download https://arxiv.org/abs/2301.07041
```

### `search`

Semantic search over your downloaded papers using a natural language query.

```bash
arxiv-fetch search "attention mechanisms in transformers"
arxiv-fetch search "reinforcement learning from human feedback" --top 10
```

Options:
- `--top N` — number of results to show (default: 5)

### `similar`

Find papers in your local collection that are semantically similar to a given paper (must already be downloaded).

```bash
arxiv-fetch similar 2301.07041
arxiv-fetch similar 2301.07041 --top 10
```

Options:
- `--top N` — number of results to show (default: 5)

### `config`

Get or set configuration values.

```bash
arxiv-fetch config get download_dir
arxiv-fetch config set download_dir ~/Papers
arxiv-fetch config set embedding_model all-mpnet-base-v2
```

Config keys:
- `download_dir` — where PDFs are saved (default: `~/Downloads`)
- `embedding_model` — sentence-transformer model used for indexing (default: `all-MiniLM-L6-v2`)

> Note: changing `embedding_model` requires re-downloading papers to rebuild embeddings.

### `models`

Browse available sentence-transformer models on HuggingFace, with download counts and local cache status.

```bash
arxiv-fetch models list
```

### `completions`

```bash
arxiv-fetch completions install   # write zsh completion script to ~/.oh-my-zsh/custom/
```

## Config & data locations

| Path | Purpose |
|---|---|
| `~/.config/arxiv-fetch/config.toml` | Configuration file |
| `~/.config/arxiv-fetch/papers.db` | SQLite index of downloaded papers |
| `~/Downloads` | Default PDF download directory |
