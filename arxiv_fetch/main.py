# PYTHON_ARGCOMPLETE_OK
import argparse
import re
import sqlite3
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import argcomplete
import numpy as np
import requests

CONFIG_PATH = Path("~/.config/arxiv-fetch/config.toml").expanduser()
DB_PATH = Path("~/.config/arxiv-fetch/papers.db").expanduser()
DEFAULT_DOWNLOAD_DIR = "~/Downloads"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HF_CACHE = Path("~/.cache/huggingface/hub").expanduser()
KNOWN_CONFIG_KEYS = {"download_dir", "embedding_model"}
PAPER_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")


def normalize_arxiv_url(input_str: str) -> str:
    """Convert arxiv /html/ URLs to their /abs/ equivalent paper ID string."""
    try:
        parsed = urlparse(input_str)
        if parsed.netloc in ("arxiv.org", "www.arxiv.org") and parsed.path.startswith("/html/"):
            return parsed.path[len("/html/"):]
    except Exception:
        pass
    return input_str


_model = None


def get_model(model_name: str):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
    return _model


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id  TEXT PRIMARY KEY,
            title     TEXT,
            abstract  TEXT,
            file_path TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    return conn


def upsert_paper(conn: sqlite3.Connection, paper_id: str, title: str, abstract: str, file_path: str, embedding: np.ndarray):
    conn.execute(
        """
        INSERT INTO papers (paper_id, title, abstract, file_path, embedding)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            title=excluded.title,
            abstract=excluded.abstract,
            file_path=excluded.file_path,
            embedding=excluded.embedding
        """,
        (paper_id, title, abstract, file_path, embedding.astype(np.float32).tobytes()),
    )
    conn.commit()


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(f'download_dir = "{DEFAULT_DOWNLOAD_DIR}"\n')
    print(f"Created config at {CONFIG_PATH}")
    return {"download_dir": DEFAULT_DOWNLOAD_DIR}


def save_config(config: dict):
    lines = [f'{k} = "{v}"\n' for k, v in config.items()]
    CONFIG_PATH.write_text("".join(lines))


ARXIV_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}


def fetch_metadata(paper_id: str) -> tuple[str | None, str | None]:
    base_id = paper_id.split("v")[0]
    url = f"https://export.arxiv.org/api/query?id_list={base_id}"
    resp = requests.get(url, headers=ARXIV_HEADERS, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        return None, None
    title_el = entry.find("atom:title", ns)
    title = " ".join(title_el.text.split()) if title_el is not None and title_el.text else None
    summary_el = entry.find("atom:summary", ns)
    abstract = " ".join(summary_el.text.split()) if summary_el is not None and summary_el.text else None
    return title, abstract


def title_to_filename(title: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|]', "_", title)
    return safe.strip().replace(" ", "_")


def extract_paper_id(input_str: str) -> str:
    normalized = normalize_arxiv_url(input_str)
    match = PAPER_ID_RE.search(normalized)
    if not match:
        print(f"Error: could not extract a paper ID from '{input_str}'", file=sys.stderr)
        sys.exit(1)
    return match.group(1)


def cmd_models_list(args):
    config = load_config()
    active = config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)

    resp = requests.get(
        "https://huggingface.co/api/models",
        params={"library": "sentence-transformers", "sort": "downloads", "direction": -1, "limit": 30},
        timeout=15,
    )
    resp.raise_for_status()
    models = resp.json()

    print(f"{'':2} {'Model ID':<45} {'Downloads':>12}  Cached")
    print("-" * 70)
    for m in models:
        model_id = m["id"]
        short = model_id.split("/")[-1]
        downloads = m.get("downloads", 0)
        dl_str = f"{downloads / 1_000_000:.1f}M" if downloads >= 1_000_000 else f"{downloads // 1000}K"
        cached = (HF_CACHE / f"models--{model_id.replace('/', '--')}").exists()
        is_active = short == active or model_id == active
        marker = "* " if is_active else "  "
        cached_str = "yes" if cached else ""
        print(f"{marker}{model_id:<45} {dl_str:>12}  {cached_str}")

    print(f"\n* = active model  |  set with: arxiv-fetch config set embedding_model <short-name>")


def cmd_config(args):
    config = load_config()
    if args.config_command == "get":
        key = args.key
        if key not in KNOWN_CONFIG_KEYS:
            print(f"Error: unknown config key '{key}'. Known keys: {', '.join(sorted(KNOWN_CONFIG_KEYS))}", file=sys.stderr)
            sys.exit(1)
        defaults = {"download_dir": DEFAULT_DOWNLOAD_DIR, "embedding_model": DEFAULT_EMBEDDING_MODEL}
        print(config.get(key, defaults[key]))
    elif args.config_command == "set":
        key, value = args.key, args.value
        if key not in KNOWN_CONFIG_KEYS:
            print(f"Error: unknown config key '{key}'. Known keys: {', '.join(sorted(KNOWN_CONFIG_KEYS))}", file=sys.stderr)
            sys.exit(1)
        config[key] = value
        save_config(config)
        print(f'Updated {key} = "{value}"')
        if key == "embedding_model":
            print("Warning: re-download papers to rebuild embeddings with the new model.")


def cmd_download(args):
    config = load_config()
    download_dir = Path(config.get("download_dir", DEFAULT_DOWNLOAD_DIR)).expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)
    model_name = config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)

    paper_id = extract_paper_id(args.paper)
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    title, abstract = fetch_metadata(paper_id)
    filename = f"{title_to_filename(title)}.pdf" if title else f"{paper_id}.pdf"
    dest = download_dir / filename

    print(f"Downloading {paper_id}{f': {title}' if title else ''} ...")
    response = requests.get(pdf_url, headers=ARXIV_HEADERS, stream=True, timeout=60)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved to {dest}")

    if abstract:
        print("Indexing paper...")
        model = get_model(model_name)
        embedding = model.encode(abstract)
        conn = init_db(DB_PATH)
        upsert_paper(conn, paper_id, title or "", abstract, str(dest), embedding)
        conn.close()
        print("Indexed.")
    else:
        print("Warning: no abstract found; skipping index.", file=sys.stderr)


def cmd_completions_install(args):
    line = 'eval "$(register-python-argcomplete arxiv-fetch)"'
    dest = Path("~/.oh-my-zsh/custom/arxiv-fetch-autocomplete.zsh").expanduser()
    if dest.exists() and line in dest.read_text():
        print(f"Completion already installed at {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f"# arxiv-fetch tab completion\n{line}\n")
    print(f"Written to {dest}")
    print("Open a new terminal or run 'source ~/.zshrc' to activate.")


def cmd_similar(args):
    paper_id = extract_paper_id(args.paper)
    conn = init_db(DB_PATH)
    rows = conn.execute("SELECT paper_id, title, file_path, embedding FROM papers").fetchall()
    conn.close()

    target = next((r for r in rows if r[0] == paper_id), None)
    if target is None:
        print(f"Error: paper '{paper_id}' not found in index. Download it first.", file=sys.stderr)
        sys.exit(1)

    target_emb = np.frombuffer(target[3], dtype=np.float32).copy()
    target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-10)

    scored = []
    for row_id, title, file_path, emb_bytes in rows:
        if row_id == paper_id:
            continue
        emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
        score = float(np.dot(target_norm, emb_norm))
        scored.append((score, row_id, title, file_path))

    if not scored:
        print("No other papers in index to compare against.")
        return

    scored.sort(reverse=True)
    top_n = args.top

    print(f"Papers similar to {paper_id} ({target[1] or paper_id}):\n")
    for rank, (score, row_id, title, file_path) in enumerate(scored[:top_n], 1):
        print(f"{rank}. [{score:.3f}] {title or row_id}")
        print(f"   ID: {row_id}  |  {file_path}")
        print()


def cmd_search(args):
    config = load_config()
    model_name = config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)

    conn = init_db(DB_PATH)
    rows = conn.execute("SELECT paper_id, title, file_path, embedding FROM papers").fetchall()
    conn.close()

    if not rows:
        print("No papers indexed yet. Run 'arxiv-fetch download <url-or-id>' first.")
        return

    model = get_model(model_name)
    query_vec = model.encode(args.query).astype(np.float32)
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)

    scored = []
    for paper_id, title, file_path, emb_bytes in rows:
        emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
        score = float(np.dot(query_norm, emb_norm))
        scored.append((score, paper_id, title, file_path))

    scored.sort(reverse=True)
    top_n = args.top if hasattr(args, "top") else 5

    print(f"Top {min(top_n, len(scored))} results for: \"{args.query}\"\n")
    for rank, (score, paper_id, title, file_path) in enumerate(scored[:top_n], 1):
        print(f"{rank}. [{score:.3f}] {title or paper_id}")
        print(f"   ID: {paper_id}  |  {file_path}")
        print()


def main():
    parser = argparse.ArgumentParser(
        prog="arxiv-fetch",
        description="Download and semantically search arxiv papers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dl_parser = subparsers.add_parser("download", help="Download a paper by URL or ID")
    dl_parser.add_argument("paper", help="arxiv URL or paper ID (e.g. 2301.07041)")

    search_parser = subparsers.add_parser("search", help="Semantic search over downloaded papers")
    search_parser.add_argument("query", help="Natural language search query")
    search_parser.add_argument("--top", type=int, default=5, help="Number of results to show (default: 5)")

    models_parser = subparsers.add_parser("models", help="Browse available embedding models")
    models_sub = models_parser.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list", help="List sentence-transformer models on HuggingFace")

    cfg_parser = subparsers.add_parser("config", help="Get or set configuration values")
    cfg_sub = cfg_parser.add_subparsers(dest="config_command", required=True)
    get_p = cfg_sub.add_parser("get", help="Print a config value")
    get_p.add_argument("key")
    set_p = cfg_sub.add_parser("set", help="Set a config value")
    set_p.add_argument("key")
    set_p.add_argument("value")

    similar_parser = subparsers.add_parser("similar", help="List papers semantically similar to a given paper")
    similar_parser.add_argument("paper", help="arxiv URL or paper ID")
    similar_parser.add_argument("--top", type=int, default=5, help="Number of results to show (default: 5)")

    comp_parser = subparsers.add_parser("completions", help="Shell completion helpers")
    comp_sub = comp_parser.add_subparsers(dest="completions_command", required=True)
    comp_sub.add_parser("install", help="Print instructions to enable tab completion")

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "models":
        if args.models_command == "list":
            cmd_models_list(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "similar":
        cmd_similar(args)
    elif args.command == "completions":
        if args.completions_command == "install":
            cmd_completions_install(args)


if __name__ == "__main__":
    main()
