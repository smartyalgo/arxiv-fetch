"""Tests for the duplicate-download detection in cmd_download."""

import sqlite3
from types import SimpleNamespace
from unittest import mock

import numpy as np

from arxiv_fetch.main import cmd_download, find_downloaded, init_db, upsert_paper


def _db_with_paper(tmp_path, paper_id: str) -> sqlite3.Connection:
    conn = init_db(tmp_path / "papers.db")
    upsert_paper(
        conn,
        paper_id,
        "A Great Paper",
        "The abstract.",
        "/downloads/A_Great_Paper.pdf",
        np.zeros(3, dtype=np.float32),
    )
    return conn


def test_find_downloaded_exact_match(tmp_path):
    conn = _db_with_paper(tmp_path, "2301.07041")
    assert find_downloaded(conn, "2301.07041")[0] == "2301.07041"
    conn.close()


def test_find_downloaded_matches_across_versions(tmp_path):
    conn = _db_with_paper(tmp_path, "2301.07041v2")
    assert find_downloaded(conn, "2301.07041")[0] == "2301.07041v2"
    assert find_downloaded(conn, "2301.07041v1")[0] == "2301.07041v2"
    conn.close()


def test_find_downloaded_no_match(tmp_path):
    conn = _db_with_paper(tmp_path, "2301.07041")
    assert find_downloaded(conn, "1999.00001") is None
    conn.close()


def test_download_skips_when_user_declines(tmp_path, monkeypatch, capsys):
    _db_with_paper(tmp_path, "2301.07041").close()
    monkeypatch.setattr("arxiv_fetch.main.DB_PATH", tmp_path / "papers.db")
    monkeypatch.setattr(
        "arxiv_fetch.main.load_config",
        lambda: {"download_dir": str(tmp_path / "dl")},
    )
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with mock.patch(
        "arxiv_fetch.main.fetch_metadata",
        side_effect=AssertionError("must not fetch on skip"),
    ):
        cmd_download(SimpleNamespace(paper="2301.07041", force=False, verbose=False))
    out = capsys.readouterr().out
    assert "Already downloaded" in out
    assert "Skipped" in out
