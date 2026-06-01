"""Tests for fetch_metadata's parsing and graceful-degradation behavior."""

from unittest import mock

import requests

from arxiv_fetch.main import fetch_metadata

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>A   Great   Paper</title>
    <summary>This is
    the abstract.</summary>
  </entry>
</feed>"""


def _resp(text: str) -> mock.Mock:
    r = mock.Mock(spec=requests.Response)
    r.status_code = 200
    r.text = text
    r.headers = {}
    r.raise_for_status = mock.Mock()
    r.close = mock.Mock()
    return r


def test_parses_title_and_abstract():
    with mock.patch("arxiv_fetch.main.arxiv_get", return_value=_resp(ATOM)):
        title, abstract = fetch_metadata("2604.15039v1")
    assert title == "A Great Paper"
    assert abstract == "This is the abstract."


def test_degrades_gracefully_when_throttled():
    """If the metadata API is throttled (arxiv_get raises), return (None, None)
    so the caller can still download the PDF instead of crashing."""
    with mock.patch(
        "arxiv_fetch.main.arxiv_get",
        side_effect=RuntimeError("arxiv request failed after 5 attempts"),
    ):
        title, abstract = fetch_metadata("2604.15039v1")
    assert title is None
    assert abstract is None


def test_degrades_gracefully_on_network_error():
    with mock.patch(
        "arxiv_fetch.main.arxiv_get",
        side_effect=requests.exceptions.Timeout("read timed out"),
    ):
        title, abstract = fetch_metadata("2604.15039v1")
    assert title is None
    assert abstract is None


def test_degrades_gracefully_on_malformed_xml():
    with mock.patch("arxiv_fetch.main.arxiv_get", return_value=_resp("<not-xml")):
        title, abstract = fetch_metadata("2604.15039v1")
    assert title is None
    assert abstract is None
