"""Tests for the arxiv_get retry helper."""

from unittest import mock

import pytest
import requests

from arxiv_fetch.main import arxiv_get


def _resp(status_code: int, headers: dict | None = None) -> mock.Mock:
    r = mock.Mock(spec=requests.Response)
    r.status_code = status_code
    r.headers = headers or {}
    r.raise_for_status = mock.Mock()
    r.close = mock.Mock()
    return r


def test_retries_on_timeout_then_succeeds():
    """A transient read timeout should be retried, not propagated."""
    ok = _resp(200)
    side_effects = [requests.exceptions.Timeout("read timed out"), ok]
    with mock.patch("arxiv_fetch.main.requests.get", side_effect=side_effects) as g:
        with mock.patch("arxiv_fetch.main.time.sleep") as sleep:
            result = arxiv_get("https://example.com", max_attempts=5)
    assert result is ok
    assert g.call_count == 2
    sleep.assert_called()  # backed off between attempts


def test_timeout_exhausts_attempts_raises_clean_error():
    """After exhausting attempts on timeouts, raise a clear error (not a raw urllib3 traceback)."""
    with mock.patch(
        "arxiv_fetch.main.requests.get",
        side_effect=requests.exceptions.Timeout("read timed out"),
    ) as g:
        with mock.patch("arxiv_fetch.main.time.sleep"):
            with pytest.raises(RuntimeError):
                arxiv_get("https://example.com", max_attempts=3)
    assert g.call_count == 3


def test_retries_on_connection_error():
    """Connection errors are transient and should be retried like timeouts."""
    ok = _resp(200)
    side_effects = [requests.exceptions.ConnectionError("conn reset"), ok]
    with mock.patch("arxiv_fetch.main.requests.get", side_effect=side_effects) as g:
        with mock.patch("arxiv_fetch.main.time.sleep"):
            result = arxiv_get("https://example.com", max_attempts=5)
    assert result is ok
    assert g.call_count == 2


def test_retries_on_429_then_succeeds():
    """Existing 429 behavior must be preserved."""
    ok = _resp(200)
    side_effects = [_resp(429), ok]
    with mock.patch("arxiv_fetch.main.requests.get", side_effect=side_effects) as g:
        with mock.patch("arxiv_fetch.main.time.sleep"):
            result = arxiv_get("https://example.com", max_attempts=5)
    assert result is ok
    assert g.call_count == 2


def test_success_first_try_no_sleep():
    ok = _resp(200)
    with mock.patch("arxiv_fetch.main.requests.get", return_value=ok):
        with mock.patch("arxiv_fetch.main.time.sleep") as sleep:
            result = arxiv_get("https://example.com")
    assert result is ok
    sleep.assert_not_called()
