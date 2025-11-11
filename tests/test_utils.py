import os
import sys

# ensure project root is on sys.path so `app` package is importable during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import utils


def test_safe_percent_normal():
    assert utils.safe_percent(1, 2) == 50.0


def test_safe_percent_zero():
    assert utils.safe_percent(1, 0) == 0.0


def test_get_openai_client_no_key(monkeypatch):
    # ensure OPENAI_API_KEY not set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert utils.get_openai_client() is None
