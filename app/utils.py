import os
from typing import Optional

from openai import OpenAI
import streamlit as st


def safe_percent(numerator: int, denominator: int) -> float:
    """Return percentage (0-100) safely avoiding division by zero."""
    try:
        return (numerator / denominator) * 100 if denominator and denominator > 0 else 0.0
    except Exception:
        return 0.0


def init_session_state() -> None:
    """Initialize commonly used session state keys with sane defaults."""
    defaults = {
        "analyze_clicked": False,
        "analysis_in_progress": False,
        "analysis_done": False,
        # place to hold data/results
        "df_chess_game": None,
        "df_source": None,
        "analysis_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_openai_client() -> Optional[OpenAI]:
    """Return an OpenAI client or None if API key not configured.

    Callers should check for None.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception:
        return None


def generate_advice_from_context(context: str, model: str = "gpt-4o-mini") -> Optional[str]:
    """Generate advice text from context using OpenAI; returns text or None on failure."""
    client = get_openai_client()
    if client is None:
        return None
    prompt = f"""
You are a chess coach. Based only on the data below (no external knowledge), give the user 3-5 actionable
suggestions to improve their chess. Focus on weaknesses and opportunities.

DATA:
{context}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful chess coach analyzing player statistics."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception:
        return None
