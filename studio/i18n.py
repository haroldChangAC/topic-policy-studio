"""Translate presentation strings while keeping widget and project values stable."""

import json
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

_option_language = ContextVar('studio_option_language', default=None)


@lru_cache(maxsize=1)
def _english_messages():
    return json.loads(Path(__file__).with_name('ui_messages_en.json').read_text(encoding='utf-8'))


def is_english():
    captured = _option_language.get()
    if captured is not None:
        return captured
    ctx = get_script_run_ctx(suppress_warning=True)
    return ctx is not None and st.session_state.get('studio_language') == 'English'


def tr(source, *values):
    """Format an interface message, leaving technical values and user text intact."""
    if not isinstance(source, str):
        return str(source)
    message = _english_messages().get(source, source) if is_english() else source
    return message.format(*values) if values else message


def diagnostic(error):
    """Preserve original details when a backend diagnostic has no translation."""
    original = str(error)
    translated = tr(original)
    if is_english() and translated == original and any('\u3400' <= c <= '\u9fff' for c in original):
        return f'Original diagnostic (Chinese): {original}'
    return translated


def localize_options(format_option=tr):
    """Keep the widget formatter stable when Streamlit calls it outside the run."""
    english = is_english()

    def label(value):
        token = _option_language.set(english)
        try:
            return format_option(value)
        finally:
            _option_language.reset(token)

    return label
