"""Offline-mode helpers shared by the RULER data generator and tokenizer."""

import os
from typing import Any

OFFLINE_ENV = "RULER_OFFLINE"


def is_offline() -> bool:
    """Whether this RULER process must avoid all runtime downloads."""
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def ensure_nltk_resources(nltk_module: Any, offline: bool | None = None) -> None:
    """Verify sentence-tokenizer data, downloading only for connected jobs."""
    if offline is None:
        offline = is_offline()

    for package, resource in (
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ):
        try:
            nltk_module.data.find(resource)
        except LookupError as exc:
            if offline:
                raise RuntimeError(
                    f"RULER is running offline but NLTK resource '{package}' is missing. "
                    "Use the RULER adapter image with punkt and punkt_tab preinstalled."
                ) from exc
            if not nltk_module.download(package, raise_on_error=True):
                raise RuntimeError(f"Failed to download NLTK resource '{package}'") from exc
