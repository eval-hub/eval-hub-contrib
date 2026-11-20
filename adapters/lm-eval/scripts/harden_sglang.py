"""Backport an opt-in remote-code default to the pinned lm_eval dependency."""

from importlib.metadata import distribution
from pathlib import Path

_UNSAFE_DEFAULT = "        trust_remote_code: bool = True,"
_SAFE_DEFAULT = "        trust_remote_code: bool = False,"


def harden_sglang_default(path: Path) -> None:
    """Change the known constructor default, failing on unexpected source."""
    source = path.read_text(encoding="utf-8")
    if source.count(_SAFE_DEFAULT) == 1 and _UNSAFE_DEFAULT not in source:
        return
    if source.count(_UNSAFE_DEFAULT) != 1 or _SAFE_DEFAULT in source:
        raise RuntimeError(
            "Unexpected SGLang trust_remote_code default; review the pinned lm_eval backport"
        )
    path.write_text(source.replace(_UNSAFE_DEFAULT, _SAFE_DEFAULT, 1), encoding="utf-8")


if __name__ == "__main__":
    backend = distribution("lm_eval").locate_file("lm_eval/models/sglang_causallms.py")
    harden_sglang_default(Path(backend))
