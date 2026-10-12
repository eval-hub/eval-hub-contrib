"""Prepare IFBench's NLTK resources without importing IFBench during the build."""

import importlib.util
import os
from pathlib import Path

import nltk


def main() -> None:
    data_dir = Path(os.environ["NLTK_DATA"])
    data_dir.mkdir(parents=True, exist_ok=True)
    data_dir.chmod(0o755)

    spec = importlib.util.find_spec("ifbench")
    if spec is None or spec.origin is None:
        raise RuntimeError("IFBench must be installed before preparing NLTK data")
    # mkdir(exist_ok=True) in upstream instructions.py accepts this directory
    # symlink; its NLTK search path then resolves to the preinstalled resources.
    package_data_dir = Path(spec.origin).parent / ".nltk_data"
    package_data_dir.symlink_to(data_dir, target_is_directory=True)

    for resource in (
        "punkt",
        "punkt_tab",
        "stopwords",
        "averaged_perceptron_tagger_eng",
    ):
        if not nltk.download(resource, download_dir=str(data_dir), quiet=True):
            raise RuntimeError(f"Failed to download NLTK resource: {resource}")

    # OpenShift can assign an arbitrary UID. All runtime users need read access.
    for path in data_dir.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


if __name__ == "__main__":
    main()
