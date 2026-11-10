"""DocVQA task wrapper that applies EvalHub's sample cap before materialization.

The upstream inspect-evals task converts the full image-heavy validation split
to Inspect Samples before the CLI applies ``--limit``. For capped jobs, select
from the cached Hugging Face Arrow dataset first so only the requested records
and images are materialized in memory.
"""

from datasets import Image
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_evals.utils.huggingface import load_dataset
from inspect_evals.docvqa.docvqa import (
    DATASET_PATH,
    DOCVQA_DATASET_REVISION,
    EVAL_VERSION,
    docvqa_scorer,
    docvqa_solver,
    record_to_sample,
)


def build_docvqa_task(
    num_examples: int | None = None,
    shuffle: bool | int = True,
) -> Task:
    """Build the pinned DocVQA validation task, selecting capped rows early."""
    if isinstance(shuffle, bool):
        shuffle_enabled = shuffle
        shuffle_seed = None
    elif isinstance(shuffle, int):
        if shuffle < 0:
            raise ValueError("shuffle seed must be non-negative")
        shuffle_enabled = True
        shuffle_seed = shuffle
    else:
        raise TypeError(f"shuffle must be a bool or integer seed, got {type(shuffle).__name__}")

    if num_examples is not None and num_examples < 1:
        raise ValueError("num_examples must be a positive integer")

    dataset = load_dataset(
        path=DATASET_PATH,
        name="DocVQA",
        split="validation",
        revision=DOCVQA_DATASET_REVISION,
    )
    # inspect-evals' record_to_sample expects {path, bytes}, but Datasets' default
    # Image feature decodes each value to PIL.Image before the record converter sees it.
    dataset = dataset.cast_column("image", Image(decode=False))

    if shuffle_enabled:
        dataset = dataset.shuffle(seed=shuffle_seed)

    if num_examples is not None:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    samples = [record_to_sample(record) for record in dataset]
    return Task(
        dataset=MemoryDataset(
            samples=samples,
            name="DocVQA",
            location=DATASET_PATH,
            shuffled=shuffle_enabled,
        ),
        solver=[docvqa_solver()],
        scorer=docvqa_scorer(),
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata(),
    )


@task
def docvqa(num_examples: int | None = None, shuffle: bool | int = True) -> Task:
    """Run the pinned DocVQA validation split with optional early sample limiting."""
    return build_docvqa_task(num_examples=num_examples, shuffle=shuffle)
