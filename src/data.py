"""GLUE task loading and tokenization."""

from __future__ import annotations

from datasets import load_dataset

# (sentence1_key, sentence2_key, num_labels). None means single-sentence task.
GLUE_TASKS = {
    "sst2":  ("sentence", None, 2),
    "mrpc":  ("sentence1", "sentence2", 2),
    "rte":   ("sentence1", "sentence2", 2),
    "qnli":  ("question", "sentence", 2),
    "qqp":   ("question1", "question2", 2),
    "mnli":  ("premise", "hypothesis", 3),
}


def load_glue(task: str, tokenizer, max_len: int = 128, subset: int | None = None):
    """Return (train_ds, eval_ds, num_labels) tokenized for `task`.

    `subset` caps the number of train/eval examples (used by the smoke config so
    the CPU run finishes in seconds). MNLI uses validation_matched for eval.
    """
    if task not in GLUE_TASKS:
        raise ValueError(f"unknown task {task!r}; options: {list(GLUE_TASKS)}")
    s1, s2, num_labels = GLUE_TASKS[task]
    # datasets>=4 dropped script-based repos; canonical GLUE lives at nyu-mll/glue.
    ds = load_dataset("nyu-mll/glue", task)

    def tok(batch):
        args = (batch[s1],) if s2 is None else (batch[s1], batch[s2])
        return tokenizer(*args, truncation=True, max_length=max_len)

    ds = ds.map(tok, batched=True)
    eval_split = "validation_matched" if task == "mnli" else "validation"
    train, ev = ds["train"], ds[eval_split]
    if subset is not None:
        train = train.select(range(min(subset, len(train))))
        ev = ev.select(range(min(subset, len(ev))))
    return train, ev, num_labels
