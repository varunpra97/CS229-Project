"""Fine-tuning loop: RoBERTa-base + LoRA/DoRA via peft, on one GLUE task."""

from __future__ import annotations

import numpy as np
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

from .data import load_glue

# RoBERTa attention + intermediate projection module names targeted by the adapter.
TARGET_MODULES = ["query", "key", "value", "output.dense", "intermediate.dense"]


def build_model(model_name: str, num_labels: int, method: str, rank: int,
                lora_alpha: int = 16, dropout: float = 0.05):
    """Load base model and wrap with a LoRA or DoRA adapter.

    DoRA is LoRA with use_dora=True (peft >= 0.11). LoRA is the same config with
    the flag off, so the two are matched for trainable parameter count.
    """
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    use_dora = method.lower() == "dora"
    if method.lower() not in ("lora", "dora"):
        raise ValueError(f"method must be 'lora' or 'dora', got {method!r}")
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        target_modules=TARGET_MODULES,
        use_dora=use_dora,
    )
    return get_peft_model(base, cfg)


def make_metric(task: str):
    acc = evaluate.load("accuracy")

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return acc.compute(predictions=preds, references=labels)

    return compute


def train_one(cfg: dict):
    """Run one fine-tuning job. Returns (peft_model, tokenizer, eval_metrics)."""
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_ds, eval_ds, num_labels = load_glue(
        cfg["task"], tok, max_len=cfg.get("max_len", 128),
        subset=cfg.get("subset"),
    )
    model = build_model(
        cfg["model_name"], num_labels, cfg["method"], cfg["rank"],
        lora_alpha=cfg.get("lora_alpha", 16), dropout=cfg.get("dropout", 0.05),
    )

    args = TrainingArguments(
        output_dir=cfg.get("output_dir", "out"),
        per_device_train_batch_size=cfg.get("batch_size", 32),
        per_device_eval_batch_size=cfg.get("batch_size", 32),
        learning_rate=cfg.get("lr", 2e-4),
        num_train_epochs=cfg.get("epochs", 3),
        max_steps=cfg.get("max_steps", -1),     # smoke config sets this small
        lr_scheduler_type="cosine",
        seed=cfg.get("seed", 0),
        logging_steps=10,
        report_to="none",
        save_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=make_metric(cfg["task"]),
    )
    trainer.train()
    metrics = trainer.evaluate()
    return model, tok, metrics
