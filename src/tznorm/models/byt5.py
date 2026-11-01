"""M2 · ByT5-small : entraînement en deux phases et inférence (n-best, score par décodage forcé).

ByT5 travaille sur les octets UTF-8 : un caractère Tifinagh ou arabe vaut 2 à 3 octets, d'où des longueurs
maximales exprimées en octets. Ne jamais entraîner T5 en fp16 (instabilité) : bf16 ou fp32.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)

from ..eval.metrics import corpus_cer
from ..io import read_jsonl, resolve, set_seed
from .base import pick_device
from .data import oversample, record_to_pair


class PairDataset(torch.utils.data.Dataset):
    """Paires (source, cible) tokenisées à la volée."""

    def __init__(self, pairs: list[tuple[str, str]], tok, max_src: int, max_tgt: int):
        self.pairs, self.tok, self.max_src, self.max_tgt = pairs, tok, max_src, max_tgt

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        src, tgt = self.pairs[i]
        enc = self.tok(src, max_length=self.max_src, truncation=True)
        labels = self.tok(text_target=tgt, max_length=self.max_tgt, truncation=True)["input_ids"]
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


def _metrics_fn(tok):
    def compute(eval_pred):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, tok.pad_token_id, preds)
        labels = np.where(labels == -100, tok.pad_token_id, labels)
        hyps = tok.batch_decode(preds, skip_special_tokens=True)
        refs = tok.batch_decode(labels, skip_special_tokens=True)
        return {"cer": corpus_cer(hyps, [[r] for r in refs])}

    return compute


def _args(cfg: dict, p: dict, out_dir: Path, epochs: float, has_dev: bool) -> Seq2SeqTrainingArguments:
    steps = int(p.get("eval_steps", 1000))
    return Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        learning_rate=float(p["learning_rate"]),
        per_device_train_batch_size=int(p["batch_size"]),
        per_device_eval_batch_size=int(p["batch_size"]),
        gradient_accumulation_steps=int(p.get("grad_accum", 1)),
        num_train_epochs=float(epochs),
        warmup_ratio=float(p.get("warmup_ratio", 0.05)),
        weight_decay=0.01,
        eval_strategy="steps" if has_dev else "no",
        eval_steps=steps,
        save_strategy="steps",
        save_steps=steps,
        save_total_limit=2,
        load_best_model_at_end=has_dev,
        metric_for_best_model="cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=int(cfg["max_target_bytes"]),
        generation_num_beams=int(cfg.get("eval_beams", 1)),
        bf16=bool(cfg.get("bf16", False)),
        fp16=False,
        logging_steps=50,
        report_to=cfg.get("report_to", "none"),
        seed=int(cfg.get("seed", 13)),
    )


def _fit(model, tok, pairs, dev_ds, cfg: dict, p: dict, out_dir: Path, epochs: float):
    train_ds = PairDataset(pairs, tok, int(cfg["max_source_bytes"]), int(cfg["max_target_bytes"]))
    trainer = Seq2SeqTrainer(
        model=model,
        args=_args(cfg, p, out_dir, epochs, dev_ds is not None),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        processing_class=tok,
        compute_metrics=_metrics_fn(tok) if dev_ds is not None else None,
    )
    trainer.train()
    return trainer.model


def _records(paths: list[str]) -> list[dict]:
    return [r for path in paths or [] for r in read_jsonl(path)]


def train_byt5(cfg: dict, phase: str) -> Path:
    """Phase A : synthétique avec curriculum de bruit. Phase B : réel + rejeu synthétique."""
    seed = int(cfg.get("seed", 13))
    set_seed(seed)
    rng = random.Random(seed)
    out_root = resolve(cfg["output_dir"])
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tag_dropout = float(cfg.get("tag_dropout", 0.0))
    p = cfg[f"phase{phase}"]

    dev_ds = None
    if cfg.get("dev") and resolve(cfg["dev"]).exists():
        dev_pairs = [record_to_pair(r) for r in read_jsonl(cfg["dev"])]
        if p.get("max_dev"):
            dev_pairs = dev_pairs[: int(p["max_dev"])]
        dev_ds = PairDataset(dev_pairs, tok, int(cfg["max_source_bytes"]), int(cfg["max_target_bytes"]))

    if phase == "A":
        model = T5ForConditionalGeneration.from_pretrained(cfg["model_name"])
        synth = _records(p["train"])
        stages = p.get("curriculum") or [[0, 1, 2, 3]]
        epochs = p.get("epochs_per_stage") or [1] * len(stages)
        for k, (levels, ep) in enumerate(zip(stages, epochs), 1):
            allowed = {int(x) for x in levels}
            recs = [r for r in synth if int(r.get("niveau_bruit", 0)) in allowed]
            rng.shuffle(recs)
            pairs = [record_to_pair(r, rng, tag_dropout) for r in recs]
            model = _fit(model, tok, pairs, dev_ds, cfg, p, out_root / "phaseA" / f"etape{k}", ep)
        final = out_root / "phaseA" / "final"
    elif phase == "B":
        model = T5ForConditionalGeneration.from_pretrained(str(resolve(p["init_from"])))
        real = oversample(_records(p["train_real"]), float(p.get("oversample_min_fraction", 0.0)), rng)
        ratio = float(p.get("replay_ratio", 0.0))
        replay = []
        if 0 < ratio < 1 and p.get("replay_synthetic"):
            pool = _records(p["replay_synthetic"])
            n = int(len(real) * ratio / (1 - ratio))
            replay = rng.sample(pool, min(n, len(pool)))
        recs = real + replay
        rng.shuffle(recs)
        pairs = [record_to_pair(r, rng, tag_dropout) for r in recs]
        model = _fit(model, tok, pairs, dev_ds, cfg, p, out_root / "phaseB", float(p["epochs"]))
        final = out_root / "phaseB" / "final"
    else:
        raise ValueError("phase doit valoir A ou B")

    model.save_pretrained(final)
    tok.save_pretrained(final)
    return final


class ByT5Normalizer:
    def __init__(self, model, tok, device: str, max_length: int = 512, batch_size: int = 16):
        self.model, self.tok, self.device = model, tok, device
        self.max_length, self.batch_size = max_length, batch_size

    @classmethod
    def load(cls, path: str | Path, device: str = "auto", max_length: int = 512, batch_size: int = 16):
        dev = pick_device(device)
        tok = AutoTokenizer.from_pretrained(str(path))
        model = T5ForConditionalGeneration.from_pretrained(str(path)).to(dev).eval()
        return cls(model, tok, dev, max_length, batch_size)

    def _encode(self, texts: list[str]):
        return self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=self.max_length).to(self.device)

    @torch.no_grad()
    def generate(self, sources: list[str], num_beams: int = 5, num_return: int = 5) -> list[list[tuple[str, float]]]:
        nr = max(1, min(num_return, num_beams))
        results: list[list[tuple[str, float]]] = []
        for i in range(0, len(sources), self.batch_size):
            batch = sources[i:i + self.batch_size]
            out = self.model.generate(**self._encode(batch), num_beams=num_beams, num_return_sequences=nr,
                                      max_new_tokens=self.max_length, output_scores=True,
                                      return_dict_in_generate=True, early_stopping=num_beams > 1)
            texts = self.tok.batch_decode(out.sequences, skip_special_tokens=True)
            if num_beams > 1 and out.sequences_scores is not None:
                scores = out.sequences_scores.tolist()
            else:
                scores = self.score([s for s in batch for _ in range(nr)], texts)
            for b in range(len(batch)):
                results.append(list(zip(texts[b * nr:(b + 1) * nr], scores[b * nr:(b + 1) * nr])))
        return results

    @torch.no_grad()
    def score(self, sources: list[str], targets: list[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(sources), self.batch_size):
            enc = self._encode(sources[i:i + self.batch_size])
            labels = self.tok(text_target=targets[i:i + self.batch_size], return_tensors="pt", padding=True,
                              truncation=True, max_length=self.max_length)["input_ids"].to(self.device)
            labels[labels == self.tok.pad_token_id] = -100
            logits = self.model(**enc, labels=labels).logits
            logp = torch.log_softmax(logits.float(), dim=-1)
            mask = labels != -100
            tok_lp = logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
            out.extend((tok_lp.sum(-1) / mask.sum(-1).clamp(min=1)).tolist())
        return out
