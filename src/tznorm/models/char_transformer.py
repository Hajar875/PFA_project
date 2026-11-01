"""M1 · Transformer caractère entraîné de zéro (modèle de contrôle, léger)."""

from __future__ import annotations

import math
import random
import re
from pathlib import Path

import torch
from torch import nn

from ..eval.metrics import corpus_cer
from ..io import read_jsonl, resolve, set_seed
from .base import pick_device
from .data import record_to_pair

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
TAGS = re.compile(r"^(<v:[^>]*>) (<s:[^>]*>) ")


class CharVocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {s: i for i, s in enumerate(itos)}

    @staticmethod
    def split(text: str) -> list[str]:
        m = TAGS.match(text)
        if m:
            return [m.group(1), m.group(2)] + list(text[m.end():])
        return list(text)

    @classmethod
    def build(cls, texts) -> "CharVocab":
        symbols = set()
        for t in texts:
            symbols.update(cls.split(t))
        return cls(SPECIALS + sorted(symbols - set(SPECIALS)))

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int, bos: bool = False) -> list[int]:
        ids = [self.stoi.get(s, UNK) for s in self.split(text)][: max_len - 1 - bos]
        return ([BOS] if bos else []) + ids + [EOS]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i > UNK)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class CharSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4, dim_feedforward: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout, batch_first=True)
        self.out = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        mask = src == PAD
        h = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(h, src_key_padding_mask=mask), mask

    def decode(self, tgt_in, memory, src_mask):
        t = tgt_in.size(1)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=tgt_in.device), diagonal=1)
        h = self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
        h = self.transformer.decoder(h, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_in == PAD,
                                     memory_key_padding_mask=src_mask)
        return self.out(h)

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)


def _pad(seqs: list[list[int]], device) -> torch.Tensor:
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (n - len(s)) for s in seqs], dtype=torch.long, device=device)


class CharTransformerNormalizer:
    def __init__(self, model: CharSeq2Seq, vocab: CharVocab, device: str, max_len: int = 256):
        self.model, self.vocab, self.device, self.max_len = model, vocab, device, max_len

    @classmethod
    def load(cls, path: str | Path, device: str = "auto", **_):
        dev = pick_device(device)
        ckpt = torch.load(str(path), map_location=dev)
        vocab = CharVocab(ckpt["itos"])
        model = CharSeq2Seq(len(vocab), **ckpt["model_cfg"]).to(dev)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return cls(model, vocab, dev, ckpt.get("max_len", 256))

    def save(self, path: str | Path, model_cfg: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "itos": self.vocab.itos, "model_cfg": model_cfg,
                    "max_len": self.max_len}, str(path))

    @torch.no_grad()
    def greedy(self, source: str) -> str:
        src = _pad([self.vocab.encode(source, self.max_len)], self.device)
        memory, mask = self.model.encode(src)
        seq = [BOS]
        for _ in range(self.max_len):
            logits = self.model.decode(torch.tensor([seq], device=self.device), memory, mask)[0, -1]
            nxt = int(logits.argmax())
            if nxt == EOS:
                break
            seq.append(nxt)
        return self.vocab.decode(seq[1:])

    @torch.no_grad()
    def beam(self, source: str, k: int = 5) -> list[tuple[str, float]]:
        src = _pad([self.vocab.encode(source, self.max_len)], self.device)
        memory, mask = self.model.encode(src)
        beams: list[tuple[list[int], float]] = [([BOS], 0.0)]
        finished: list[tuple[list[int], float]] = []
        for _ in range(self.max_len):
            tgt = torch.tensor([s for s, _ in beams], device=self.device)
            logits = self.model.decode(tgt, memory.expand(len(beams), -1, -1), mask.expand(len(beams), -1))
            logp = torch.log_softmax(logits[:, -1].float(), dim=-1)
            top_v, top_i = logp.topk(k, dim=-1)
            cands = [(beams[b][0] + [int(top_i[b, j])], beams[b][1] + float(top_v[b, j]))
                     for b in range(len(beams)) for j in range(k)]
            cands.sort(key=lambda x: -x[1])
            beams = []
            for seq, lp in cands:
                if seq[-1] == EOS:
                    finished.append((seq, lp / (len(seq) - 1)))
                else:
                    beams.append((seq, lp))
                if len(beams) == k:
                    break
            if len(finished) >= k or not beams:
                break
        if not finished:
            finished = [(s, lp / max(1, len(s) - 1)) for s, lp in beams]
        finished.sort(key=lambda x: -x[1])
        return [(self.vocab.decode(s[1:]), sc) for s, sc in finished[:k]]

    def generate(self, sources: list[str], num_beams: int = 5, num_return: int = 5) -> list[list[tuple[str, float]]]:
        if num_beams <= 1:
            texts = [self.greedy(s) for s in sources]
            return [[(t, sc)] for t, sc in zip(texts, self.score(sources, texts))]
        return [self.beam(s, num_beams)[:num_return] for s in sources]

    @torch.no_grad()
    def score(self, sources: list[str], targets: list[str], batch_size: int = 64) -> list[float]:
        out: list[float] = []
        for i in range(0, len(sources), batch_size):
            src = _pad([self.vocab.encode(s, self.max_len) for s in sources[i:i + batch_size]], self.device)
            tgt = [self.vocab.encode(t, self.max_len + 1, bos=True) for t in targets[i:i + batch_size]]
            tgt_in = _pad([t[:-1] for t in tgt], self.device)
            tgt_out = _pad([t[1:] for t in tgt], self.device)
            logp = torch.log_softmax(self.model(src, tgt_in).float(), dim=-1)
            mask = tgt_out != PAD
            lp = logp.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1) * mask
            out.extend((lp.sum(-1) / mask.sum(-1).clamp(min=1)).tolist())
        return out


def _batches(pairs, size, rng):
    order = list(range(len(pairs)))
    rng.shuffle(order)
    for i in range(0, len(order), size):
        yield [pairs[j] for j in order[i:i + size]]


def train_char(cfg: dict) -> Path:
    seed = int(cfg.get("seed", 13))
    set_seed(seed)
    rng = random.Random(seed)
    device = pick_device(cfg.get("device", "auto"))
    max_len = int(cfg.get("max_len", 256))
    tcfg, mcfg = cfg["training"], cfg["model"]

    train = [r for p in cfg["train"] if resolve(p).exists() for r in read_jsonl(p)]
    pairs = [record_to_pair(r, rng, float(cfg.get("tag_dropout", 0.0))) for r in train]
    dev_pairs = [record_to_pair(r) for r in read_jsonl(cfg["dev"])][: int(cfg.get("max_dev", 300))] \
        if cfg.get("dev") and resolve(cfg["dev"]).exists() else []
    vocab = CharVocab.build([s for s, _ in pairs] + [t for _, t in pairs])
    model = CharSeq2Seq(len(vocab), **mcfg).to(device)
    wrapper = CharTransformerNormalizer(model, vocab, device, max_len)

    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["learning_rate"]), betas=(0.9, 0.98), eps=1e-9)
    warmup = int(tcfg.get("warmup_steps", 4000))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, (warmup / (s + 1)) ** 0.5))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=float(tcfg.get("label_smoothing", 0.1)))
    out_path = resolve(cfg["output"])
    best, bad = math.inf, 0

    for epoch in range(int(tcfg["epochs"])):
        model.train()
        total = n = 0
        for batch in _batches(pairs, int(tcfg["batch_size"]), rng):
            src = _pad([vocab.encode(s, max_len) for s, _ in batch], device)
            tgt = [vocab.encode(t, max_len + 1, bos=True) for _, t in batch]
            tgt_in, tgt_out = _pad([t[:-1] for t in tgt], device), _pad([t[1:] for t in tgt], device)
            logits = model(src, tgt_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("clip_norm", 1.0)))
            opt.step()
            sched.step()
            total += float(loss)
            n += 1
        model.eval()
        if dev_pairs:
            cer = corpus_cer([wrapper.greedy(s) for s, _ in dev_pairs], [[t] for _, t in dev_pairs])
        else:
            cer = total / max(n, 1)  # sans dev : on suit la perte d'entraînement
        print(f"époque {epoch + 1} : perte {total / max(n, 1):.4f}, CER dev {cer:.4f}")
        if cer < best:
            best, bad = cer, 0
            wrapper.save(out_path, mcfg)
        else:
            bad += 1
            if bad >= int(tcfg.get("patience", 5)):
                print("arrêt anticipé")
                break
    return out_path
