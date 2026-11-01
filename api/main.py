"""Service REST.

Lancement : uvicorn api.main:app --reload
Configuration : variable d'environnement TZNORM_CONFIG (par défaut configs/pipeline.yaml).
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tznorm import __version__
from tznorm.pipeline import Normalizer

MAX_CHARS = 2000
app = FastAPI(title="tznorm", version=__version__,
              description="Normalisation du tamazight marocain (latin, arabe, Tifinagh) vers le Tifinagh IRCAM")


@lru_cache(maxsize=1)
def get_normalizer() -> Normalizer:
    return Normalizer.from_config(os.environ.get("TZNORM_CONFIG", "configs/pipeline.yaml"))


class NormalizeRequest(BaseModel):
    text: str = Field(..., description="texte à normaliser")
    variety: str | None = Field(None, pattern="^(rif|tzm|shi)$")
    script: str | None = Field(None, pattern="^(lat|ara|tfng)$")


class BatchRequest(BaseModel):
    texts: list[str]
    variety: str | None = Field(None, pattern="^(rif|tzm|shi)$")


@app.get("/health")
def health() -> dict:
    norm = get_normalizer()
    return {"status": "ok", "version": __version__, "modele": norm.model is not None, "regles": norm.rules.name}


@app.post("/normalize")
def normalize(req: NormalizeRequest) -> dict:
    if len(req.text) > MAX_CHARS:
        raise HTTPException(413, f"texte limité à {MAX_CHARS} caractères")
    return get_normalizer()(req.text, req.variety, req.script).to_dict()


@app.post("/normalize/batch")
def normalize_batch(req: BatchRequest) -> list[dict]:
    if len(req.texts) > 256 or any(len(t) > MAX_CHARS for t in req.texts):
        raise HTTPException(413, f"au plus 256 textes de {MAX_CHARS} caractères")
    res = get_normalizer().normalize_batch(req.texts, [req.variety] * len(req.texts))
    return [r.to_dict() for r in res]
