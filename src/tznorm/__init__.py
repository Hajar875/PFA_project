"""tznorm : normalisation du tamazight marocain (latin, arabe, Tifinagh) vers le Tifinagh IRCAM."""

__version__ = "0.1.0"
__all__ = ["Normalizer", "NormResult", "__version__"]


def __getattr__(name):
    # Import paresseux : `import tznorm` ne charge ni torch ni transformers.
    if name in ("Normalizer", "NormResult"):
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module 'tznorm' has no attribute {name!r}")
