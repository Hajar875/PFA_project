"""A · Fabrique de données synthétiques."""

from .calibrate import calibrate
from .generate import generate_synthetic
from .noise import Noiser
from .reverse import Renderer

__all__ = ["Noiser", "Renderer", "calibrate", "generate_synthetic"]
