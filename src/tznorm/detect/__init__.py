"""M1 · Détection de l'écriture, de la langue et de la variété."""

from .lang import Detector, NgramNB
from .script import char_script, text_script, token_script

__all__ = ["Detector", "NgramNB", "char_script", "text_script", "token_script"]
