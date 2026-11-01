"""Démonstration Gradio. Lancement : python demo/app.py"""

from __future__ import annotations

import os

import gradio as gr

from tznorm.pipeline import Normalizer

norm = Normalizer.from_config(os.environ.get("TZNORM_CONFIG", "configs/pipeline.yaml"))

EXEMPLES = [
    ["tanmirt s watas", "auto", "auto"],
    ["أزول", "auto", "auto"],
    ["thamazighth", "rif", "auto"],
    ["ur ssnx", "auto", "auto"],
]


def run(text: str, variety: str, script: str):
    res = norm(text, None if variety == "auto" else variety, None if script == "auto" else script)
    details = {"écriture détectée": res.ecriture_detectee, "variété": res.variete, "confiance": res.confiance,
               "source": res.source, "alternatives": res.alternatives}
    return res.sortie, details


with gr.Blocks(title="tznorm") as demo:
    gr.Markdown("# Normalisation du tamazight vers le Tifinagh IRCAM\nLatin, arabe ou Tifinagh, formel ou informel.")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Texte", lines=3, placeholder="azul fellawen")
            variety = gr.Dropdown(["auto", "rif", "tzm", "shi"], value="auto", label="Variété")
            script = gr.Dropdown(["auto", "lat", "ara", "tfng"], value="auto", label="Écriture")
            btn = gr.Button("Normaliser", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Tifinagh IRCAM", lines=3)
            info = gr.JSON(label="Détails")
    btn.click(run, [text, variety, script], [out, info])
    gr.Examples(EXEMPLES, [text, variety, script])

if __name__ == "__main__":
    demo.launch()
