"""Graficos de la comparacion FlowVAE vs. baselines (ver run_baselines.py).

Dos figuras:
  - Heatmap de recall (%% filas marcadas anomalas) por familia de ataque x
    metodo -- magnitud sobre dos ejes categoricos, un solo hue secuencial
    (azul, claro->oscuro), con el valor anotado en cada celda.
  - Barras horizontales del falso-positivo sobre benigno held-out (test.npz)
    por metodo -- una magnitud por entidad, color categorico fijo por
    metodo (mismo mapeo de color que se reusaria en cualquier otra figura
    de esta comparacion).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

METHOD_ORDER = [
    "IsolationForest", "OneClassSVM", "LOF", "PCA", "Autoencoder", "DeepSVDD", "VanillaVAE",
    "RandomForest (supervised)", "FlowVAE_T2 (proposed)", "FlowVAE_SPE (proposed)",
]
METHOD_COLORS = {
    "IsolationForest": "#2a78d6",
    "OneClassSVM": "#eb6834",
    "LOF": "#6a3d9a",
    "PCA": "#a3522a",
    "Autoencoder": "#1baf7a",
    "DeepSVDD": "#eda100",
    "VanillaVAE": "#e87ba4",
    "RandomForest (supervised)": "#008300",
    "FlowVAE_T2 (proposed)": "#4a3aa7",
    "FlowVAE_SPE (proposed)": "#e34948",
}
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#3987e5", "#0d366b"])


def plot_heatmap(wide_df, out_path):
    families = [a for a in wide_df.index if a != "(benign test)"]
    methods = [m for m in METHOD_ORDER if m in wide_df.columns]
    data = wide_df.loc[families, methods].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(1.15 * len(methods) + 2, 0.4 * len(families) + 2))
    im = ax.imshow(data, cmap=SEQ_BLUE, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=9)

    for i in range(len(families)):
        for j in range(len(methods)):
            v = data[i, j]
            color = "white" if v > 55 else "#0b0b0b"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("% of attack rows flagged anomalous (recall)")
    ax.set_title("FlowVAE vs. baselines -- recall by attack family")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_benign_fpr(wide_df, out_path):
    row = wide_df.loc["(benign test)"]
    methods = [m for m in METHOD_ORDER if m in wide_df.columns]
    values = [row[m] for m in methods]
    order = np.argsort(values)
    methods = [methods[i] for i in order]
    values = [values[i] for i in order]
    colors = [METHOD_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(methods) + 1.5))
    bars = ax.barh(methods, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=9)

    ax.set_xlim(0, 100)
    ax.set_xlabel("% of held-out benign test rows flagged anomalous -- false positive rate")
    ax.set_title("False positive rate on held-out benign traffic, by method")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    wide_df = pd.read_csv(OUTPUT_DIR / "baselines_comparison_wide.csv", index_col=0)
    plot_heatmap(wide_df, OUTPUT_DIR / "baselines_heatmap_recall.png")
    plot_benign_fpr(wide_df, OUTPUT_DIR / "baselines_fpr_benign.png")
    print(f"Saved: {OUTPUT_DIR / 'baselines_heatmap_recall.png'}")
    print(f"Saved: {OUTPUT_DIR / 'baselines_fpr_benign.png'}")


if __name__ == "__main__":
    main()
