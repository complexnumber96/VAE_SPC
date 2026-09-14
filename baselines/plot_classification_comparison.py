"""Comparacion final de las 3 rondas pedidas por el usuario:

  Ronda A (SPC, umbral = UCL bootstrap sobre benigno de train): FlowVAE T2/SPE
    (ver control_charts/classification_metrics.py) vs. AE estandar T2/SPE
    (ver baselines/ae_control_chart.py) -- la comparacion central: VAE+SPE
    vs. AE+SPE.
  Ronda B (entrenados SOLO con benigno, comparacion "pura" sin umbral --
    ROC/AUC): Isolation Forest, One-Class SVM, LOF, PCA, Autoencoder
    (reconstruccion simple), Deep SVDD, VAE convencional (ver
    baselines/classification_metrics.py). El umbral de mejor-F1 que ese
    script calcula queda como dato supletorio, no es el foco de esta ronda.
  Ronda C (supervisados, entrenados con benigno+maligno MEZCLADO, umbral
    nativo P(ataque)>0.5): Random Forest, XGBoost, LightGBM, red neuronal
    (ver baselines/supervised_classification_metrics.py).

Dos figuras:
  - ROC en 3 paneles (uno por ronda) -- separar por ronda evita el
    "espagueti" de 15 curvas en un solo eje y deja comparar honestamente
    solo dentro de metodologias compatibles.
  - Barras de AUC (todas las rondas juntas, ordenadas de mayor a menor,
    coloreadas por metodo con la MISMA paleta de los paneles) -- la
    comparacion "capacidad predictiva pura" que pidio el usuario, ya que
    AUC no depende de ningun umbral y es comparable entre las 3 rondas.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONTROL_CHARTS_OUTPUT = ROOT / "control_charts" / "output"
BASELINES_OUTPUT = ROOT / "baselines" / "output"

ROUNDS = {
    "A: SPC (UCL bootstrap)": ["FlowVAE_T2", "FlowVAE_SPE", "AE_T2", "AE_SPE"],
    "B: benign-only (pure ROC/AUC)": ["IsolationForest", "OneClassSVM", "LOF", "PCA", "Autoencoder", "DeepSVDD", "VanillaVAE"],
    "C: supervised (benign+attack, mixed training)": ["RandomForest", "XGBoost", "LightGBM", "NeuralNet"],
}
METHOD_COLORS = {
    "FlowVAE_T2": "#4a3aa7", "FlowVAE_SPE": "#e34948", "AE_T2": "#8a7fd6", "AE_SPE": "#c9524f",
    "IsolationForest": "#2a78d6", "OneClassSVM": "#eb6834", "LOF": "#6a3d9a", "PCA": "#a3522a",
    "Autoencoder": "#1baf7a", "DeepSVDD": "#eda100", "VanillaVAE": "#e87ba4",
    "RandomForest": "#008300", "XGBoost": "#c9862f", "LightGBM": "#2f8fc9", "NeuralNet": "#c92f8f",
}


def load_all_curves():
    curves = {}

    with open(CONTROL_CHARTS_OUTPUT / "roc_points.json") as f:
        flowvae_roc = json.load(f)
    flowvae_metrics = pd.read_csv(CONTROL_CHARTS_OUTPUT / "classification_metrics.csv").set_index("chart")
    curves["FlowVAE_T2"] = {"fpr": np.array(flowvae_roc["t2"]["fpr"]), "tpr": np.array(flowvae_roc["t2"]["tpr"]),
                             "auc": float(flowvae_metrics.loc["T2", "auc"]), "f1": float(flowvae_metrics.loc["T2", "f1"])}
    curves["FlowVAE_SPE"] = {"fpr": np.array(flowvae_roc["spe"]["fpr"]), "tpr": np.array(flowvae_roc["spe"]["tpr"]),
                              "auc": float(flowvae_metrics.loc["SPE", "auc"]), "f1": float(flowvae_metrics.loc["SPE", "f1"])}

    with open(BASELINES_OUTPUT / "roc_points_ae.json") as f:
        ae_roc = json.load(f)
    ae_metrics = pd.read_csv(BASELINES_OUTPUT / "ae_classification_metrics_combined.csv").set_index("chart")
    curves["AE_T2"] = {"fpr": np.array(ae_roc["t2"]["fpr"]), "tpr": np.array(ae_roc["t2"]["tpr"]),
                        "auc": float(ae_metrics.loc["AE_T2", "auc"]), "f1": float(ae_metrics.loc["AE_T2", "f1"])}
    curves["AE_SPE"] = {"fpr": np.array(ae_roc["spe"]["fpr"]), "tpr": np.array(ae_roc["spe"]["tpr"]),
                         "auc": float(ae_metrics.loc["AE_SPE", "auc"]), "f1": float(ae_metrics.loc["AE_SPE", "f1"])}

    with open(BASELINES_OUTPUT / "roc_points_baselines.json") as f:
        roundb_roc = json.load(f)
    roundb_metrics = pd.read_csv(BASELINES_OUTPUT / "classification_metrics_combined.csv").set_index("method")
    for method in ROUNDS["B: benign-only (pure ROC/AUC)"]:
        p = roundb_roc[method]
        curves[method] = {"fpr": np.array(p["fpr"]), "tpr": np.array(p["tpr"]),
                           "auc": p["auc"], "f1": float(roundb_metrics.loc[method, "f1"])}

    with open(BASELINES_OUTPUT / "roc_points_supervised.json") as f:
        roundc_roc = json.load(f)
    roundc_metrics = pd.read_csv(BASELINES_OUTPUT / "supervised_classification_metrics_combined.csv").set_index("method")
    for method in ROUNDS["C: supervised (benign+attack, mixed training)"]:
        p = roundc_roc[method]
        curves[method] = {"fpr": np.array(p["fpr"]), "tpr": np.array(p["tpr"]),
                           "auc": p["auc"], "f1": float(roundc_metrics.loc[method, "f1"])}

    return curves


def plot_roc_panels(curves, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (round_name, methods) in zip(axes, ROUNDS.items()):
        for method in methods:
            c = curves[method]
            ax.plot(c["fpr"], c["tpr"], label=f"{method} (AUC={c['auc']:.3f})",
                    color=METHOD_COLORS[method], linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle=":", color="#999999", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(round_name, fontsize=10)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    fig.suptitle("ROC by round -- FlowVAE+AE (SPC/UCL) vs. benign-only (no threshold) vs. supervised (mixed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_auc_ranked_bar(curves, out_path):
    all_methods = [m for methods in ROUNDS.values() for m in methods]
    all_methods = sorted(all_methods, key=lambda m: curves[m]["auc"])
    aucs = [curves[m]["auc"] for m in all_methods]
    colors = [METHOD_COLORS[m] for m in all_methods]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(all_methods) + 1.5))
    bars = ax.barh(all_methods, aucs, color=colors)
    for bar, v in zip(bars, aucs):
        ax.text(min(v + 0.01, 0.97), bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(0.5, color="#999999", linestyle=":", linewidth=1)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUC (threshold-independent -- comparable across all 3 rounds)")
    ax.set_title("Pure predictive capacity: AUC across all 3 rounds, ranked")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    curves = load_all_curves()
    BASELINES_OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_roc_panels(curves, BASELINES_OUTPUT / "roc_curves_all_methods.png")
    plot_auc_ranked_bar(curves, BASELINES_OUTPUT / "auc_f1_comparison.png")

    rows = []
    for round_name, methods in ROUNDS.items():
        for m in methods:
            rows.append({"round": round_name, "method": m, "auc": curves[m]["auc"], "f1": curves[m]["f1"]})
    summary = pd.DataFrame(rows).sort_values("auc", ascending=False)
    summary.to_csv(BASELINES_OUTPUT / "final_comparison_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {BASELINES_OUTPUT / 'roc_curves_all_methods.png'}")
    print(f"Saved: {BASELINES_OUTPUT / 'auc_f1_comparison.png'}")
    print(f"Saved: {BASELINES_OUTPUT / 'final_comparison_summary.csv'}")


if __name__ == "__main__":
    main()
