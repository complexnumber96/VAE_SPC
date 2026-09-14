"""Matrices de confusion, metricas de clasificacion y curvas ROC/AUC para las
cartas de control T2/SPE (ver phase1.py, phase2.py, common.py).

Trata la deteccion "fuera de control" como un clasificador binario:
  - Clase negativa (0): trafico benigno held-out (processed/test.npz -- NO
    participo en la calibracion del UCL, ver phase1.py).
  - Clase positiva (1): trafico de cada familia de malware bajo MALWARE_ROOT
    (las mismas carpetas que recorre phase2.py).

Para cada punto se calculan T2 y SPE (mismo pipeline que phase1/phase2). Cada
estadistico se evalua como clasificador independiente:
  - Regla binaria: predicho positivo si stat > UCL (el UCL de fase 1,
    bootstrap de bloques moviles sobre train benigno) -> matriz de confusion,
    accuracy, precision, recall/sensibilidad, especificidad, F1, balanced
    accuracy, MCC, FPR, FNR.
  - Score continuo: ROC y AUC de stat vs. label (T2 y SPE per separado, sin
    combinar ambos en un solo score).

Los resultados por fila de cada ataque se cachean en output/scores/<ataque>.npz
(mismo costo de encoder que phase2.py, ~47M filas totales) para poder recalcular
metricas sin repetir el paso mas caro (lectura+encode de CSVs de varios GB).

Ademas de la matriz de confusion/AUC agregada (benigno test + TODOS los
ataques pooled), se reporta el desglose por familia: cada familia se agrupa
con el mismo benigno test compartido (benigno=0, esa familia=1) y se evalua
con el MISMO UCL global -- no se re-calibra un umbral por familia, porque en
despliegue real no se sabe de antemano que familia de ataque se esta viendo.
El AUC por familia si es informativo aun con umbral fijo, porque no depende
de el (mide que tan separable es esa familia del benigno en el score
continuo, mas alla de donde caiga el UCL).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common
from phase2 import score_csv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_common import binary_metrics, plot_confusion_matrix, roc_points

SCORES_DIR = common.OUTPUT_DIR / "scores"


def load_benign_test():
    import config  # sys.path already patched by `import common`
    data = np.load(config.PROCESSED_DIR / "test.npz")
    with open(config.PROCESSED_DIR / "metadata.json") as f:
        categorical_cols = json.load(f)["categorical_cols"]
    cat_arrays = {c: data[f"cat__{c}"] for c in categorical_cols}
    return cat_arrays, data["numeric"]


def score_benign_test(vae, z_mean, cov_inv):
    cache_path = SCORES_DIR / "benign_test.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        return d["t2"], d["spe"]

    cat_arrays, num_array = load_benign_test()
    z, spe = common.encode_batch(vae, cat_arrays, num_array)
    t2 = common.hotelling_t2(z, z_mean, cov_inv)

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, t2=t2, spe=spe)
    return t2, spe


def score_attack(folder, vae, metadata, vocab_maps, scaler, z_mean, cov_inv):
    safe_name = folder.name.replace(" ", "_")
    cache_path = SCORES_DIR / f"{safe_name}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        return d["t2"], d["spe"]

    usecols = [c for c in (metadata["categorical_cols"] + metadata["numeric_cols"] + [metadata["timestamp_col"]])]
    csvs = common.attack_csvs(folder)
    t2_all, spe_all = [], []
    t0 = time.time()
    for path in csvs:
        _, t2, spe = score_csv(path, vae, metadata, vocab_maps, scaler, z_mean, cov_inv, usecols)
        t2_all.append(t2)
        spe_all.append(spe)
    t2 = np.concatenate(t2_all)
    spe = np.concatenate(spe_all)
    print(f"[{folder.name}] scored {len(t2):,} rows in {time.time() - t0:.0f}s")

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, t2=t2, spe=spe)
    return t2, spe


def plot_roc(curves, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (fpr, tpr, auc), color in curves:
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, linewidth=1.8)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC -- T2 vs SPE as anomaly scores (benign test vs. all attacks)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, default=None, help="Solo (re)scorea esta carpeta de ataque")
    args = parser.parse_args()

    ref = np.load(common.PHASE1_STATS_PATH)
    z_mean, cov_inv = ref["z_mean"], ref["cov_inv"]
    ucl_t2, ucl_spe = float(ref["ucl_t2"]), float(ref["ucl_spe"])
    print(f"Phase 1 reference: UCL_T2={ucl_t2:.4f}  UCL_SPE={ucl_spe:.4f}")

    vae, metadata, vocab_maps, scaler = common.load_artifacts()

    print("Scoring benign held-out test...")
    t2_benign, spe_benign = score_benign_test(vae, z_mean, cov_inv)
    print(f"  benign test: n={len(t2_benign)}")

    folders = common.attack_folders()
    if args.attack:
        folders = [f for f in folders if f.name == args.attack]
        if not folders:
            raise SystemExit(f"Attack folder '{args.attack}' does not exist")

    benign_label = np.zeros(len(t2_benign), dtype=np.int8)
    per_attack_rows = []
    t2_parts = [t2_benign]
    spe_parts = [spe_benign]
    label_parts = [benign_label]

    for folder in folders:
        print(f"Scoring attack: {folder.name}")
        t2_a, spe_a = score_attack(folder, vae, metadata, vocab_maps, scaler, z_mean, cov_inv)
        t2_parts.append(t2_a)
        spe_parts.append(spe_a)
        label_parts.append(np.ones(len(t2_a), dtype=np.int8))

        # Cada familia se agrupa con el MISMO benigno test compartido y se
        # evalua contra el UCL global (no un umbral re-calibrado por
        # familia -- ver docstring del modulo). Da matriz de confusion +
        # AUC por familia, no solo el recall.
        y_family = np.concatenate([benign_label, np.ones(len(t2_a), dtype=np.int8)])
        t2_family = np.concatenate([t2_benign, t2_a])
        spe_family = np.concatenate([spe_benign, spe_a])
        pred_t2_family = (t2_family > ucl_t2).astype(np.int8)
        pred_spe_family = (spe_family > ucl_spe).astype(np.int8)
        m_t2 = binary_metrics(y_family, pred_t2_family, t2_family, "T2")
        m_spe = binary_metrics(y_family, pred_spe_family, spe_family, "SPE")

        per_attack_rows.append({
            "attack": folder.name,
            "n_rows": len(t2_a),
            "recall_t2": m_t2["recall_sensitivity"],
            "recall_spe": m_spe["recall_sensitivity"],
            "precision_t2": m_t2["precision"],
            "precision_spe": m_spe["precision"],
            "f1_t2": m_t2["f1"],
            "f1_spe": m_spe["f1"],
            "mcc_t2": m_t2["mcc"],
            "mcc_spe": m_spe["mcc"],
            "auc_t2": m_t2["auc"],
            "auc_spe": m_spe["auc"],
            "tn_t2": m_t2["tn"], "fp_t2": m_t2["fp"], "fn_t2": m_t2["fn"], "tp_t2": m_t2["tp"],
            "tn_spe": m_spe["tn"], "fp_spe": m_spe["fp"], "fn_spe": m_spe["fn"], "tp_spe": m_spe["tp"],
        })

    if args.attack:
        print(f"Solo se (re)scoreo '{args.attack}'. Corre sin --attack para agregar todas las metricas.")
        return

    y_true = np.concatenate(label_parts)
    t2_all = np.concatenate(t2_parts)
    spe_all = np.concatenate(spe_parts)
    print(f"Total pooled: n={len(y_true):,}  benign={int((y_true == 0).sum()):,}  attack={int((y_true == 1).sum()):,}")

    pred_t2 = (t2_all > ucl_t2).astype(np.int8)
    pred_spe = (spe_all > ucl_spe).astype(np.int8)

    metrics_t2 = binary_metrics(y_true, pred_t2, t2_all, "T2")
    metrics_spe = binary_metrics(y_true, pred_spe, spe_all, "SPE")

    cm_t2 = np.array([[metrics_t2["tn"], metrics_t2["fp"]], [metrics_t2["fn"], metrics_t2["tp"]]])
    cm_spe = np.array([[metrics_spe["tn"], metrics_spe["fp"]], [metrics_spe["fn"], metrics_spe["tp"]]])

    fpr_t2, tpr_t2 = roc_points(y_true, t2_all)
    fpr_spe, tpr_spe = roc_points(y_true, spe_all)

    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm_t2, f"Confusion Matrix -- T2 (UCL={ucl_t2:.2f})",
                           common.OUTPUT_DIR / "confusion_matrix_T2.png")
    plot_confusion_matrix(cm_spe, f"Confusion Matrix -- SPE (UCL={ucl_spe:.2f})",
                           common.OUTPUT_DIR / "confusion_matrix_SPE.png")
    plot_roc(
        [
            ("T2", (fpr_t2, tpr_t2, metrics_t2["auc"]), "#3366cc"),
            ("SPE", (fpr_spe, tpr_spe, metrics_spe["auc"]), "#d62728"),
        ],
        common.OUTPUT_DIR / "roc_curves.png",
    )

    metrics_df = pd.DataFrame([metrics_t2, metrics_spe]).rename(columns={"label": "chart"})
    metrics_path = common.OUTPUT_DIR / "classification_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    per_attack_df = pd.DataFrame(per_attack_rows)
    per_attack_path = common.OUTPUT_DIR / "classification_per_attack.csv"
    per_attack_df.to_csv(per_attack_path, index=False)

    roc_points_json = {
        "t2": {"fpr": fpr_t2.tolist(), "tpr": tpr_t2.tolist()},
        "spe": {"fpr": fpr_spe.tolist(), "tpr": tpr_spe.tolist()},
    }
    with open(common.OUTPUT_DIR / "roc_points.json", "w") as f:
        json.dump(roc_points_json, f)

    print("\n=== T2 (combined) ===")
    print(metrics_df[metrics_df.chart == "T2"].to_string(index=False))
    print("\n=== SPE (combined) ===")
    print(metrics_df[metrics_df.chart == "SPE"].to_string(index=False))
    print("\n=== Per-family ===")
    print(per_attack_df.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {per_attack_path}")
    print(f"Charts: {common.OUTPUT_DIR / 'confusion_matrix_T2.png'}, "
          f"{common.OUTPUT_DIR / 'confusion_matrix_SPE.png'}, {common.OUTPUT_DIR / 'roc_curves.png'}")


if __name__ == "__main__":
    main()
