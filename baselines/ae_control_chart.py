"""Replica la metodologia de cartas de control T2/SPE del FlowVAE propuesto
(ver control_charts/phase1.py, control_charts/classification_metrics.py)
para un Autoencoder ESTANDAR (baselines/torch_models.py:AutoencoderDetector
-- sin variational, sin KL, entrenado solo con MSE), sobre la misma
representacion de features compartida por los baselines (frequency encoding,
ver features.py) en vez de los embeddings aprendidos del FlowVAE.

Igual que el FlowVAE:
  - T2 = distancia de Mahalanobis del bottleneck z=encoder(x) al centro de
    la nube de benigno de train (covarianza con shrinkage Ledoit-Wolf).
  - SPE = error de reconstruccion (decoder(encoder(x)) vs. x).
  - El UCL de cada carta se calibra SOLO con benigno de train.npz, via el
    mismo bootstrap de bloques moviles que phase1.py (common.py).
  - Metricas: matriz de confusion (stat > UCL) + ROC/AUC, benigno test y
    cada familia de ataque MEZCLADOS (pooled) -- por familia y combinado.

Diferencia de poblacion (documentada tambien en run_baselines.py): el AE se
evalua sobre las muestras de hasta --eval-n+--rf-train-n filas por familia ya
cacheadas por baselines/classification_metrics.py (baselines/output/eval_cache/),
no sobre la poblacion completa de cada ataque como hace el FlowVAE (que si
puede permitirse re-leer los CSVs completos porque ya estan cacheados por
fila en control_charts/output/scores/).

Uso (requiere haber corrido antes baselines/classification_metrics.py al
menos una vez, para poblar baselines/output/eval_cache/):
    python3 baselines/ae_control_chart.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from control_charts import common

from baselines.run_baselines import load_processed, load_split_features
from baselines.torch_models import AutoencoderDetector
from baselines import features
from eval_common import binary_metrics, plot_confusion_matrix, roc_points

BASELINES_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASELINES_DIR / "output"
EVAL_CACHE_DIR = OUTPUT_DIR / "eval_cache"


def load_cached_family_sample(name):
    safe_name = name.replace(" ", "_")
    cache_path = EVAL_CACHE_DIR / f"{safe_name}.npz"
    if not cache_path.exists():
        return None
    d = np.load(cache_path)
    return d["X_eval"]  # muestra completa cacheada (ver classification_metrics.py)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    metadata, vocab_maps, scaler = load_processed()
    categorical_cols = metadata["categorical_cols"]
    vocab_sizes = metadata["vocab_sizes"]

    train_data = np.load(config.PROCESSED_DIR / "train.npz")
    cat_train_idx = {c: train_data[f"cat__{c}"] for c in categorical_cols}
    freq_tables = features.build_freq_tables(cat_train_idx, vocab_sizes, categorical_cols)
    X_train = features.encode_features(cat_train_idx, train_data["numeric"], freq_tables, categorical_cols)
    X_test = load_split_features("test", categorical_cols, freq_tables)
    input_dim = X_train.shape[1]

    with open(config.CHECKPOINT_DIR / "optuna_best_params.json") as f:
        best_params = json.load(f)["params"]
    hidden_dim, z_dim = best_params["hidden_dim"], best_params["z_dim"]

    print(f"Entrenando AE estandar sobre benigno (train.npz, n={len(X_train)}), "
          f"hidden_dim={hidden_dim}, z_dim={z_dim}...", flush=True)
    ae = AutoencoderDetector(input_dim, hidden_dim, z_dim, seed=args.seed)
    ae.fit(X_train)

    z_train = ae.encode(X_train)
    spe_train = ae.score(X_train)
    z_mean, cov_inv = common.fit_t2_reference(z_train)
    t2_train = common.hotelling_t2(z_train, z_mean, cov_inv)

    block_length = common.default_block_length(len(X_train))
    ucl_t2, _ = common.moving_block_bootstrap_ucl(t2_train, block_length, args.n_boot, args.quantile, rng)
    ucl_spe, _ = common.moving_block_bootstrap_ucl(spe_train, block_length, args.n_boot, args.quantile, rng)
    print(f"UCL_T2={ucl_t2:.4f}  UCL_SPE={ucl_spe:.4f}  (calibrados solo con train benigno, quantile={args.quantile})")

    z_test = ae.encode(X_test)
    t2_benign = common.hotelling_t2(z_test, z_mean, cov_inv)
    spe_benign = ae.score(X_test)
    benign_label = np.zeros(len(X_test), dtype=np.int8)

    folders = common.attack_folders()
    per_attack_rows = []
    t2_parts, spe_parts, label_parts = [t2_benign], [spe_benign], [benign_label]

    for folder in folders:
        X_eval = load_cached_family_sample(folder.name)
        if X_eval is None:
            print(f"[{folder.name}] sin muestra cacheada (corre baselines/classification_metrics.py primero), se omite")
            continue
        z_a = ae.encode(X_eval)
        t2_a = common.hotelling_t2(z_a, z_mean, cov_inv)
        spe_a = ae.score(X_eval)
        t2_parts.append(t2_a); spe_parts.append(spe_a)
        label_parts.append(np.ones(len(t2_a), dtype=np.int8))

        y_family = np.concatenate([benign_label, np.ones(len(t2_a), dtype=np.int8)])
        t2_family = np.concatenate([t2_benign, t2_a])
        spe_family = np.concatenate([spe_benign, spe_a])
        m_t2 = binary_metrics(y_family, (t2_family > ucl_t2).astype(np.int8), t2_family, "T2")
        m_spe = binary_metrics(y_family, (spe_family > ucl_spe).astype(np.int8), spe_family, "SPE")
        row = {"attack": folder.name, "n_rows": len(t2_a)}
        row.update({f"{k}_t2": v for k, v in m_t2.items() if k != "label"})
        row.update({f"{k}_spe": v for k, v in m_spe.items() if k != "label"})
        per_attack_rows.append(row)
        print(f"[{folder.name}] n={len(t2_a):,}  recall_T2={m_t2['recall_sensitivity']:.3f}  "
              f"recall_SPE={m_spe['recall_sensitivity']:.3f}")

    y_true = np.concatenate(label_parts)
    t2_all = np.concatenate(t2_parts)
    spe_all = np.concatenate(spe_parts)

    metrics_t2 = binary_metrics(y_true, (t2_all > ucl_t2).astype(np.int8), t2_all, "AE_T2")
    metrics_spe = binary_metrics(y_true, (spe_all > ucl_spe).astype(np.int8), spe_all, "AE_SPE")

    fpr_t2, tpr_t2 = roc_points(y_true, t2_all)
    fpr_spe, tpr_spe = roc_points(y_true, spe_all)

    cm_t2 = np.array([[metrics_t2["tn"], metrics_t2["fp"]], [metrics_t2["fn"], metrics_t2["tp"]]])
    cm_spe = np.array([[metrics_spe["tn"], metrics_spe["fp"]], [metrics_spe["fn"], metrics_spe["tp"]]])
    plot_confusion_matrix(cm_t2, f"Confusion Matrix -- AE T2 (UCL={ucl_t2:.2f})", OUTPUT_DIR / "confusion_matrix_AE_T2.png")
    plot_confusion_matrix(cm_spe, f"Confusion Matrix -- AE SPE (UCL={ucl_spe:.2f})", OUTPUT_DIR / "confusion_matrix_AE_SPE.png")

    combined_df = pd.DataFrame([metrics_t2, metrics_spe]).rename(columns={"label": "chart"})
    combined_df.to_csv(OUTPUT_DIR / "ae_classification_metrics_combined.csv", index=False)
    pd.DataFrame(per_attack_rows).to_csv(OUTPUT_DIR / "ae_classification_metrics_per_family.csv", index=False)

    with open(OUTPUT_DIR / "roc_points_ae.json", "w") as f:
        json.dump({
            "t2": {"fpr": fpr_t2.tolist(), "tpr": tpr_t2.tolist(), "auc": metrics_t2["auc"]},
            "spe": {"fpr": fpr_spe.tolist(), "tpr": tpr_spe.tolist(), "auc": metrics_spe["auc"]},
        }, f)

    print("\n=== AE T2 (combined) ===")
    print(combined_df[combined_df.chart == "AE_T2"].to_string(index=False))
    print("\n=== AE SPE (combined) ===")
    print(combined_df[combined_df.chart == "AE_SPE"].to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'ae_classification_metrics_combined.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'ae_classification_metrics_per_family.csv'}")
    print(f"Tiempo total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
