"""Ronda 2 de la comparacion pedida por el usuario: a diferencia de la ronda
1 (Isolation Forest / One-Class SVM, entrenados SOLO con benigno, ver
baselines/classification_metrics.py -- ahi la comparacion es "pura", sin
umbral, solo ROC/AUC), estos 4 modelos son clasificadores SUPERVISADOS
estandar (Random Forest, XGBoost, LightGBM, red neuronal) entrenados con
benigno Y maligno MEZCLADOS (una muestra de ataque etiquetada, pooled de
todas las familias -- mismo protocolo que RandomForestBaseline ya usaba en
run_baselines.py/classification_metrics.py).

A diferencia de T2/SPE (umbral = UCL bootstrap) y de la ronda 1 (sin
umbral), aca SI hay un umbral, pero es el NATIVO del clasificador (regla de
decision estandar de clasificacion binaria: P(ataque) > 0.5) -- no se
calibra nada, es la comparacion "convencional" en el sentido literal.

Reutiliza las muestras de ataque ya cacheadas por classification_metrics.py
(baselines/output/eval_cache/) -- la porcion held-out (--eval-n) para
evaluar, la porcion --rf-train-n para entrenar (pooled de las 12 familias).

Uso (requiere haber corrido antes baselines/classification_metrics.py, para
poblar el cache):
    python3 baselines/supervised_classification_metrics.py
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

from baselines.classification_metrics import per_family_metrics, sample_attack_family_cached
from baselines.run_baselines import load_processed, load_split_features
from baselines.sklearn_models import LightGBMBaseline, RandomForestBaseline, XGBoostBaseline
from baselines.torch_models import NeuralNetBaseline
from baselines import features
from eval_common import binary_metrics, plot_confusion_matrix, roc_points, sanitize_scores

BASELINES_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASELINES_DIR / "output"


def build_supervised_models(input_dim, hidden_dim, seed):
    return {
        "RandomForest": RandomForestBaseline(seed=seed),
        "XGBoost": XGBoostBaseline(seed=seed),
        "LightGBM": LightGBMBaseline(seed=seed),
        "NeuralNet": NeuralNetBaseline(input_dim, hidden_dim, seed=seed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-n", type=int, default=200_000)
    parser.add_argument("--rf-train-n", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    metadata, vocab_maps, scaler = load_processed()
    categorical_cols = metadata["categorical_cols"]
    numeric_cols = metadata["numeric_cols"]
    vocab_sizes = metadata["vocab_sizes"]

    train_data = np.load(config.PROCESSED_DIR / "train.npz")
    cat_train_idx = {c: train_data[f"cat__{c}"] for c in categorical_cols}
    freq_tables = features.build_freq_tables(cat_train_idx, vocab_sizes, categorical_cols)
    X_train = features.encode_features(cat_train_idx, train_data["numeric"], freq_tables, categorical_cols)
    X_test = load_split_features("test", categorical_cols, freq_tables)
    input_dim = X_train.shape[1]

    with open(config.CHECKPOINT_DIR / "optuna_best_params.json") as f:
        best_params = json.load(f)["params"]
    hidden_dim = best_params["hidden_dim"]

    usecols = categorical_cols + numeric_cols + [metadata["timestamp_col"]]
    folders = common.attack_folders()

    print("Cargando muestras de ataque cacheadas (eval + rf-train)...", flush=True)
    eval_by_family, rf_train_parts = {}, []
    for folder in folders:
        X_eval, X_rf_train, n_total_family = sample_attack_family_cached(
            folder, usecols, args.eval_n, args.rf_train_n, rng, metadata, vocab_maps, scaler, freq_tables, categorical_cols)
        if len(X_eval) == 0:
            continue
        eval_by_family[folder.name] = X_eval
        rf_train_parts.append(X_rf_train)
    X_attack_train = np.concatenate(rf_train_parts, axis=0) if rf_train_parts else np.empty((0, input_dim))
    print(f"Entrenamiento supervisado: benigno={len(X_train):,} + ataque etiquetado (pooled, held-out de eval)={len(X_attack_train):,}")

    models = build_supervised_models(input_dim, hidden_dim, args.seed)
    scores_test, preds_test = {}, {}
    scores_eval = {name: {} for name in models}
    preds_eval = {name: {} for name in models}

    for name, model in models.items():
        t0 = time.time()
        print(f"[{name}] entrenando con benigno + {len(X_attack_train):,} filas de ataque...", flush=True)
        model.fit(X_train, X_attack_train)
        scores_test[name] = sanitize_scores(model.score(X_test))
        preds_test[name] = model.predict_anomaly(X_test)
        for fam_name, X_eval in eval_by_family.items():
            scores_eval[name][fam_name] = sanitize_scores(model.score(X_eval))
            preds_eval[name][fam_name] = model.predict_anomaly(X_eval)
        print(f"[{name}] listo ({time.time() - t0:.0f}s)", flush=True)

    combined_rows, per_family_rows, roc_curves = [], [], {}
    for method in models:
        y_true = np.concatenate([np.zeros(len(X_test), dtype=np.int8)] +
                                 [np.ones(len(s), dtype=np.int8) for s in scores_eval[method].values()])
        y_score = np.concatenate([scores_test[method]] + list(scores_eval[method].values()))
        y_pred = np.concatenate([preds_test[method]] + list(preds_eval[method].values())).astype(np.int8)

        metrics = binary_metrics(y_true, y_pred, y_score, method)
        metrics["method"] = method
        combined_rows.append(metrics)

        fpr, tpr = roc_points(y_true, y_score)
        roc_curves[method] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": metrics["auc"]}

        # Umbral nativo (P(ataque) > 0.5) fijo -- reusa per_family_metrics
        # con ese "umbral" solo para armar la matriz de confusion por
        # familia de forma consistente (predict_anomaly ya aplico 0.5).
        for row in per_family_metrics(scores_test[method], scores_eval[method], threshold=0.5):
            row["method"] = method
            per_family_rows.append(row)

        cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        plot_confusion_matrix(cm, f"Confusion Matrix -- {method} (supervisado, umbral nativo=0.5)",
                               OUTPUT_DIR / f"confusion_matrix_{method}.png")

        print(f"[{method}] AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
              f"precision={metrics['precision']:.4f}  recall={metrics['recall_sensitivity']:.4f}")

    combined_df = pd.DataFrame(combined_rows).drop(columns=["label"])
    combined_df.to_csv(OUTPUT_DIR / "supervised_classification_metrics_combined.csv", index=False)
    per_family_df = pd.DataFrame(per_family_rows)
    per_family_df.to_csv(OUTPUT_DIR / "supervised_classification_metrics_per_family.csv", index=False)
    with open(OUTPUT_DIR / "roc_points_supervised.json", "w") as f:
        json.dump(roc_curves, f)

    print(f"\nSaved: {OUTPUT_DIR / 'supervised_classification_metrics_combined.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'supervised_classification_metrics_per_family.csv'}")
    print(f"Tiempo total: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
