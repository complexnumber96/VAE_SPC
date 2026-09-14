"""Matrices de confusion, metricas de clasificacion y curvas ROC/AUC para los
baselines no-supervisados (Isolation Forest, One-Class SVM, LOF, PCA,
Autoencoder, Deep SVDD, VAE convencional) y el Random Forest supervisado --
para comparar contra las cartas de control T2/SPE del FlowVAE propuesto
(ver control_charts/classification_metrics.py).

Diferencia deliberada con el FlowVAE: aca el umbral NO se calibra con un
bootstrap de bloques moviles sobre benigno de train (como el UCL de fase 1).
Se usa el criterio "convencional" de deteccion de anomalias cuando se quiere
reportar matriz de confusion: el umbral que maximiza F1 sobre el score,
barriendo todos los puntos de corte (ver eval_common.best_f1_threshold).
A diferencia del UCL, este umbral SI mira ataques (aunque sea de forma
agregada, no por familia) -- es una comparacion deliberadamente favorable a
los baselines en ese sentido; el ROC-AUC (que no depende de ningun umbral)
es la metrica mas justa para comparar contra el FlowVAE.

Metodologia (ver baselines/run_baselines.py para el resto del protocolo:
frequency-encoding compartido, benigno SOLO en train, ataques muestreados a
--eval-n filas por familia via sampling.py):
  1. Cada modelo no-supervisado se entrena SOLO con benigno de train.npz.
  2. Umbral GLOBAL de mejor F1 sobre el pool (benigno test + TODAS las
     familias de ataque muestreadas).
  3. Metricas combinadas: matriz de confusion + accuracy/precision/recall/
     especificidad/F1/MCC/AUC sobre ese mismo pool.
  4. Por familia: se agrupa esa familia con el MISMO benigno test
     compartido y se evalua con el MISMO umbral global (no un umbral por
     familia -- en despliegue real no se sabe de antemano que familia se
     esta viendo) -- da matriz de confusion + AUC por familia.
  5. Random Forest (supervisado): mismo protocolo de metricas, score =
     P(ataque) del clasificador, evaluado siempre sobre filas de ataque
     held-out (no vistas en su propio fit).

Las muestras de ataque por familia (features ya frequency-encoded) se
cachean en output/eval_cache/<familia>.npz para no repetir el paso caro de
muestrear+leer CSVs de varios GB si se vuelve a correr el script.

Uso:
    python3 baselines/classification_metrics.py --eval-n 200000
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from control_charts import common

from baselines import features, sampling
from baselines.run_baselines import build_unsupervised_models, load_processed, load_split_features
from baselines.sklearn_models import RandomForestBaseline
from eval_common import best_f1_threshold, binary_metrics, plot_confusion_matrix, roc_points, sanitize_scores

BASELINES_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASELINES_DIR / "output"
EVAL_CACHE_DIR = OUTPUT_DIR / "eval_cache"


def sample_attack_family_cached(folder, usecols, eval_n, rf_train_n, rng, metadata, vocab_maps, scaler, freq_tables, categorical_cols):
    """Cachea la muestra COMPLETA (hasta eval_n+rf_train_n filas) bajo la key
    'X_eval' por compatibilidad -- el split real entre evaluacion y
    entrenamiento del RF se hace siempre al leer, no al cachear, para poder
    ajustar --rf-train-n sin invalidar el cache."""
    target_n = eval_n + rf_train_n
    safe_name = folder.name.replace(" ", "_")
    cache_path = EVAL_CACHE_DIR / f"{safe_name}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        X_all, n_total_family = d["X_eval"], int(d["n_total_family"])
    else:
        csvs = common.attack_csvs(folder)
        dfs = []
        n_total_family = 0
        for path in csvs:
            df, n_total = sampling.sample_csv(path, usecols, target_n, rng)
            dfs.append(df)
            n_total_family += n_total
        df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=usecols)
        if len(df_all) > target_n:
            idx = rng.choice(len(df_all), size=target_n, replace=False)
            df_all = df_all.iloc[idx].reset_index(drop=True)

        cat_arrays, num_array, _ = common.transform_dataframe(df_all, metadata, vocab_maps, scaler)
        X_all = features.encode_features(cat_arrays, num_array, freq_tables, categorical_cols)

        EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X_eval=X_all, n_total_family=n_total_family)

    n_eval = min(eval_n, len(X_all))
    return X_all[:n_eval], X_all[n_eval:], n_total_family


def pooled_metrics_and_threshold(scores_benign, scores_by_family):
    y_true = np.concatenate([np.zeros(len(scores_benign), dtype=np.int8)] +
                             [np.ones(len(s), dtype=np.int8) for s in scores_by_family.values()])
    y_score = np.concatenate([scores_benign] + list(scores_by_family.values()))
    threshold = best_f1_threshold(y_true, y_score)
    y_pred = (y_score > threshold).astype(np.int8)
    metrics = binary_metrics(y_true, y_pred, y_score, "combined")
    metrics["threshold"] = threshold
    return metrics, y_true, y_score, threshold


def per_family_metrics(scores_benign, scores_by_family, threshold):
    rows = []
    benign_label = np.zeros(len(scores_benign), dtype=np.int8)
    for name, s in scores_by_family.items():
        y_family = np.concatenate([benign_label, np.ones(len(s), dtype=np.int8)])
        score_family = np.concatenate([scores_benign, s])
        y_pred = (score_family > threshold).astype(np.int8)
        m = binary_metrics(y_family, y_pred, score_family, name)
        rows.append({"attack": name, **{k: v for k, v in m.items() if k != "label"}})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-n", type=int, default=200_000,
                         help="Cap de filas de ataque muestreadas por familia para puntuar")
    parser.add_argument("--rf-train-n", type=int, default=5_000,
                         help="Filas de ataque por familia reservadas (held-out de --eval-n) para entrenar el Random Forest supervisado")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack", type=str, default=None, help="Solo (re)muestrea/cachea una familia (debug)")
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
    print(f"features: input_dim={input_dim}  X_train={X_train.shape}  X_test={X_test.shape}")

    with open(config.CHECKPOINT_DIR / "optuna_best_params.json") as f:
        best_params = json.load(f)["params"]
    hidden_dim, z_dim = best_params["hidden_dim"], best_params["z_dim"]

    models = build_unsupervised_models(input_dim, hidden_dim, z_dim, args.seed)
    scores_test = {}
    for name, model in models.items():
        t0 = time.time()
        print(f"[{name}] entrenando sobre benigno (train.npz, n={len(X_train)})...", flush=True)
        model.fit(X_train)
        scores_test[name] = sanitize_scores(model.score(X_test))
        print(f"[{name}] listo ({time.time() - t0:.0f}s)", flush=True)

    usecols = categorical_cols + numeric_cols + [metadata["timestamp_col"]]
    folders = common.attack_folders()
    if args.attack:
        folders = [f for f in folders if f.name == args.attack]
        if not folders:
            raise SystemExit(f"Familia '{args.attack}' no existe")

    scores_eval = {name: {} for name in models}
    rf_train_parts = []

    for folder in folders:
        name = folder.name
        t0 = time.time()
        X_eval, X_rf_train, n_total_family = sample_attack_family_cached(
            folder, usecols, args.eval_n, args.rf_train_n, rng, metadata, vocab_maps, scaler, freq_tables, categorical_cols)
        if len(X_eval) == 0:
            print(f"[{name}] sin filas, se omite")
            continue
        rf_train_parts.append(X_rf_train)

        for model_name, model in models.items():
            scores_eval[model_name][name] = sanitize_scores(model.score(X_eval))

        print(f"[{name}] n_total={n_total_family:,}  n_eval={len(X_eval):,}  ({time.time() - t0:.0f}s)", flush=True)

    if args.attack:
        print(f"Solo se (re)muestreo '{args.attack}'. Corre sin --attack para agregar todas las metricas.")
        return

    print("\n[RandomForest (supervised)] entrenando con benigno + muestra de ataque etiquetada...", flush=True)
    X_attack_train_rf = np.concatenate(rf_train_parts, axis=0) if rf_train_parts else np.empty((0, input_dim))
    rf = RandomForestBaseline(seed=args.seed)
    rf.fit(X_train, X_attack_train_rf)
    scores_test["RandomForest (supervised)"] = sanitize_scores(rf.score(X_test))
    scores_eval["RandomForest (supervised)"] = {}
    for folder in folders:
        # Reusa la MISMA porcion held-out (excluye las filas que entrenaron
        # el RF) que se uso para los demas modelos -- no el array completo.
        X_eval, _, _ = sample_attack_family_cached(
            folder, usecols, args.eval_n, args.rf_train_n, rng, metadata, vocab_maps, scaler, freq_tables, categorical_cols)
        if len(X_eval) == 0:
            continue
        scores_eval["RandomForest (supervised)"][folder.name] = sanitize_scores(rf.score(X_eval))
    print(f"[RandomForest (supervised)] entrenado con {len(X_attack_train_rf):,} filas de ataque held-out.")

    combined_rows = []
    per_family_rows = []
    roc_curves = {}
    method_names = list(models.keys()) + ["RandomForest (supervised)"]
    for method in method_names:
        metrics, y_true, y_score, threshold = pooled_metrics_and_threshold(scores_test[method], scores_eval[method])
        metrics["method"] = method
        combined_rows.append(metrics)

        fpr, tpr = roc_points(y_true, y_score)
        roc_curves[method] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": metrics["auc"]}

        for row in per_family_metrics(scores_test[method], scores_eval[method], threshold):
            row["method"] = method
            per_family_rows.append(row)

        cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        plot_confusion_matrix(cm, f"Confusion Matrix -- {method} (best-F1 threshold={threshold:.4f})",
                               OUTPUT_DIR / f"confusion_matrix_{method.replace(' ', '_').replace('(', '').replace(')', '')}.png")

        print(f"[{method}] threshold(best-F1)={threshold:.4f}  AUC={metrics['auc']:.4f}  "
              f"F1={metrics['f1']:.4f}  precision={metrics['precision']:.4f}  recall={metrics['recall_sensitivity']:.4f}")

    combined_df = pd.DataFrame(combined_rows).drop(columns=["label"])
    combined_path = OUTPUT_DIR / "classification_metrics_combined.csv"
    combined_df.to_csv(combined_path, index=False)

    per_family_df = pd.DataFrame(per_family_rows)
    per_family_path = OUTPUT_DIR / "classification_metrics_per_family.csv"
    per_family_df.to_csv(per_family_path, index=False)

    with open(OUTPUT_DIR / "roc_points_baselines.json", "w") as f:
        json.dump(roc_curves, f)

    print(f"\nSaved: {combined_path}")
    print(f"Saved: {per_family_path}")
    print(f"Saved: {OUTPUT_DIR / 'roc_points_baselines.json'}")
    print(f"Tiempo total: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
