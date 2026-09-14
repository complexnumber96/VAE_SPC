"""Compara el FlowVAE propuesto contra baselines establecidos de deteccion
de anomalias, en respuesta al comentario del revisor:

    "The manuscript lacks comparisons with established anomaly detection
    baselines such as Isolation Forest, One-Class SVM, standard
    Autoencoders, Deep SVDD, or conventional VAE-based anomaly detection
    approaches."

Metodologia (misma logica de fase1/fase2, ver control_charts/):
  1. Cada baseline no-supervisado (Isolation Forest, One-Class SVM,
     Autoencoder, Deep SVDD, VAE convencional) se entrena SOLO con trafico
     benigno de train.npz -- igual que el FlowVAE.
  2. El umbral (UCL) de cada uno se calibra sobre sus propios scores de
     train via el mismo bootstrap de bloques moviles que phase1.py
     (--quantile, default 0.95 -- igual que la calibracion actual del
     FlowVAE).
  3. Se reporta el falso-positivo sobre test.npz (benigno held-out) y el
     recall (% de filas marcadas anomalas) sobre cada familia de ataque.
  4. Ademas se agrega un Random Forest SUPERVISADO (no pedido por el
     revisor, sugerido como techo de referencia) -- entrenado con una
     muestra de ataque etiquetada, evaluado siempre sobre filas held-out.

Todos los baselines usan la misma representacion de features (frequency
encoding + scaler, ver features.py), DISTINTA de los embeddings aprendidos
por el FlowVAE, para que ningun metodo se beneficie de una representacion
entrenada por el modelo que se esta comparando.

Las 6 familias de ataque con millones de filas se muestrean (--eval-n,
default 200k por familia, aprox. uniforme sobre todo el archivo -- ver
sampling.py) en vez de puntuarse completas, para que el tiempo de computo
sea manejable; el porcentaje de anomalias del FlowVAE (phase2_summary.csv)
sigue siendo sobre la poblacion COMPLETA, se aclara en la tabla final.

Uso:
    python3 baselines/run_baselines.py --eval-n 200000 --quantile 0.95
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
from baselines.sklearn_models import (
    IsolationForestDetector, LOFDetector, OneClassSVMDetector, PCADetector, RandomForestBaseline,
)
from baselines.torch_models import AutoencoderDetector, DeepSVDDDetector, VanillaVAEDetector

BASELINES_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASELINES_DIR / "output"


def load_processed():
    with open(config.PROCESSED_DIR / "metadata.json") as f:
        metadata = json.load(f)
    with open(config.PROCESSED_DIR / "vocabs.pkl", "rb") as f:
        vocab_maps = pickle.load(f)
    with open(config.PROCESSED_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return metadata, vocab_maps, scaler


def load_split_features(name, categorical_cols, freq_tables):
    data = np.load(config.PROCESSED_DIR / f"{name}.npz")
    cat_arrays = {c: data[f"cat__{c}"] for c in categorical_cols}
    return features.encode_features(cat_arrays, data["numeric"], freq_tables, categorical_cols)


def build_unsupervised_models(input_dim, hidden_dim, z_dim, seed):
    return {
        "IsolationForest": IsolationForestDetector(seed=seed),
        "OneClassSVM": OneClassSVMDetector(),
        "LOF": LOFDetector(),
        "PCA": PCADetector(seed=seed),
        "Autoencoder": AutoencoderDetector(input_dim, hidden_dim, z_dim, seed=seed),
        "DeepSVDD": DeepSVDDDetector(input_dim, hidden_dim, z_dim, seed=seed),
        "VanillaVAE": VanillaVAEDetector(input_dim, hidden_dim, z_dim, seed=seed),
    }


def sample_attack_family(folder, usecols, target_n, rng):
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
    return df_all, n_total_family


def flowvae_reference_rows():
    """Numeros ya calculados del FlowVAE propuesto (control_charts/phase1.py
    + phase2.py), para incluir en la misma tabla comparativa. Se recomputa
    el falso-positivo sobre test.npz porque phase1.py no lo guarda por
    separado (solo train+test combinado)."""
    ref_path = common.PHASE1_STATS_PATH
    summary_path = common.OUTPUT_DIR / "phase2_summary.csv"
    if not ref_path.exists() or not summary_path.exists():
        print("  (no se encontraron resultados previos de phase1/phase2.py, se omite FlowVAE de la tabla)")
        return {}

    ref = np.load(ref_path)
    z_mean, cov_inv = ref["z_mean"], ref["cov_inv"]
    ucl_t2, ucl_spe = float(ref["ucl_t2"]), float(ref["ucl_spe"])

    vae, metadata, vocab_maps, scaler = common.load_artifacts()
    cat_test, num_test, _ = _load_split_raw("test", metadata)
    z_test, spe_test = common.encode_batch(vae, cat_test, num_test)
    t2_test = common.hotelling_t2(z_test, z_mean, cov_inv)
    fpr_t2 = 100 * (t2_test > ucl_t2).mean()
    fpr_spe = 100 * (spe_test > ucl_spe).mean()

    summary = pd.read_csv(summary_path)
    rows_t2 = [{"attack": "(benign test)", "n_rows_sampled": len(spe_test), "pct_anom": fpr_t2}]
    rows_spe = [{"attack": "(benign test)", "n_rows_sampled": len(spe_test), "pct_anom": fpr_spe}]
    for _, row in summary.iterrows():
        rows_t2.append({"attack": row["attack"], "n_rows_sampled": int(row["n_rows"]),
                         "n_rows_total": int(row["n_rows"]), "pct_anom": row["pct_anom_t2"],
                         "poblacion_completa": True})
        rows_spe.append({"attack": row["attack"], "n_rows_sampled": int(row["n_rows"]),
                          "n_rows_total": int(row["n_rows"]), "pct_anom": row["pct_anom_spe"],
                          "poblacion_completa": True})
    return {"FlowVAE_T2 (proposed)": rows_t2, "FlowVAE_SPE (proposed)": rows_spe}


def _load_split_raw(name, metadata):
    data = np.load(config.PROCESSED_DIR / f"{name}.npz")
    categorical_cols = metadata["categorical_cols"]
    cat_arrays = {c: data[f"cat__{c}"] for c in categorical_cols}
    return cat_arrays, data["numeric"], None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--eval-n", type=int, default=200_000,
                         help="Cap de filas de ataque muestreadas por familia para puntuar")
    parser.add_argument("--rf-train-n", type=int, default=5_000,
                         help="Filas de ataque por familia reservadas (held-out de --eval-n) para entrenar el Random Forest supervisado")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack", type=str, default=None, help="Correr solo una familia (debug)")
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
    print(f"features: frequency-encoded cat ({len(categorical_cols)}) + numeric ({len(numeric_cols)}) "
          f"= input_dim {input_dim}. X_train={X_train.shape}, X_test={X_test.shape}")

    with open(config.CHECKPOINT_DIR / "optuna_best_params.json") as f:
        best_params = json.load(f)["params"]
    hidden_dim, z_dim = best_params["hidden_dim"], best_params["z_dim"]
    print(f"capacidad equivalente al FlowVAE: hidden_dim={hidden_dim}, z_dim={z_dim}")

    models = build_unsupervised_models(input_dim, hidden_dim, z_dim, args.seed)
    ucls = {}
    results = {name: [] for name in models}

    for name, model in models.items():
        t0 = time.time()
        print(f"[{name}] entrenando sobre benigno (train.npz, n={len(X_train)})...", flush=True)
        model.fit(X_train)
        train_scores = model.score(X_train)
        block_length = common.default_block_length(len(train_scores))
        ucl, _ = common.moving_block_bootstrap_ucl(train_scores, block_length, args.n_boot, args.quantile, rng)
        ucls[name] = ucl
        fpr = 100 * (model.score(X_test) > ucl).mean()
        results[name].append({"attack": "(benign test)", "n_rows_sampled": len(X_test), "pct_anom": fpr})
        print(f"[{name}] UCL={ucl:.4f}  falso-positivo benigno={fpr:.2f}%  ({time.time() - t0:.0f}s)", flush=True)

    usecols = categorical_cols + numeric_cols + [metadata["timestamp_col"]]
    folders = common.attack_folders()
    if args.attack:
        folders = [f for f in folders if f.name == args.attack]
        if not folders:
            raise SystemExit(f"Familia '{args.attack}' no existe")

    eval_cache = {}
    rf_train_parts = []
    target_n = args.eval_n + args.rf_train_n

    for folder in folders:
        name = folder.name
        t0 = time.time()
        df_all, n_total_family = sample_attack_family(folder, usecols, target_n, rng)
        if len(df_all) == 0:
            print(f"[{name}] sin filas, se omite")
            continue

        cat_arrays, num_array, _ = common.transform_dataframe(df_all, metadata, vocab_maps, scaler)
        X_all = features.encode_features(cat_arrays, num_array, freq_tables, categorical_cols)

        n_eval = min(args.eval_n, len(X_all))
        X_eval, X_rf_train = X_all[:n_eval], X_all[n_eval:]
        eval_cache[name] = (X_eval, n_total_family)
        rf_train_parts.append(X_rf_train)

        for model_name, model in models.items():
            scores = model.score(X_eval)
            pct = 100 * (scores > ucls[model_name]).mean()
            results[model_name].append({"attack": name, "n_rows_sampled": len(X_eval),
                                         "n_rows_total": n_total_family, "pct_anom": pct})

        print(f"[{name}] n_total={n_total_family:,}  n_eval={len(X_eval):,}  "
              f"n_rf_train={len(X_rf_train):,}  ({time.time() - t0:.0f}s)", flush=True)

    print("\n[RandomForest (supervised)] entrenando con benigno + muestra de ataque etiquetada...", flush=True)
    X_attack_train_rf = np.concatenate(rf_train_parts, axis=0) if rf_train_parts else np.empty((0, input_dim))
    rf = RandomForestBaseline(seed=args.seed)
    rf.fit(X_train, X_attack_train_rf)
    rf_fpr = 100 * rf.predict_anomaly(X_test).mean()
    rf_rows = [{"attack": "(benign test)", "n_rows_sampled": len(X_test), "pct_anom": rf_fpr}]
    for name, (X_eval, n_total_family) in eval_cache.items():
        pct = 100 * rf.predict_anomaly(X_eval).mean()
        rf_rows.append({"attack": name, "n_rows_sampled": len(X_eval),
                         "n_rows_total": n_total_family, "pct_anom": pct})
    results["RandomForest (supervised)"] = rf_rows
    print(f"[RandomForest (supervised)] entrenado con {len(X_attack_train_rf):,} filas de ataque "
          f"(held-out de la evaluacion). falso-positivo benigno={rf_fpr:.2f}%")

    print("\nAgregando resultados de referencia del FlowVAE (control_charts/)...")
    results.update(flowvae_reference_rows())

    long_rows = []
    for method, rows in results.items():
        for row in rows:
            long_rows.append({"method": method, **row})
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUTPUT_DIR / "baselines_comparison_long.csv", index=False)

    wide_df = long_df.pivot_table(index="attack", columns="method", values="pct_anom", aggfunc="first")
    attack_order = ["(benign test)"] + [f.name for f in folders]
    wide_df = wide_df.reindex([a for a in attack_order if a in wide_df.index])
    wide_df.to_csv(OUTPUT_DIR / "baselines_comparison_wide.csv")

    print(f"\nTabla comparativa (%% filas marcadas anomalas, benigno = falso-positivo):\n")
    print(wide_df.round(2).to_string())
    print(f"\nGuardado en {OUTPUT_DIR / 'baselines_comparison_long.csv'} y "
          f"{OUTPUT_DIR / 'baselines_comparison_wide.csv'}")
    print(f"Tiempo total: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
