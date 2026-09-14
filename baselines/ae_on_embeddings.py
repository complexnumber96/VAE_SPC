"""Autoencoder ESTANDAR (determinista, sin KL) entrenado sobre los EMBEDDINGS
YA APRENDIDOS del FlowVAE propuesto -- la celda que falta para poder aislar,
por separado, el aporte de la arquitectura (variational vs. determinista) del
aporte de la representacion (embeddings aprendidos vs. frequency-encoding),
en respuesta al comentario del revisor 2:

    "An ablation study is missing. In particular, the contribution of each
    component (VAE, embeddings, and SPC monitoring) should be quantified
    separately. It remains unclear whether the reported performance gains
    originate from the VAE architecture itself, the embedding representation,
    or the SPC-based monitoring stage."

Con esto, las 4 celdas del 2x2 {representacion} x {arquitectura} quedan
disponibles, todas calibradas con el MISMO protocolo (bootstrap de bloques
moviles, quantile=0.95, solo benigno de train):

                        embeddings (aprendidos)     frequency-encoding
    variational         FlowVAE (control_charts/)   VanillaVAE (run_baselines.py)
    determinista         ESTE SCRIPT                AE estandar (ae_control_chart.py)

Metodologia: se usa el EMBEDDER YA ENTRENADO del FlowVAE (congelado, sin
fine-tuning) para producir el vector de entrada -- exactamente el mismo
`vae.assemble(x_cat, x_num)` que usa el FlowVAE -- y encima se entrena un
`baselines.torch_models.AutoencoderDetector` (la MISMA clase que usan los
demas baselines, sin cambios) con MSE puro, sin muestreo ni termino KL. Asi
la unica diferencia contra el FlowVAE es la arquitectura (determinista vs.
variational); la representacion de entrada es identica.

Uso:
    python3 baselines/ae_on_embeddings.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from control_charts import common

from baselines import sampling
from baselines.torch_models import AutoencoderDetector
from eval_common import binary_metrics, plot_confusion_matrix, roc_points

BASELINES_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASELINES_DIR / "output"
EVAL_CACHE_DIR = OUTPUT_DIR / "eval_cache"


@torch.no_grad()
def assemble_batch(vae, cat_arrays, num_array, batch_size=65536):
    """Vector plano embeddings+numericas (vae.assemble), por lotes -- analogo
    a common.encode_batch pero devolviendo x en vez de z/SPE."""
    n = num_array.shape[0]
    out = np.empty((n, vae.input_dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_cat = {c: torch.from_numpy(cat_arrays[c][start:end]).long() for c in vae.categorical_cols}
        x_num = torch.from_numpy(num_array[start:end])
        out[start:end] = vae.assemble(x_cat, x_num).cpu().numpy()
    return out


def load_split_embeddings(name, vae, metadata):
    data = np.load(config.PROCESSED_DIR / f"{name}.npz")
    categorical_cols = metadata["categorical_cols"]
    cat_arrays = {c: data[f"cat__{c}"] for c in categorical_cols}
    return assemble_batch(vae, cat_arrays, data["numeric"])


def sample_family_embeddings(folder, vae, metadata, vocab_maps, scaler, sample_n, rng):
    usecols = metadata["categorical_cols"] + metadata["numeric_cols"] + [metadata["timestamp_col"]]
    dfs, n_total = [], 0
    for path in common.attack_csvs(folder):
        df, n = sampling.sample_csv(path, usecols, sample_n, rng)
        dfs.append(df)
        n_total += n
    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=usecols)
    if len(df_all) > sample_n:
        idx = rng.choice(len(df_all), size=sample_n, replace=False)
        df_all = df_all.iloc[idx].reset_index(drop=True)
    if len(df_all) == 0:
        return None, n_total
    cat_arrays, num_array, _ = common.transform_dataframe(df_all, metadata, vocab_maps, scaler)
    return assemble_batch(vae, cat_arrays, num_array), n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--sample-n", type=int, default=8_000,
                         help="Filas de ataque muestreadas por familia (igual criterio que latent_separability.py)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack", type=str, default=None, help="Correr solo una familia (debug)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    vae, metadata, vocab_maps, scaler = common.load_artifacts()

    X_train = load_split_embeddings("train", vae, metadata)
    X_test = load_split_embeddings("test", vae, metadata)
    input_dim = X_train.shape[1]

    with open(config.CHECKPOINT_DIR / "optuna_best_params.json") as f:
        best_params = json.load(f)["params"]
    hidden_dim, z_dim = best_params["hidden_dim"], best_params["z_dim"]

    print(f"Entrenando AE determinista sobre EMBEDDINGS del FlowVAE (train.npz, n={len(X_train)}), "
          f"input_dim={input_dim}, hidden_dim={hidden_dim}, z_dim={z_dim}...", flush=True)
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
    if args.attack:
        folders = [f for f in folders if f.name == args.attack]
        if not folders:
            raise SystemExit(f"Attack folder '{args.attack}' does not exist")
    per_attack_rows = []
    t2_parts, spe_parts, label_parts = [t2_benign], [spe_benign], [benign_label]

    for folder in folders:
        X_eval, n_total = sample_family_embeddings(folder, vae, metadata, vocab_maps, scaler, args.sample_n, rng)
        if X_eval is None or len(X_eval) == 0:
            print(f"[{folder.name}] sin filas, se omite")
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
        row = {"attack": folder.name, "n_rows": len(t2_a), "n_total_family": n_total}
        row.update({f"{k}_t2": v for k, v in m_t2.items() if k != "label"})
        row.update({f"{k}_spe": v for k, v in m_spe.items() if k != "label"})
        per_attack_rows.append(row)
        print(f"[{folder.name}] n={len(t2_a):,}/{n_total:,}  recall_T2={m_t2['recall_sensitivity']:.3f}  "
              f"recall_SPE={m_spe['recall_sensitivity']:.3f}")

    y_true = np.concatenate(label_parts)
    t2_all = np.concatenate(t2_parts)
    spe_all = np.concatenate(spe_parts)

    metrics_t2 = binary_metrics(y_true, (t2_all > ucl_t2).astype(np.int8), t2_all, "AE_embeddings_T2")
    metrics_spe = binary_metrics(y_true, (spe_all > ucl_spe).astype(np.int8), spe_all, "AE_embeddings_SPE")

    fpr_t2, tpr_t2 = roc_points(y_true, t2_all)
    fpr_spe, tpr_spe = roc_points(y_true, spe_all)

    cm_t2 = np.array([[metrics_t2["tn"], metrics_t2["fp"]], [metrics_t2["fn"], metrics_t2["tp"]]])
    cm_spe = np.array([[metrics_spe["tn"], metrics_spe["fp"]], [metrics_spe["fn"], metrics_spe["tp"]]])
    plot_confusion_matrix(cm_t2, f"Confusion Matrix -- AE on FlowVAE embeddings, T2 (UCL={ucl_t2:.2f})",
                           OUTPUT_DIR / "confusion_matrix_AE_embeddings_T2.png")
    plot_confusion_matrix(cm_spe, f"Confusion Matrix -- AE on FlowVAE embeddings, SPE (UCL={ucl_spe:.2f})",
                           OUTPUT_DIR / "confusion_matrix_AE_embeddings_SPE.png")

    combined_df = pd.DataFrame([metrics_t2, metrics_spe]).rename(columns={"label": "chart"})
    combined_df.to_csv(OUTPUT_DIR / "ae_embeddings_classification_metrics_combined.csv", index=False)
    pd.DataFrame(per_attack_rows).to_csv(OUTPUT_DIR / "ae_embeddings_classification_metrics_per_family.csv", index=False)

    with open(OUTPUT_DIR / "roc_points_ae_embeddings.json", "w") as f:
        json.dump({
            "t2": {"fpr": fpr_t2.tolist(), "tpr": tpr_t2.tolist(), "auc": metrics_t2["auc"]},
            "spe": {"fpr": fpr_spe.tolist(), "tpr": tpr_spe.tolist(), "auc": metrics_spe["auc"]},
        }, f)

    print("\n=== AE on embeddings -- T2 (combined) ===")
    print(combined_df[combined_df.chart == "AE_embeddings_T2"].to_string(index=False))
    print("\n=== AE on embeddings -- SPE (combined) ===")
    print(combined_df[combined_df.chart == "AE_embeddings_SPE"].to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'ae_embeddings_classification_metrics_combined.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'ae_embeddings_classification_metrics_per_family.csv'}")
    print(f"Tiempo total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
