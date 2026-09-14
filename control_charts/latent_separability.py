"""Metricas cuantitativas de separabilidad del espacio latente del FlowVAE
(benigno vs. cada familia de ataque), en respuesta al comentario del revisor 2:

    "The latent-space analysis relies mainly on t-SNE visualizations. Since
    t-SNE is primarily an exploratory visualization technique and may distort
    distances and cluster relationships, additional quantitative measures of
    latent-space separability would strengthen the conclusions."

Para cada familia se calcula, sobre z = E[q(z|x)] (el mismo z que usan T2/SPE):
  - silhouette score (sklearn): [-1, 1], mayor = clusters mas separados.
  - Davies-Bouldin index (sklearn): >= 0, MENOR = clusters mas separados.
  - distancia de Mahalanobis centroide-a-centroide (usa la misma covarianza
    Ledoit-Wolf de fase 1, common.fit_t2_reference) -- conecta directamente
    con el T2 ya reportado en phase2_summary.csv.

Esto tambien aporta evidencia cuantitativa para el comentario mayor del
revisor 1 ("Analysis of Hard-to-Detect Attacks", Hide and Seek / IRCBot): las
familias con silhouette bajo / Davies-Bouldin alto son las que el espacio
latente no separa bien del benigno.

Benigno: z de test.npz (held-out, no participo en la calibracion de fase 1).
Ataque: muestra moderada por familia (--sample-n, default 8000 filas,
muestreo aprox. uniforme sobre todo el archivo via baselines/sampling.py) --
no hace falta la poblacion completa para una estimacion de silhouette
confiable, y evita re-leer CSVs de varios GB fila por fila.

Uso:
    python3 control_charts/latent_separability.py --sample-n 8000
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

import common

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines import sampling


def load_benign_test_z(vae, metadata, vocab_maps, scaler):
    import config
    data = np.load(config.PROCESSED_DIR / "test.npz")
    categorical_cols = metadata["categorical_cols"]
    cat_arrays = {c: data[f"cat__{c}"] for c in categorical_cols}
    z, _ = common.encode_batch(vae, cat_arrays, data["numeric"])
    return z


def sample_family_z(folder, vae, metadata, vocab_maps, scaler, sample_n, rng):
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
    z, _ = common.encode_batch(vae, cat_arrays, num_array)
    return z, n_total


def mahalanobis_centroid_distance(z_benign, z_attack):
    """Distancia de Mahalanobis entre los centroides de benigno y ataque,
    usando la covarianza (Ledoit-Wolf) del pool de ambos grupos."""
    z_all = np.concatenate([z_benign, z_attack], axis=0)
    _, cov_inv = common.fit_t2_reference(z_all)
    diff = z_attack.mean(axis=0) - z_benign.mean(axis=0)
    return float(np.sqrt(diff @ cov_inv @ diff))


def plot_bars(df, out_path):
    df_sorted = df.sort_values("silhouette")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].barh(df_sorted["attack"], df_sorted["silhouette"], color="#3366cc")
    axes[0].axvline(0, color="#555555", linewidth=1)
    axes[0].set_xlabel("Silhouette score (higher = better separated)")
    axes[0].set_title("Latent-space separability -- silhouette")

    df_sorted2 = df.sort_values("davies_bouldin", ascending=False)
    axes[1].barh(df_sorted2["attack"], df_sorted2["davies_bouldin"], color="#d62728")
    axes[1].set_xlabel("Davies-Bouldin index (lower = better separated)")
    axes[1].set_title("Latent-space separability -- Davies-Bouldin")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-n", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack", type=str, default=None)
    args = parser.parse_args()

    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    vae, metadata, vocab_maps, scaler = common.load_artifacts()
    z_benign = load_benign_test_z(vae, metadata, vocab_maps, scaler)
    print(f"Benign test z: n={len(z_benign)}, z_dim={z_benign.shape[1]}")

    folders = common.attack_folders()
    if args.attack:
        folders = [f for f in folders if f.name == args.attack]
        if not folders:
            raise SystemExit(f"Attack folder '{args.attack}' does not exist")

    rows = []
    for folder in folders:
        t0 = time.time()
        z_attack, n_total = sample_family_z(folder, vae, metadata, vocab_maps, scaler, args.sample_n, rng)
        if z_attack is None or len(z_attack) == 0:
            print(f"[{folder.name}] sin filas, se omite")
            continue

        z_combined = np.concatenate([z_benign, z_attack], axis=0)
        labels = np.concatenate([np.zeros(len(z_benign)), np.ones(len(z_attack))])

        sil = silhouette_score(z_combined, labels)
        db = davies_bouldin_score(z_combined, labels)
        maha = mahalanobis_centroid_distance(z_benign, z_attack)

        rows.append({
            "attack": folder.name, "n_sampled": len(z_attack), "n_total": n_total,
            "silhouette": sil, "davies_bouldin": db, "centroid_mahalanobis": maha,
        })
        print(f"[{folder.name}] n_sampled={len(z_attack):,}/{n_total:,}  "
              f"silhouette={sil:.4f}  davies_bouldin={db:.4f}  "
              f"centroid_mahalanobis={maha:.2f}  ({time.time() - t0:.0f}s)")

    if not rows:
        print("No se proceso ninguna familia.")
        return

    df = pd.DataFrame(rows).sort_values("silhouette")
    out_csv = common.OUTPUT_DIR / "latent_separability.csv"
    df.to_csv(out_csv, index=False)
    plot_bars(df, common.OUTPUT_DIR / "latent_separability.png")

    print(f"\nSaved: {out_csv}")
    print(f"Chart: {common.OUTPUT_DIR / 'latent_separability.png'}")
    print("\n(silhouette mas bajo / davies_bouldin mas alto = familia peor separada del benigno)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
