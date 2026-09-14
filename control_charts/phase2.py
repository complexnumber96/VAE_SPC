"""Fase 2: evalua cada ataque (una carpeta = una familia de malware, CSVs de
nfstream bajo Capturas_malignas/Malware/capturas_csv/<ataque>/*.csv) contra
los limites de control (UCL de T2 y SPE) calculados en fase 1 sobre trafico
benigno (ver phase1.py / output/phase1_reference.npz).

Procesa cada CSV en chunks (no carga el archivo completo en RAM -- algunos
pesan varios GB) usando exactamente el mismo pipeline de columnas/vocabulario
/scaler que preprocess.py, y corre cada chunk por el encoder del VAE para
obtener T2 y SPE. Un punto se marca en rojo si supera el UCL de esa carta.

El eje x de cada carta es un id de ocurrencia (1..n, orden = timestamp), no
el timestamp crudo. Graficos y mensajes de consola en ingles.

Para archivos con millones de filas, el grafico grafica como maximo
`--max-plot-points` puntos (todas las anomalias + una muestra aleatoria de
los puntos en control) mediante submuestreo -- el conteo/porcentaje de
anomalias que se reporta siempre es sobre el total de filas, no sobre la
muestra graficada.
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common

CHUNK_SIZE = 200_000
MAX_PLOT_POINTS = 200_000


def score_csv(path, vae, metadata, vocab_maps, scaler, z_mean, cov_inv, usecols):
    ts_chunks, t2_chunks, spe_chunks = [], [], []
    n_rows = 0
    t0 = time.time()
    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False)):
        cat_arrays, num_array, ts_array = common.transform_dataframe(chunk, metadata, vocab_maps, scaler)
        z, spe = common.encode_batch(vae, cat_arrays, num_array)
        t2 = common.hotelling_t2(z, z_mean, cov_inv)

        ts_chunks.append(ts_array)
        t2_chunks.append(t2)
        spe_chunks.append(spe)
        n_rows += len(chunk)
        print(f"    {path.name}: chunk {i + 1}, {n_rows:,} rows ({time.time() - t0:.0f}s)", flush=True)

    return np.concatenate(ts_chunks), np.concatenate(t2_chunks), np.concatenate(spe_chunks)


def downsample_for_plot(anomaly, rng, max_points):
    n = len(anomaly)
    if n <= max_points:
        return np.arange(n)
    anom_idx = np.flatnonzero(anomaly)
    budget_normal = max(0, max_points - len(anom_idx))
    normal_idx = np.flatnonzero(~anomaly)
    if len(normal_idx) > budget_normal:
        normal_idx = rng.choice(normal_idx, size=budget_normal, replace=False)
    idx = np.concatenate([anom_idx, normal_idx])
    idx.sort()
    return idx


def plot_chart(ts, stat, ucl, title, ylabel, out_path, rng, max_points):
    """x-axis is the occurrence id (1..n) after sorting by timestamp, not
    the raw timestamp itself. `ts`/`stat` must already be sorted by
    timestamp when passed in."""
    anomaly = stat > ucl
    occurrence_id = np.arange(1, len(stat) + 1)
    idx = downsample_for_plot(anomaly, rng, max_points)
    occ_p = occurrence_id[idx]
    stat_p = stat[idx]
    anomaly_p = anomaly[idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(occ_p[~anomaly_p], stat_p[~anomaly_p], s=6, color="#3366cc", label="In control", alpha=0.5, rasterized=True)
    if anomaly_p.any():
        ax.scatter(occ_p[anomaly_p], stat_p[anomaly_p], s=8, color="#d62728", label="Anomaly", alpha=0.8, zorder=3, rasterized=True)
    ax.axhline(ucl, color="#d62728", linestyle="--", linewidth=1.5, label=f"UCL (phase 1) = {ucl:.2f}")

    n_anom = int(anomaly.sum())
    subtitle = f"{n_anom:,}/{len(stat):,} anomalies ({100 * n_anom / len(stat):.2f}%)"
    if len(idx) < len(stat):
        subtitle += f"  -- chart subsampled to {len(idx):,} points"
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("Occurrence ID (ordered by timestamp)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return n_anom


def process_attack(folder, vae, metadata, vocab_maps, scaler, z_mean, cov_inv, ucl_t2, ucl_spe, rng, max_plot_points):
    name = folder.name
    csvs = common.attack_csvs(folder)
    if not csvs:
        print(f"[{name}] no top-level CSVs found, skipping")
        return None

    usecols = [c for c in (metadata["categorical_cols"] + metadata["numeric_cols"] + [metadata["timestamp_col"]])]

    print(f"[{name}] {len(csvs)} csv(s): {[c.name for c in csvs]}")
    ts_all, t2_all, spe_all = [], [], []
    for path in csvs:
        ts, t2, spe = score_csv(path, vae, metadata, vocab_maps, scaler, z_mean, cov_inv, usecols)
        ts_all.append(ts)
        t2_all.append(t2)
        spe_all.append(spe)

    ts = np.concatenate(ts_all)
    t2 = np.concatenate(t2_all)
    spe = np.concatenate(spe_all)
    order = np.argsort(ts, kind="stable")
    ts, t2, spe = ts[order], t2[order], spe[order]

    safe_name = name.replace(" ", "_")
    n_anom_t2 = plot_chart(
        ts, t2, ucl_t2,
        f"Phase 2 -- {name} -- Hotelling's T2 Control Chart",
        "T2",
        common.OUTPUT_DIR / f"phase2_{safe_name}_T2.png",
        rng, max_plot_points,
    )
    n_anom_spe = plot_chart(
        ts, spe, ucl_spe,
        f"Phase 2 -- {name} -- SPE Control Chart",
        "SPE (reconstruction error)",
        common.OUTPUT_DIR / f"phase2_{safe_name}_SPE.png",
        rng, max_plot_points,
    )

    n = len(ts)
    print(f"[{name}] n={n:,}  T2 out of control={n_anom_t2:,} ({100*n_anom_t2/n:.2f}%)  "
          f"SPE out of control={n_anom_spe:,} ({100*n_anom_spe/n:.2f}%)")

    return {
        "attack": name,
        "n_rows": n,
        "n_files": len(csvs),
        "n_anom_t2": n_anom_t2,
        "pct_anom_t2": 100 * n_anom_t2 / n,
        "n_anom_spe": n_anom_spe,
        "pct_anom_spe": 100 * n_anom_spe / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, default=None, help="Exact name of a single attack folder")
    parser.add_argument("--max-plot-points", type=int, default=MAX_PLOT_POINTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = np.load(common.PHASE1_STATS_PATH)
    z_mean, cov_inv = ref["z_mean"], ref["cov_inv"]
    ucl_t2, ucl_spe = float(ref["ucl_t2"]), float(ref["ucl_spe"])
    print(f"Phase 1 reference: UCL_T2={ucl_t2:.4f}  UCL_SPE={ucl_spe:.4f}")

    vae, metadata, vocab_maps, scaler = common.load_artifacts()
    rng = np.random.default_rng(args.seed)

    folders = common.attack_folders()
    if args.attack:
        folders = [f for f in folders if f.name == args.attack]
        if not folders:
            raise SystemExit(f"Attack folder '{args.attack}' does not exist")

    summary_path = common.OUTPUT_DIR / "phase2_summary.csv"
    write_header = not summary_path.exists() or args.attack is None
    mode = "w" if write_header else "a"
    rows = []
    with open(summary_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "n_rows", "n_files", "n_anom_t2", "pct_anom_t2", "n_anom_spe", "pct_anom_spe"])
        if write_header:
            writer.writeheader()
        for folder in folders:
            result = process_attack(folder, vae, metadata, vocab_maps, scaler, z_mean, cov_inv, ucl_t2, ucl_spe, rng, args.max_plot_points)
            if result:
                writer.writerow(result)
                f.flush()
                rows.append(result)

    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
