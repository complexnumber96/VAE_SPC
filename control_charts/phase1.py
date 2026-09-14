"""Fase 1: cartas de control T2 de Hotelling y SPE sobre trafico BENIGNO.

La referencia T2 (media/covarianza de z) y los UCL de ambas cartas se
calibran SOLO sobre processed/train.npz (no train+test combinado). Motivo
(pedido explicito del usuario, 2026-07-19): calibrar con train+test dejaba
que un puñado de outliers extremos del split de test (T2 hasta ~4.2M, vs.
un maximo de ~48 en train -- ver memoria del proyecto) infle la covarianza y
el UCL, dejando limites demasiado altos para detectar trafico malicioso real
(muchos falsos negativos en fase 2). Calibrar solo con train da limites mucho
mas ajustados.

El grafico de fase 1 sigue mostrando la serie COMPLETA (train+test, ordenada
por timestamp) contra esos limites calibrados-en-train, con una linea
vertical marcando donde termina train y empieza el test held-out -- asi se
ve explicitamente cuanto del test (que no participo en la calibracion)
queda fuera de control bajo el limite mas estricto.

Pipeline:
  1. Corre el encoder del mejor VAE (checkpoints/vae_best.pt) -> z_loc, SPE
     sobre train.
  2. Ajusta la referencia T2 (media/covarianza, Ledoit-Wolf) sobre z de train.
  3. Calcula el UCL de cada carta via bootstrap de bloques moviles sobre la
     serie de train (preserva autocorrelacion temporal).
  4. Grafica ambas cartas sobre train+test completo (eje x = id de
     ocurrencia, orden = timestamp; linea horizontal = UCL, rojo = fuera de
     control) y guarda la referencia (z_mean, cov_inv, UCLs) para que fase 2
     la reutilice sobre el trafico de ataque. Graficos y consola en ingles.

ADVERTENCIA (ver memoria del proyecto): la fase 1 tiene solo ~1781 filas de
train benigno (muestra chica seleccionada a proposito por el usuario, no el
corpus completo IoT-23). Los limites de control bootstrap son la mejor
estimacion posible con esos datos, pero con una fase 1 tan chica hay que
leerlos con cautela -- no son un limite de proceso "industrial" con miles de
observaciones de referencia.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common
import config


def load_split(name):
    data = np.load(config.PROCESSED_DIR / f"{name}.npz")
    ts = np.load(config.PROCESSED_DIR / f"{name}_timestamp.npy")
    with open(config.PROCESSED_DIR / "metadata.json") as f:
        categorical_cols = json.load(f)["categorical_cols"]
    cat_arrays = {c: data[f"cat__{c}"] for c in categorical_cols}
    return cat_arrays, data["numeric"], ts


def load_combined_series():
    """train+test combinado y ordenado por timestamp -- solo para graficar
    (la calibracion de referencia/UCL usa unicamente train, ver load_split)."""
    cat_train, num_train, ts_train = load_split("train")
    cat_test, num_test, ts_test = load_split("test")

    cat_arrays = {c: np.concatenate([cat_train[c], cat_test[c]]) for c in cat_train}
    num_array = np.concatenate([num_train, num_test])
    ts_array = np.concatenate([ts_train, ts_test])

    order = np.argsort(ts_array, kind="stable")
    ts_array = ts_array[order]
    num_array = num_array[order]
    for c in cat_arrays:
        cat_arrays[c] = cat_arrays[c][order]

    return cat_arrays, num_array, ts_array, len(ts_train)


def plot_chart(ts, stat, ucl, title, ylabel, out_path, train_boundary=None):
    """x-axis is the occurrence id (1..n) after sorting by timestamp, not
    the raw timestamp itself."""
    order = np.argsort(ts, kind="stable")
    stat = stat[order]
    occurrence_id = np.arange(1, len(stat) + 1)
    anomaly = stat > ucl

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(occurrence_id[~anomaly], stat[~anomaly], s=10, color="#3366cc", label="In control", alpha=0.8)
    if anomaly.any():
        ax.scatter(occurrence_id[anomaly], stat[anomaly], s=14, color="#d62728", label="Out of control", alpha=0.9, zorder=3)
    ax.axhline(ucl, color="#d62728", linestyle="--", linewidth=1.5, label=f"UCL = {ucl:.2f}")
    if train_boundary is not None:
        ax.axvline(train_boundary + 0.5, color="#555555", linestyle=":", linewidth=1.5,
                    label="End of train (UCL calibration) / start of held-out test")

    ax.set_title(title)
    ax.set_xlabel("Occurrence ID (ordered by timestamp)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return int(anomaly.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vae, metadata, vocab_maps, scaler = common.load_artifacts()

    cat_train, num_train, ts_train = load_split("train")
    n_train = num_train.shape[0]
    block_length = args.block_length or common.default_block_length(n_train)
    rng = np.random.default_rng(args.seed)

    print(f"Phase 1 calibration: {n_train} rows of benign TRAIN traffic only, "
          f"block_length={block_length}, n_boot={args.n_boot}, quantile={args.quantile}")

    z_train, spe_train = common.encode_batch(vae, cat_train, num_train)
    z_mean, cov_inv = common.fit_t2_reference(z_train)
    t2_train = common.hotelling_t2(z_train, z_mean, cov_inv)

    ucl_t2, _ = common.moving_block_bootstrap_ucl(t2_train, block_length, args.n_boot, args.quantile, rng)
    ucl_spe, _ = common.moving_block_bootstrap_ucl(spe_train, block_length, args.n_boot, args.quantile, rng)

    # Grafico de diagnostico: train+test completo contra los limites
    # calibrados solo con train (muestra cuanto del held-out test queda
    # fuera de control bajo el limite mas estricto).
    cat_full, num_full, ts_full, n_train_sorted = load_combined_series()
    z_full, spe_full = common.encode_batch(vae, cat_full, num_full)
    t2_full = common.hotelling_t2(z_full, z_mean, cov_inv)

    n_oc_t2 = plot_chart(
        ts_full, t2_full, ucl_t2,
        "Phase 1 (benign) -- Hotelling's T2 Control Chart (limits calibrated on train only)",
        "T2",
        common.OUTPUT_DIR / "phase1_T2.png",
        train_boundary=n_train_sorted,
    )
    n_oc_spe = plot_chart(
        ts_full, spe_full, ucl_spe,
        "Phase 1 (benign) -- SPE Control Chart (limits calibrated on train only)",
        "SPE (reconstruction error)",
        common.OUTPUT_DIR / "phase1_SPE.png",
        train_boundary=n_train_sorted,
    )

    np.savez(
        common.PHASE1_STATS_PATH,
        z_mean=z_mean, cov_inv=cov_inv,
        ucl_t2=ucl_t2, ucl_spe=ucl_spe,
        block_length=block_length, n_boot=args.n_boot, quantile=args.quantile,
        n_phase1=n_train,
    )

    n_full = num_full.shape[0]
    print(f"UCL T2  = {ucl_t2:.4f}  ({n_oc_t2}/{n_full} out of control over train+test, "
          f"{100 * n_oc_t2 / n_full:.2f}%)")
    print(f"UCL SPE = {ucl_spe:.4f}  ({n_oc_spe}/{n_full} out of control over train+test, "
          f"{100 * n_oc_spe / n_full:.2f}%)")
    print(f"Reference saved to {common.PHASE1_STATS_PATH}")
    print(f"Charts: {common.OUTPUT_DIR / 'phase1_T2.png'}, {common.OUTPUT_DIR / 'phase1_SPE.png'}")


if __name__ == "__main__":
    main()
