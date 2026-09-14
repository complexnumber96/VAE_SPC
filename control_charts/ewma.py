"""Carta de control EWMA (Exponentially Weighted Moving Average) sobre T2/SPE,
POR FLUJO -- en respuesta a dos comentarios de los revisores sobre los
"bloques de 30 minutos no solapados" que describe el paper:

  - Revisor 1, "Time Window Practicality": "a 30-minute delay before an
    out-of-control alarm is triggered may be too long to prevent damage from
    fast-acting attacks like DDoS."
  - Revisor 2, #7: "The choice of 30-minute monitoring blocks appears
    arbitrary... nor does it investigate the sensitivity of the results to
    alternative block sizes."

En vez de agregar estadisticas dentro de bloques fijos (lo que introduce
latencia = tamano del bloque), la carta EWMA suaviza la serie POR FLUJO con
un decaimiento exponencial: reacciona mas rapido que un bloque fijo (no hay
que esperar a que cierre la ventana) pero sigue filtrando ruido de flujo a
flujo, igual que un bloque lo haria. El parametro lambda (decaimiento) juega
el mismo rol que el tamano de bloque -- lambda chico = mas suavizado/mas
latencia, lambda grande = mas reactivo/mas ruidoso -- y es lo que se barre en
`--sweep` para responder el pedido de sensibilidad al "tamano de ventana".

Referencia (fase 1): se transforma T2_train/SPE_train (ya ordenados por
timestamp, ver preprocess.py) con EWMA, y el UCL de esa serie transformada se
calibra con el MISMO bootstrap de bloques moviles que usa el resto del
pipeline (common.moving_block_bootstrap_ucl) -- no se introduce un
multiplicador L "de manual" (SPC clasico) nuevo, para mantener consistencia
metodologica con phase1.py.

Ataques: el cache de scores de classification_metrics.py NO guarda el
timestamp (solo t2/spe), asi que este script reprocesa cada familia
reusando phase2.score_csv (mismo pipeline chunked que phase2.py), ordena por
timestamp y cachea ts+t2+spe en output/scores_ts/<familia>.npz (cache nuevo,
no pisa el de output/scores/).

Cada familia de ataque (y el benigno de test) arranca su EWMA en la media de
referencia de fase 1 (y0 = media de T2/SPE de train), NO encadenada desde el
ultimo valor de la EWMA de train: cada carpeta de malware es una captura de
red independiente (otro dispositivo/escenario, ver MALWARE_ROOT en common.py),
no la continuacion temporal del trafico benigno -- encadenar heredaria a cada
ataque cualquier drift acumulado al final de esa serie sin relacion real con
esa familia. Es la practica estandar en SPC: cada corrida de monitoreo nueva
arranca en el target del proceso, no en el estado final de una corrida previa
no relacionada.

Salidas: ademas de la carta de referencia de fase 1 (ewma_T2.png/ewma_SPE.png,
train), se guarda una carta EWMA POR FAMILIA (ewma_<ataque>_T2.png/_SPE.png,
mismo naming que phase2.py) cuando se corre con un solo lambda (no en --sweep,
donde solo se guardan las metricas agregadas para no generar decenas de PNGs).

Uso:
    python3 control_charts/ewma.py --lam 0.15
    python3 control_charts/ewma.py --sweep
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
from phase2 import score_csv

SCORES_TS_DIR = common.OUTPUT_DIR / "scores_ts"
DEFAULT_LAMBDAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


def ewma_transform(x, lam, y0):
    """y_i = lam*x_i + (1-lam)*y_{i-1}, y_{-1} = y0. x debe estar ordenado
    por timestamp (causal: y_i solo depende del pasado).

    Es un filtro IIR de primer orden (b=[lam], a=[1, -(1-lam)]) -- se usa
    scipy.signal.lfilter (implementado en C) en vez de un loop de Python,
    necesario para que el barrido de lambda (--sweep) sea viable sobre
    familias de millones de filas (IRCbot, Kenjiro, etc.)."""
    from scipy.signal import lfilter, lfiltic

    x = np.asarray(x, dtype=np.float64)
    b, a = [lam], [1.0, -(1 - lam)]
    zi = lfiltic(b, a, [y0])
    y, _ = lfilter(b, a, x, zi=zi)
    return y


def load_train_series():
    import json
    import config
    with open(config.PROCESSED_DIR / "metadata.json") as f:
        categorical_cols = json.load(f)["categorical_cols"]
    data = np.load(config.PROCESSED_DIR / "train.npz")
    cat_arrays = {c: data[f"cat__{c}"] for c in categorical_cols}
    return cat_arrays, data["numeric"]


def load_or_score_family_ts(folder, vae, metadata, vocab_maps, scaler, z_mean, cov_inv):
    """t2/spe por familia ORDENADOS por timestamp, con cache propio (a
    diferencia de output/scores/*.npz, que no guarda ts)."""
    SCORES_TS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = folder.name.replace(" ", "_")
    cache_path = SCORES_TS_DIR / f"{safe_name}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        return d["ts"], d["t2"], d["spe"]

    usecols = metadata["categorical_cols"] + metadata["numeric_cols"] + [metadata["timestamp_col"]]
    csvs = common.attack_csvs(folder)
    ts_all, t2_all, spe_all = [], [], []
    t0 = time.time()
    for path in csvs:
        ts, t2, spe = score_csv(path, vae, metadata, vocab_maps, scaler, z_mean, cov_inv, usecols)
        ts_all.append(ts); t2_all.append(t2); spe_all.append(spe)
    ts = np.concatenate(ts_all); t2 = np.concatenate(t2_all); spe = np.concatenate(spe_all)
    order = np.argsort(ts, kind="stable")
    ts, t2, spe = ts[order], t2[order], spe[order]
    print(f"[{folder.name}] scored+sorted {len(ts):,} rows in {time.time() - t0:.0f}s")

    np.savez(cache_path, ts=ts, t2=t2, spe=spe)
    return ts, t2, spe


def detection_latency(ts, ewma_series, ucl):
    """Primer indice fuera de control: en # de flujos y en minutos desde el
    primer flujo de la familia (proxy de latencia de deteccion 'real-time')."""
    out = np.flatnonzero(ewma_series > ucl)
    if len(out) == 0:
        return None, None, 0, len(ts)
    first = out[0]
    minutes = (ts[first] - ts[0]) / 60_000.0  # timestamps en ms
    return int(first) + 1, float(minutes), len(out), len(ts)


MAX_PLOT_POINTS = 200_000


def downsample_for_plot(anomaly, rng, max_points):
    """Todas las anomalias + una muestra aleatoria de los puntos en control,
    tope max_points -- misma logica que phase2.downsample_for_plot, necesaria
    para que las familias de millones de filas (IRCbot, Kenjiro, etc.) no
    tarden minutos en plotearse."""
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


def plot_ewma_chart(occurrence_id, series, ucl, title, ylabel, out_path, rng,
                     boundary=None, max_points=MAX_PLOT_POINTS):
    anomaly = series > ucl
    idx = downsample_for_plot(anomaly, rng, max_points)
    occ_p, series_p, anomaly_p = occurrence_id[idx], series[idx], anomaly[idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(occ_p[~anomaly_p], series_p[~anomaly_p], s=6, color="#3366cc", label="In control", alpha=0.6, rasterized=True)
    if anomaly_p.any():
        ax.scatter(occ_p[anomaly_p], series_p[anomaly_p], s=8, color="#d62728", label="Out of control", alpha=0.8, zorder=3, rasterized=True)
    ax.axhline(ucl, color="#d62728", linestyle="--", linewidth=1.5, label=f"UCL = {ucl:.2f}")
    if boundary is not None:
        ax.axvline(boundary + 0.5, color="#555555", linestyle=":", linewidth=1.5, label="End of train / start of test")

    n_anom = int(anomaly.sum())
    subtitle = f"{n_anom:,}/{len(series):,} out of control ({100 * n_anom / len(series):.2f}%)"
    if len(idx) < len(series):
        subtitle += f" -- chart subsampled to {len(idx):,} points"
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("Occurrence ID (ordered by timestamp)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_single_lambda(lam, vae, metadata, vocab_maps, scaler, z_mean, cov_inv,
                       cat_train, num_train, folders, n_boot, quantile, rng, make_plots):
    z_train, spe_train = common.encode_batch(vae, cat_train, num_train)
    t2_train = common.hotelling_t2(z_train, z_mean, cov_inv)

    ewma_t2_train = ewma_transform(t2_train, lam, y0=t2_train.mean())
    ewma_spe_train = ewma_transform(spe_train, lam, y0=spe_train.mean())

    block_length = common.default_block_length(len(t2_train))
    ucl_t2, _ = common.moving_block_bootstrap_ucl(ewma_t2_train, block_length, n_boot, quantile, rng)
    ucl_spe, _ = common.moving_block_bootstrap_ucl(ewma_spe_train, block_length, n_boot, quantile, rng)

    if make_plots:
        occ = np.arange(1, len(ewma_t2_train) + 1)
        plot_ewma_chart(occ, ewma_t2_train, ucl_t2,
                         f"Phase 1 (benign train) -- EWMA T2 Control Chart (lambda={lam})",
                         "EWMA(T2)", common.OUTPUT_DIR / "ewma_T2.png", rng)
        plot_ewma_chart(np.arange(1, len(ewma_spe_train) + 1), ewma_spe_train, ucl_spe,
                         f"Phase 1 (benign train) -- EWMA SPE Control Chart (lambda={lam})",
                         "EWMA(SPE)", common.OUTPUT_DIR / "ewma_SPE.png", rng)

    rows = []
    for folder in folders:
        ts, t2, spe = load_or_score_family_ts(folder, vae, metadata, vocab_maps, scaler, z_mean, cov_inv)
        # y0 = media de referencia de fase 1 (el mismo target que arranca la
        # EWMA de train), NO el ultimo valor de la EWMA de train -- cada
        # familia es una captura de red independiente (otro dispositivo/
        # escenario, ver docstring de common.py:MALWARE_ROOT), no la
        # continuacion temporal del trafico benigno. Encadenar el estado de
        # train le heredaria a cada ataque cualquier drift que haya quedado
        # acumulado al final de esa serie, sin relacion real con esa familia.
        ewma_t2 = ewma_transform(t2, lam, y0=t2_train.mean())
        ewma_spe = ewma_transform(spe, lam, y0=spe_train.mean())

        first_t2, mins_t2, n_oc_t2, n = detection_latency(ts, ewma_t2, ucl_t2)
        first_spe, mins_spe, n_oc_spe, _ = detection_latency(ts, ewma_spe, ucl_spe)

        rows.append({
            "attack": folder.name, "lambda": lam, "n_rows": n,
            "recall_t2_pct": 100 * n_oc_t2 / n, "first_alarm_flow_t2": first_t2, "first_alarm_min_t2": mins_t2,
            "recall_spe_pct": 100 * n_oc_spe / n, "first_alarm_flow_spe": first_spe, "first_alarm_min_spe": mins_spe,
        })
        print(f"  [lambda={lam}] {folder.name}: recall_T2={100*n_oc_t2/n:.2f}% "
              f"(1st alarm @flow {first_t2}, {mins_t2 if mins_t2 is None else f'{mins_t2:.1f}min'})  "
              f"recall_SPE={100*n_oc_spe/n:.2f}% (1st alarm @flow {first_spe}, "
              f"{mins_spe if mins_spe is None else f'{mins_spe:.1f}min'})")

        if make_plots:
            safe_name = folder.name.replace(" ", "_")
            occ = np.arange(1, len(ewma_t2) + 1)
            plot_ewma_chart(occ, ewma_t2, ucl_t2,
                             f"Phase 2 -- {folder.name} -- EWMA T2 Control Chart (lambda={lam})",
                             "EWMA(T2)", common.OUTPUT_DIR / f"ewma_{safe_name}_T2.png", rng)
            plot_ewma_chart(occ, ewma_spe, ucl_spe,
                             f"Phase 2 -- {folder.name} -- EWMA SPE Control Chart (lambda={lam})",
                             "EWMA(SPE)", common.OUTPUT_DIR / f"ewma_{safe_name}_SPE.png", rng)

    # falso-positivo en benigno test -- mismo reset a la media de referencia
    # de fase 1 (y0), NO encadenado desde el final de la EWMA de train (ver
    # nota arriba; aplica igual aca: test es una sesion de monitoreo nueva).
    cat_test, num_test = load_test_series()
    z_test, spe_test = common.encode_batch(vae, cat_test, num_test)
    t2_test = common.hotelling_t2(z_test, z_mean, cov_inv)
    ewma_t2_test = ewma_transform(t2_test, lam, y0=t2_train.mean())
    ewma_spe_test = ewma_transform(spe_test, lam, y0=spe_train.mean())
    fpr_t2 = 100 * (ewma_t2_test > ucl_t2).mean()
    fpr_spe = 100 * (ewma_spe_test > ucl_spe).mean()
    rows.append({"attack": "(benign test)", "lambda": lam, "n_rows": len(t2_test),
                 "recall_t2_pct": fpr_t2, "first_alarm_flow_t2": None, "first_alarm_min_t2": None,
                 "recall_spe_pct": fpr_spe, "first_alarm_flow_spe": None, "first_alarm_min_spe": None})
    print(f"  [lambda={lam}] (benign test): FPR_T2={fpr_t2:.2f}%  FPR_SPE={fpr_spe:.2f}%")

    return rows, ucl_t2, ucl_spe


def load_test_series():
    import json
    import config
    with open(config.PROCESSED_DIR / "metadata.json") as f:
        categorical_cols = json.load(f)["categorical_cols"]
    data = np.load(config.PROCESSED_DIR / "test.npz")
    cat_arrays = {c: data[f"cat__{c}"] for c in categorical_cols}
    return cat_arrays, data["numeric"]


def plot_sensitivity(df, out_path):
    agg = df[df["attack"] != "(benign test)"].groupby("lambda").agg(
        mean_latency_min_t2=("first_alarm_min_t2", "mean"),
        mean_recall_t2=("recall_t2_pct", "mean"),
    ).reset_index()
    fpr = df[df["attack"] == "(benign test)"][["lambda", "recall_t2_pct"]].rename(columns={"recall_t2_pct": "fpr_t2"})
    agg = agg.merge(fpr, on="lambda")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(agg["lambda"], agg["mean_latency_min_t2"], "o-", color="#3366cc", label="Mean detection latency (min, T2)")
    ax1.set_xlabel("lambda (EWMA decay)")
    ax1.set_ylabel("Mean detection latency (minutes)", color="#3366cc")
    ax1.tick_params(axis="y", labelcolor="#3366cc")

    ax2 = ax1.twinx()
    ax2.plot(agg["lambda"], agg["fpr_t2"], "s--", color="#d62728", label="Benign FPR (T2)")
    ax2.set_ylabel("Benign false-positive rate (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title("EWMA lambda sensitivity: detection latency vs. false-positive rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, default=0.15)
    parser.add_argument("--sweep", action="store_true", help="Barre --lambdas en vez de un solo lambda")
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack", type=str, default=None)
    args = parser.parse_args()

    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    vae, metadata, vocab_maps, scaler = common.load_artifacts()
    ref = np.load(common.PHASE1_STATS_PATH)
    z_mean, cov_inv = ref["z_mean"], ref["cov_inv"]

    cat_train, num_train = load_train_series()
    folders = common.attack_folders()
    if args.attack:
        folders = [f for f in folders if f.name == args.attack]
        if not folders:
            raise SystemExit(f"Attack folder '{args.attack}' does not exist")

    lambdas = args.lambdas if args.sweep else [args.lam]
    all_rows = []
    for lam in lambdas:
        print(f"\n=== lambda={lam} ===")
        rows, ucl_t2, ucl_spe = run_single_lambda(
            lam, vae, metadata, vocab_maps, scaler, z_mean, cov_inv,
            cat_train, num_train, folders, args.n_boot, args.quantile, rng,
            make_plots=(not args.sweep),
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if args.sweep:
        out_csv = common.OUTPUT_DIR / "ewma_lambda_sensitivity.csv"
        df.to_csv(out_csv, index=False)
        plot_sensitivity(df, common.OUTPUT_DIR / "ewma_lambda_sensitivity.png")
        print(f"\nSaved: {out_csv}")
        print(f"Chart: {common.OUTPUT_DIR / 'ewma_lambda_sensitivity.png'}")
    else:
        out_csv = common.OUTPUT_DIR / "ewma_summary.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")
        print(f"Phase 1 reference charts: {common.OUTPUT_DIR / 'ewma_T2.png'}, {common.OUTPUT_DIR / 'ewma_SPE.png'}")
        print(f"Per-attack charts: {common.OUTPUT_DIR / 'ewma_<attack>_T2.png'} / "
              f"{common.OUTPUT_DIR / 'ewma_<attack>_SPE.png'} (one pair per family)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
