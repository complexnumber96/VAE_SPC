"""Compara el UCL empirico (bootstrap de bloques moviles, ver phase1.py) con
un UCL teorico clasico de SPC, en respuesta al comentario del revisor 2:

    "The proposed framework uses empirical 99th-percentile thresholds,
    whereas classical SPC literature typically derives control limits from
    theoretical distributions or well-established Phase I procedures."

  - T2 de Hotelling: formula clasica de Tracy, Young & Mason (1992) para una
    observacion individual nueva contra una muestra de referencia de tamano n
    y dimension p (common.theoretical_t2_ucl), via la distribucion F.
  - SPE: aproximacion de Box / Jackson-Mudholkar (chi-cuadrado escalada por
    metodo de momentos sobre el SPE de referencia, common.theoretical_spe_ucl)
    -- no se asume el decoder_sigma fijo del entrenamiento como varianza real
    de los residuos, se estima empiricamente de fase 1 (train benigno), igual
    que el bootstrap.

Recalcula z_train/spe_train/t2_train (barato, ~1800 filas, mismo patron que
phase1.py) para no depender de que phase1.py haya guardado esos arrays crudos
en phase1_reference.npz (que solo guarda el UCL bootstrap ya calculado).

Reevalua clasificacion con el UCL teorico sobre los scores por familia ya
cacheados en output/scores/*.npz (control_charts/classification_metrics.py),
para mostrar el impacto real en recall/FPR de usar un limite u otro -- no solo
la diferencia numerica entre ambos UCL.

Uso:
    python3 control_charts/theoretical_limits.py --alpha 0.05
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_common import binary_metrics

SCORES_DIR = common.OUTPUT_DIR / "scores"


def load_train_stats():
    import json
    import config
    with open(config.PROCESSED_DIR / "metadata.json") as f:
        categorical_cols = json.load(f)["categorical_cols"]
    data = np.load(config.PROCESSED_DIR / "train.npz")
    cat_arrays = {c: data[f"cat__{c}"] for c in categorical_cols}
    return cat_arrays, data["numeric"]


def load_family_scores():
    families = {}
    for path in sorted(SCORES_DIR.glob("*.npz")):
        d = np.load(path)
        families[path.stem] = (d["t2"], d["spe"])
    return families


def plot_overlay(ts_train_boundary, stat_full, ucl_bootstrap, ucl_theoretical, title, ylabel, out_path):
    n = len(stat_full)
    occurrence_id = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(occurrence_id, stat_full, s=8, color="#3366cc", alpha=0.5, label="Statistic")
    ax.axhline(ucl_bootstrap, color="#d62728", linestyle="--", linewidth=1.5,
                label=f"UCL bootstrap = {ucl_bootstrap:.2f}")
    ax.axhline(ucl_theoretical, color="#2ca02c", linestyle="-.", linewidth=1.5,
                label=f"UCL theoretical = {ucl_theoretical:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Occurrence ID (ordered by timestamp)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.05,
                         help="Nivel de significancia (0.05 <-> quantile=0.95, "
                              "para comparar contra el bootstrap default de phase1.py)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    vae, metadata, vocab_maps, scaler = common.load_artifacts()
    cat_train, num_train = load_train_stats()
    n_train = num_train.shape[0]
    p = vae.z_dim

    print(f"Fase 1 (train benigno): n={n_train}, p (z_dim)={p}, alpha={args.alpha}")

    z_train, spe_train = common.encode_batch(vae, cat_train, num_train)
    z_mean, cov_inv = common.fit_t2_reference(z_train)
    t2_train = common.hotelling_t2(z_train, z_mean, cov_inv)

    block_length = common.default_block_length(n_train)
    ucl_t2_boot, _ = common.moving_block_bootstrap_ucl(t2_train, block_length, args.n_boot, 1 - args.alpha, rng)
    ucl_spe_boot, _ = common.moving_block_bootstrap_ucl(spe_train, block_length, args.n_boot, 1 - args.alpha, rng)

    ucl_t2_theo = common.theoretical_t2_ucl(n_train, p, args.alpha)
    ucl_spe_theo = common.theoretical_spe_ucl(spe_train, args.alpha)

    print(f"T2  -- UCL bootstrap = {ucl_t2_boot:.4f}   UCL theoretical (Tracy-Young-Mason F) = {ucl_t2_theo:.4f}")
    print(f"SPE -- UCL bootstrap = {ucl_spe_boot:.4f}   UCL theoretical (Box/Jackson-Mudholkar chi2) = {ucl_spe_theo:.4f}")

    families = load_family_scores()
    if not families:
        print("(no se encontraron scores cacheados en output/scores/ -- corre "
              "control_charts/classification_metrics.py primero para poblar el cache "
              "y poder comparar el impacto en clasificacion)")
    else:
        rows = []
        for name, (t2, spe) in families.items():
            label = "(benign test)" if name == "benign_test" else name
            y = np.zeros(len(t2), dtype=np.int8) if name == "benign_test" else np.ones(len(t2), dtype=np.int8)
            rows.append({"attack": label, "n_rows": len(t2), "limit": "bootstrap",
                         **{f"{k}_t2": v for k, v in binary_metrics(y, (t2 > ucl_t2_boot).astype(np.int8), t2, "T2").items() if k != "label"},
                         **{f"{k}_spe": v for k, v in binary_metrics(y, (spe > ucl_spe_boot).astype(np.int8), spe, "SPE").items() if k != "label"}})
            rows.append({"attack": label, "n_rows": len(t2), "limit": "theoretical",
                         **{f"{k}_t2": v for k, v in binary_metrics(y, (t2 > ucl_t2_theo).astype(np.int8), t2, "T2").items() if k != "label"},
                         **{f"{k}_spe": v for k, v in binary_metrics(y, (spe > ucl_spe_theo).astype(np.int8), spe, "SPE").items() if k != "label"}})
        df = pd.DataFrame(rows)
        out_csv = common.OUTPUT_DIR / "theoretical_vs_empirical_ucl.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")
        print(df[["attack", "limit", "recall_sensitivity_t2", "fpr_t2", "recall_sensitivity_spe", "fpr_spe"]]
              .to_string(index=False))

        if "benign_test" in families:
            t2_bt, spe_bt = families["benign_test"]
            attack_t2 = np.concatenate([v[0] for k, v in families.items() if k != "benign_test"]) if len(families) > 1 else np.array([])
            attack_spe = np.concatenate([v[1] for k, v in families.items() if k != "benign_test"]) if len(families) > 1 else np.array([])
            full_t2 = np.concatenate([t2_bt, attack_t2])
            full_spe = np.concatenate([spe_bt, attack_spe])
            plot_overlay(len(t2_bt), full_t2, ucl_t2_boot, ucl_t2_theo,
                         "Bootstrap vs. theoretical UCL -- Hotelling's T2 (benign test + all attacks, pooled)",
                         "T2", common.OUTPUT_DIR / "theoretical_limits_T2.png")
            plot_overlay(len(spe_bt), full_spe, ucl_spe_boot, ucl_spe_theo,
                         "Bootstrap vs. theoretical UCL -- SPE (benign test + all attacks, pooled)",
                         "SPE (reconstruction error)", common.OUTPUT_DIR / "theoretical_limits_SPE.png")
            print(f"Charts: {common.OUTPUT_DIR / 'theoretical_limits_T2.png'}, "
                  f"{common.OUTPUT_DIR / 'theoretical_limits_SPE.png'}")

    summary = pd.DataFrame([
        {"chart": "T2", "alpha": args.alpha, "ucl_bootstrap": ucl_t2_boot, "ucl_theoretical": ucl_t2_theo,
         "pct_diff": 100 * (ucl_t2_theo - ucl_t2_boot) / ucl_t2_boot},
        {"chart": "SPE", "alpha": args.alpha, "ucl_bootstrap": ucl_spe_boot, "ucl_theoretical": ucl_spe_theo,
         "pct_diff": 100 * (ucl_spe_theo - ucl_spe_boot) / ucl_spe_boot},
    ])
    summary_path = common.OUTPUT_DIR / "theoretical_vs_empirical_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
