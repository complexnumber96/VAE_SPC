"""Consolida los resultados ya calculados (control_charts/classification_metrics.py,
baselines/ae_control_chart.py, baselines/classification_metrics.py,
baselines/supervised_classification_metrics.py) en 3 tablas de comparacion,
todas con el MISMO esquema de columnas (para que sean directamente
comparables entre si) y sin valores faltantes: una fila "combined" (benigno
test + TODOS los ataques pooled) y una fila por familia, apiladas
(columna `scope`).

  1. comparison_VAE_vs_AE.csv -- FlowVAE vs. AE estandar, AMBAS cartas
     (T2 y SPE) de cada uno -- 4 "metodos": VAE_T2, VAE_SPE, AE_T2, AE_SPE.
     Umbral = UCL bootstrap sobre benigno de train en los 4 casos.
  2. comparison_benign_only_models.csv -- LOS MISMOS 4 (VAE_T2, VAE_SPE,
     AE_T2, AE_SPE) MAS los baselines tradicionales entrenados SOLO con
     benigno: Isolation Forest, One-Class SVM, LOF, PCA, Autoencoder,
     Deep SVDD, VAE convencional (umbral = mejor F1 para estos ultimos,
     ver baselines/classification_metrics.py) -- los 11 juntos y
     comparables, ya que todos comparten el mismo protocolo de
     entrenamiento (solo benigno).
  3. comparison_mixed_trained_models.csv -- Random Forest, XGBoost,
     LightGBM, red neuronal (entrenados con benigno+maligno mezclado;
     umbral nativo = 0.5, ver baselines/supervised_classification_metrics.py).
     VAE/AE no entran aca porque nunca se entrenan con maligno.

No recalcula scores -- solo lee los CSV ya generados por esos 4 scripts
(todos derivados de eval_common.binary_metrics, mismo set de columnas en
todos lados) y los reordena. Correr despues de haber corrido los 4.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONTROL_CHARTS_OUTPUT = ROOT / "control_charts" / "output"
BASELINES_OUTPUT = ROOT / "baselines" / "output"

METRIC_COLS = ["tn", "fp", "fn", "tp", "accuracy", "precision", "recall_sensitivity",
               "specificity", "fpr", "fnr", "f1", "balanced_accuracy", "mcc", "auc"]


def _vae_per_family_metrics(r, suffix):
    """classification_per_attack.csv (control_charts/classification_metrics.py)
    stores a reduced column subset per chart (recall/precision/f1/mcc/auc/
    tn/fp/fn/tp) instead of the full METRIC_COLS set the AE per-family CSV
    has -- derive the rest (accuracy/specificity/fpr/fnr/balanced_accuracy)
    from the confusion counts, using the same formulas as
    eval_common.binary_metrics, so the two are on equal footing."""
    tn, fp, fn, tp = (int(r[f"tn_{suffix}"]), int(r[f"fp_{suffix}"]),
                      int(r[f"fn_{suffix}"]), int(r[f"tp_{suffix}"]))
    n = tn + fp + fn + tp
    recall = r[f"recall_{suffix}"]
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tp + tn) / n,
        "precision": r[f"precision_{suffix}"],
        "recall_sensitivity": recall,
        "specificity": specificity,
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "fnr": fn / (fn + tp) if (fn + tp) else float("nan"),
        "f1": r[f"f1_{suffix}"],
        "balanced_accuracy": (recall + specificity) / 2 if not (np.isnan(recall) or np.isnan(specificity)) else float("nan"),
        "mcc": r[f"mcc_{suffix}"],
        "auc": r[f"auc_{suffix}"],
    }


def vae_ae_rows():
    """Filas VAE_T2/VAE_SPE/AE_T2/AE_SPE (combined + por familia), listas
    para usarse solas o mezcladas con los baselines benigno-only."""
    vae_combined_df = pd.read_csv(CONTROL_CHARTS_OUTPUT / "classification_metrics.csv")
    vae_per_family = pd.read_csv(CONTROL_CHARTS_OUTPUT / "classification_per_attack.csv")
    ae_combined_df = pd.read_csv(BASELINES_OUTPUT / "ae_classification_metrics_combined.csv")
    ae_per_family = pd.read_csv(BASELINES_OUTPUT / "ae_classification_metrics_per_family.csv")

    rows = []
    for combined_chart_key, suffix, method_name in [
        ("T2", "t2", "VAE_T2"), ("SPE", "spe", "VAE_SPE"),
    ]:
        c = vae_combined_df[vae_combined_df["chart"] == combined_chart_key].iloc[0]
        rows.append({"method": method_name, "scope": "combined",
                      "n_attack_rows": int(c["tp"] + c["fn"]), **c[METRIC_COLS].to_dict()})
        for _, r in vae_per_family.iterrows():
            m = _vae_per_family_metrics(r, suffix)
            rows.append({"method": method_name, "scope": r["attack"], "n_attack_rows": int(r["n_rows"]), **m})

    for combined_chart_key, suffix, method_name in [
        ("AE_T2", "t2", "AE_T2"), ("AE_SPE", "spe", "AE_SPE"),
    ]:
        c = ae_combined_df[ae_combined_df["chart"] == combined_chart_key].iloc[0]
        rows.append({"method": method_name, "scope": "combined",
                      "n_attack_rows": int(c["tp"] + c["fn"]), **c[METRIC_COLS].to_dict()})
        for _, r in ae_per_family.iterrows():
            m = {col: r[f"{col}_{suffix}"] for col in METRIC_COLS}
            rows.append({"method": method_name, "scope": r["attack"], "n_attack_rows": int(r["n_rows"]), **m})

    return rows, sorted(vae_per_family["attack"].unique())


def finalize(rows, attack_names, out_name):
    df = pd.DataFrame(rows)
    scope_order = ["combined"] + attack_names
    df["scope"] = pd.Categorical(df["scope"], categories=scope_order, ordered=True)
    df = df.sort_values(["scope", "method"]).reset_index(drop=True)
    out_path = BASELINES_OUTPUT / out_name
    df.to_csv(out_path, index=False)
    return out_path, df


def round_table_rows(combined_path, per_family_path, exclude_methods=()):
    combined = pd.read_csv(combined_path)
    per_family = pd.read_csv(per_family_path)
    if exclude_methods:
        combined = combined[~combined["method"].isin(exclude_methods)]
        per_family = per_family[~per_family["method"].isin(exclude_methods)]

    rows = []
    for _, r in combined.iterrows():
        row = {"method": r["method"], "scope": "combined", "n_attack_rows": int(r["tp"] + r["fn"])}
        row.update({c: r[c] for c in METRIC_COLS})
        rows.append(row)
    for _, r in per_family.iterrows():
        row = {"method": r["method"], "scope": r["attack"], "n_attack_rows": int(r["tp"] + r["fn"])}
        row.update({c: r[c] for c in METRIC_COLS})
        rows.append(row)
    return rows, sorted(per_family["attack"].unique())


def main():
    vae_ae_rows_list, attack_names = vae_ae_rows()

    p1, df1 = finalize(vae_ae_rows_list, attack_names, "comparison_VAE_vs_AE.csv")
    old = BASELINES_OUTPUT / "comparison_VAE_SPE_vs_AE_SPE.csv"
    if old.exists():
        old.unlink()
    print(f"Saved: {p1}  ({len(df1)} filas: {df1['method'].nunique()} metodos x {df1['scope'].nunique()} scopes)")

    benign_only_rows, _ = round_table_rows(
        BASELINES_OUTPUT / "classification_metrics_combined.csv",
        BASELINES_OUTPUT / "classification_metrics_per_family.csv",
        # "RandomForest (supervised)" quedo en este CSV de una version anterior del
        # script pero en realidad se entreno con benigno+ataque mezclado -- no es
        # benigno-only, corresponde a la tabla de modelos supervisados (ver abajo).
        exclude_methods=("RandomForest (supervised)",),
    )
    p2, df2 = finalize(vae_ae_rows_list + benign_only_rows, attack_names, "comparison_benign_only_models.csv")
    print(f"Saved: {p2}  ({len(df2)} filas: {df2['method'].nunique()} metodos x {df2['scope'].nunique()} scopes)")

    mixed_rows, _ = round_table_rows(
        BASELINES_OUTPUT / "supervised_classification_metrics_combined.csv",
        BASELINES_OUTPUT / "supervised_classification_metrics_per_family.csv",
    )
    p3, df3 = finalize(mixed_rows, attack_names, "comparison_mixed_trained_models.csv")
    print(f"Saved: {p3}  ({len(df3)} filas: {df3['method'].nunique()} metodos x {df3['scope'].nunique()} scopes)")


if __name__ == "__main__":
    main()
