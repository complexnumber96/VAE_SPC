"""Consolida el ablation study por componente pedido por el revisor 2 (#4, #5)
en una sola tabla + grafico, juntando resultados de scripts ya corridos:

  - control_charts/classification_metrics.py -> FlowVAE (embeddings, variational)
  - baselines/run_baselines.py + baselines/classification_metrics.py -> VanillaVAE
    (frequency-encoding, variational) via baselines_comparison / classification_metrics_combined
  - baselines/ae_on_embeddings.py -> AE on embeddings (embeddings, deterministic) [NUEVO]
  - baselines/ae_control_chart.py -> AE estandar (frequency-encoding, deterministic)
  - control_charts/spc_threshold_ablation.py -> aporte de la etapa de monitoreo SPC

No recalcula nada -- solo lee los CSV ya generados por esos 5 scripts y arma:
  1. ablation_2x2_representation_architecture.csv -- las 4 celdas (x T2/SPE),
     mismo protocolo bootstrap-UCL quantile=0.95 en las 4.
  2. ablation_2x2_auc.png -- barras de AUC (libre de umbral) por celda.
  3. ablation_component_contributions.csv -- las 4 celdas + las 2 filas de
     spc_threshold_ablation.py, todo junto para la tabla del paper.

Correr despues de: control_charts/phase1.py, control_charts/classification_metrics.py,
baselines/run_baselines.py, baselines/classification_metrics.py,
baselines/ae_control_chart.py, baselines/ae_on_embeddings.py,
control_charts/spc_threshold_ablation.py.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONTROL_CHARTS_OUTPUT = ROOT / "control_charts" / "output"
BASELINES_OUTPUT = ROOT / "baselines" / "output"

METRIC_COLS = ["tn", "fp", "fn", "tp", "accuracy", "precision", "recall_sensitivity",
               "specificity", "fpr", "fnr", "f1", "balanced_accuracy", "mcc", "auc"]


def combined_row(df, chart_col, chart_value, method_name, threshold_protocol):
    r = df[df[chart_col] == chart_value].iloc[0]
    cols = [c for c in METRIC_COLS if c in r.index]
    row = {"method": method_name, "threshold_protocol": threshold_protocol,
           **{c: r[c] for c in cols}}
    return row


def main():
    rows = []
    BOOT = "bootstrap-UCL (quantile=0.95, benign-only)"

    # embeddings + variational: FlowVAE
    vae_combined = pd.read_csv(CONTROL_CHARTS_OUTPUT / "classification_metrics.csv")
    rows.append({**combined_row(vae_combined, "chart", "T2", "FlowVAE_T2", BOOT),
                 "representation": "embeddings (learned)", "architecture": "variational"})
    rows.append({**combined_row(vae_combined, "chart", "SPE", "FlowVAE_SPE", BOOT),
                 "representation": "embeddings (learned)", "architecture": "variational"})

    # frequency-encoding + variational: VanillaVAE. AUC (libre de umbral) es
    # comparable directamente contra las otras 3 celdas; recall/precision/F1
    # NO lo son -- vienen de classification_metrics_combined.csv, que usa
    # umbral de mejor-F1 (mira ataques, favorable al baseline -- ver docstring
    # de baselines/classification_metrics.py), no el bootstrap-UCL de las
    # otras celdas. Se deja explicito en la columna threshold_protocol.
    baselines_combined = pd.read_csv(BASELINES_OUTPUT / "classification_metrics_combined.csv")
    vv = baselines_combined[baselines_combined["method"] == "VanillaVAE"]
    if len(vv):
        rows.append({**combined_row(vv, "method", "VanillaVAE", "VanillaVAE_reconprob",
                                     "best-F1 (sees attacks, NOT comparable to bootstrap-UCL rows except via AUC)"),
                     "representation": "frequency-encoding", "architecture": "variational"})

    # embeddings + deterministic: AE on embeddings (NUEVO, ae_on_embeddings.py)
    ae_emb_path = BASELINES_OUTPUT / "ae_embeddings_classification_metrics_combined.csv"
    if ae_emb_path.exists():
        ae_emb = pd.read_csv(ae_emb_path)
        rows.append({**combined_row(ae_emb, "chart", "AE_embeddings_T2", "AE_embeddings_T2", BOOT),
                     "representation": "embeddings (learned)", "architecture": "deterministic"})
        rows.append({**combined_row(ae_emb, "chart", "AE_embeddings_SPE", "AE_embeddings_SPE", BOOT),
                     "representation": "embeddings (learned)", "architecture": "deterministic"})
    else:
        print(f"AVISO: no se encontro {ae_emb_path} -- corre baselines/ae_on_embeddings.py primero")

    # frequency-encoding + deterministic: AE estandar
    ae_std_path = BASELINES_OUTPUT / "ae_classification_metrics_combined.csv"
    if ae_std_path.exists():
        ae_std = pd.read_csv(ae_std_path)
        rows.append({**combined_row(ae_std, "chart", "AE_T2", "AE_standard_T2", BOOT),
                     "representation": "frequency-encoding", "architecture": "deterministic"})
        rows.append({**combined_row(ae_std, "chart", "AE_SPE", "AE_standard_SPE", BOOT),
                     "representation": "frequency-encoding", "architecture": "deterministic"})
    else:
        print(f"AVISO: no se encontro {ae_std_path} -- corre baselines/ae_control_chart.py primero")

    df_2x2 = pd.DataFrame(rows)
    out1 = BASELINES_OUTPUT / "ablation_2x2_representation_architecture.csv"
    df_2x2.to_csv(out1, index=False)
    print(f"Saved: {out1}")
    print(df_2x2[["method", "representation", "architecture", "recall_sensitivity", "fpr", "f1", "auc"]].to_string(index=False))

    if len(df_2x2):
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"embeddings (learned)": "#3366cc", "frequency-encoding": "#d62728"}
        bar_colors = [colors[r] for r in df_2x2["representation"]]
        ax.barh(df_2x2["method"], df_2x2["auc"], color=bar_colors)
        ax.set_xlabel("AUC (threshold-independent)")
        ax.set_title("Ablation: representation x architecture (AUC)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
        ax.legend(handles, colors.keys(), loc="lower right")
        fig.tight_layout()
        fig.savefig(BASELINES_OUTPUT / "ablation_2x2_auc.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {BASELINES_OUTPUT / 'ablation_2x2_auc.png'}")

    # tercer componente: aporte del monitoreo SPC (umbral bootstrap vs. best-F1)
    spc_path = CONTROL_CHARTS_OUTPUT / "spc_threshold_ablation.csv"
    final_rows = df_2x2.to_dict("records")
    if spc_path.exists():
        spc_df = pd.read_csv(spc_path)
        for _, r in spc_df.iterrows():
            row = {"method": f"FlowVAE_{r['chart']} [{r['threshold']}]",
                   "representation": "embeddings (learned)", "architecture": "variational"}
            row.update({c: r[c] for c in METRIC_COLS if c in r.index})
            final_rows.append(row)
    else:
        print(f"AVISO: no se encontro {spc_path} -- corre control_charts/spc_threshold_ablation.py primero")

    out2 = BASELINES_OUTPUT / "ablation_component_contributions.csv"
    pd.DataFrame(final_rows).to_csv(out2, index=False)
    print(f"\nSaved: {out2}")


if __name__ == "__main__":
    main()
