"""Aisla el aporte del propio MONITOREO SPC (la calibracion del UCL via
bootstrap de bloques moviles, SOLO sobre benigno) frente a un umbral
"convencional" que si mira ataques -- la tercera pata del comentario del
revisor 2 sobre ablacion (ver tambien baselines/ae_on_embeddings.py para las
otras dos: representacion y arquitectura):

    "the contribution of each component (VAE, embeddings, and SPC
    monitoring) should be quantified separately."

Toma los MISMOS scores T2/SPE del FlowVAE propuesto (ya cacheados por
control_charts/classification_metrics.py en output/scores/*.npz -- no se
recalcula nada del modelo) y los evalua con DOS umbrales distintos:

  1. UCL bootstrap de fase 1 (el que usa el paper) -- calibrado SOLO con
     benigno de train, sin mirar ningun ataque.
  2. Umbral de mejor F1 (eval_common.best_f1_threshold, el mismo criterio que
     usan los demas baselines en baselines/classification_metrics.py) sobre
     el pool benigno-test + todos los ataques -- este SI mira ataques
     (aunque sea de forma agregada), es una comparacion deliberadamente
     favorable al umbral alternativo.

La diferencia entre ambas filas, con el MISMO score de entrada, es
exactamente lo que aporta (o cuesta) la etapa de monitoreo SPC no supervisada
frente a un umbral que optimiza directamente sobre datos etiquetados que en
un despliegue real no se tendrian de antemano.

Uso (requiere haber corrido antes control_charts/phase1.py y
control_charts/classification_metrics.py, para poblar output/scores/ y
output/phase1_reference.npz):
    python3 control_charts/spc_threshold_ablation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_common import best_f1_threshold, binary_metrics

SCORES_DIR = common.OUTPUT_DIR / "scores"


def load_family_scores():
    families = {}
    for path in sorted(SCORES_DIR.glob("*.npz")):
        d = np.load(path)
        families[path.stem] = (d["t2"], d["spe"])
    return families


def main():
    ref = np.load(common.PHASE1_STATS_PATH)
    ucl_t2_spc, ucl_spe_spc = float(ref["ucl_t2"]), float(ref["ucl_spe"])
    print(f"UCL (SPC bootstrap, fase 1): T2={ucl_t2_spc:.4f}  SPE={ucl_spe_spc:.4f}")

    families = load_family_scores()
    if "benign_test" not in families:
        raise SystemExit("No se encontro output/scores/benign_test.npz -- corre "
                          "control_charts/classification_metrics.py primero.")

    t2_benign, spe_benign = families.pop("benign_test")
    y_parts = [np.zeros(len(t2_benign), dtype=np.int8)]
    t2_parts, spe_parts = [t2_benign], [spe_benign]
    for t2_a, spe_a in families.values():
        y_parts.append(np.ones(len(t2_a), dtype=np.int8))
        t2_parts.append(t2_a); spe_parts.append(spe_a)
    y_true = np.concatenate(y_parts)
    t2_all = np.concatenate(t2_parts)
    spe_all = np.concatenate(spe_parts)

    ucl_t2_bestf1 = best_f1_threshold(y_true, t2_all)
    ucl_spe_bestf1 = best_f1_threshold(y_true, spe_all)
    print(f"Umbral (best-F1, mira ataques): T2={ucl_t2_bestf1:.4f}  SPE={ucl_spe_bestf1:.4f}")

    rows = []
    for threshold_name, ucl_t2, ucl_spe in [
        ("SPC bootstrap (unsupervised, benign-only)", ucl_t2_spc, ucl_spe_spc),
        ("Best-F1 (sees attacks, pooled)", ucl_t2_bestf1, ucl_spe_bestf1),
    ]:
        m_t2 = binary_metrics(y_true, (t2_all > ucl_t2).astype(np.int8), t2_all, "T2")
        m_spe = binary_metrics(y_true, (spe_all > ucl_spe).astype(np.int8), spe_all, "SPE")
        for chart, m in [("T2", m_t2), ("SPE", m_spe)]:
            rows.append({"threshold": threshold_name, "chart": chart,
                         **{k: v for k, v in m.items() if k != "label"}})

    df = pd.DataFrame(rows)
    out_path = common.OUTPUT_DIR / "spc_threshold_ablation.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(df[["threshold", "chart", "recall_sensitivity", "precision", "fpr", "f1", "auc"]].to_string(index=False))
    print("\n(AUC es igual entre ambas filas del mismo chart -- no depende del umbral; "
          "la diferencia relevante esta en recall/precision/FPR a umbral fijo)")


if __name__ == "__main__":
    main()
