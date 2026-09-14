"""Utilidades de evaluacion binaria (benigno=0 / ataque=1) compartidas por
control_charts/classification_metrics.py (FlowVAE T2/SPE -- umbral = UCL de
fase 1, calibrado por bootstrap de bloques moviles SOLO sobre benigno) y
baselines/classification_metrics.py (Isolation Forest, One-Class SVM,
Autoencoder, Deep SVDD, VAE convencional, LOF, PCA -- umbral = mejor F1
sobre su propio score, ver docstring de ese archivo para la justificacion de
por que estos dos grupos usan reglas de umbral distintas).
"""

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve


def sanitize_scores(scores):
    """Reemplaza NaN/+-inf en un array de scores de anomalia por el
    minimo/maximo finito del mismo array, para que best_f1_threshold/
    roc_auc_score (que no toleran valores no finitos) no fallen si algun
    modelo produce un score degenerado en un puñado de filas."""
    scores = np.asarray(scores, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 0.0
    return np.nan_to_num(scores, nan=lo, posinf=hi, neginf=lo)


def best_f1_threshold(y_true, y_score):
    """Umbral que maximiza F1 sobre (y_true, y_score), barriendo todos los
    puntos de corte que devuelve precision_recall_curve. Estandar en
    literatura de deteccion de anomalias cuando se reporta matriz de
    confusion junto con AUC (a diferencia del UCL del FlowVAE, que se
    calibra SOLO con benigno de train, sin ver ataques)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[np.argmax(f1)])


def binary_metrics(y_true, y_pred, y_score, label):
    """Matriz de confusion + metricas de clasificacion binaria (benigno=0,
    ataque=1) a partir de una prediccion binaria fija, mas AUC (que no
    depende del umbral)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")  # sensitivity / TPR
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")  # TNR
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    balanced_acc = (recall + specificity) / 2 if not (np.isnan(recall) or np.isnan(specificity)) else float("nan")
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "label": label,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(n),
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "auc": auc,
    }


def roc_points(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


def plot_confusion_matrix(cm, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign (0)", "Attack (1)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign (0)", "Attack (1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > vmax / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
