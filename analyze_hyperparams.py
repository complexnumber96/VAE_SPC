"""Analisis post-hoc de la busqueda de hiperparametros de optuna_search.py, en
respuesta a dos comentarios de los revisores:

  - Revisor 2, #9: "the choice of beta = 1.1 should be justified, and a
    sensitivity analysis would help demonstrate the robustness of the results
    with respect to this parameter."
  - Revisor 1, minor "Hyperparameter Search Space": "the authors should
    include a short table or list defining the exact search space or the
    range of values tested for these hyperparameters."

Sensibilidad de beta: optuna_search.py corre una busqueda TPE CONJUNTA (beta,
z_dim, hidden_dim, lr, batch_size varian todos a la vez, no un sweep 1-D de
beta), asi que un scatter crudo de test_loss vs. beta esta confundido por los
demas hiperparametros. En vez de eso, se binea beta en deciles y se grafica
mediana + rango intercuartil de la test loss por bin -- una forma honesta de
mostrar la tendencia marginal sin pretender que es un sweep controlado (se
aclara explicitamente en el grafico y en la consola). Los trials podados
(pruned, cortados antes de terminar por MedianPruner) se muestran aparte, no
se mezclan con los completos para el binning.

Search space: los rangos son los mismos `trial.suggest_*` de
optuna_search.py:make_objective -- se transcriben aca en una tabla, no se
re-derivan del CSV (el CSV solo tiene los valores muestreados, no los limites
del espacio de busqueda).

Uso:
    python3 analyze_hyperparams.py
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# Debe coincidir con optuna_search.py:make_objective -- transcrito a mano
# porque optuna no expone el espacio de busqueda declarado, solo los valores
# muestreados por trial.
SEARCH_SPACE = [
    {"hyperparameter": "z_dim", "type": "int", "range": "[2, 64]",
     "note": "techo absoluto para forzar un cuello de botella real (ver model.py / optuna_search.py)"},
    {"hyperparameter": "hidden_dim", "type": "int", "range": "[z_dim + 1, 256]",
     "note": "restringido a z_dim < hidden_dim < input_dim"},
    {"hyperparameter": "beta", "type": "float", "range": "[0.5, 2.0]",
     "note": "techo tambien acotado -- beta alto + decoder chico colapsa el posterior"},
    {"hyperparameter": "lr", "type": "float (log-uniform)", "range": "[1e-5, 1e-2]", "note": ""},
    {"hyperparameter": "batch_size", "type": "categorical", "range": "{128, 256, 512, 1024, 5000}", "note": ""},
]


def beta_sensitivity(trials, n_bins=10):
    complete = trials[trials["state"] == "COMPLETE"].copy()
    pruned = trials[trials["state"] != "COMPLETE"].copy()

    complete["beta_bin"] = pd.qcut(complete["params_beta"], q=min(n_bins, complete["params_beta"].nunique()), duplicates="drop")
    summary = complete.groupby("beta_bin", observed=True)["value"].agg(
        median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75), n="count",
    ).reset_index()
    summary["beta_mid"] = summary["beta_bin"].apply(lambda iv: iv.mid)
    summary = summary.drop(columns="beta_bin").sort_values("beta_mid")
    return summary, complete, pruned


def plot_beta_sensitivity(summary, complete, pruned, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(complete["params_beta"], complete["value"], s=18, color="#3366cc", alpha=0.5, label="Completed trials")
    if len(pruned):
        ax.scatter(pruned["params_beta"], pruned["value"], s=14, color="#999999", alpha=0.4,
                    marker="x", label="Pruned trials (early-stopped)")
    ax.plot(summary["beta_mid"], summary["median"], color="#d62728", linewidth=2, marker="o",
            label="Median test loss per beta decile")
    ax.fill_between(summary["beta_mid"], summary["q1"], summary["q3"], color="#d62728", alpha=0.15,
                     label="IQR per beta decile")
    ax.set_xlabel("beta")
    ax.set_ylabel("Best test loss (-ELBO / obs)")
    ax.set_title("Sensitivity of test loss to beta\n(joint TPE search -- other hyperparameters vary too, see caption)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    trials_path = config.CHECKPOINT_DIR / "optuna_trials.csv"
    trials = pd.read_csv(trials_path)
    print(f"Loaded {len(trials)} trials from {trials_path} "
          f"({(trials['state'] == 'COMPLETE').sum()} complete, "
          f"{(trials['state'] != 'COMPLETE').sum()} pruned/other)")

    summary, complete, pruned = beta_sensitivity(trials, args.n_bins)
    summary_path = config.CHECKPOINT_DIR / "beta_sensitivity.csv"
    summary.to_csv(summary_path, index=False)
    plot_path = config.CHECKPOINT_DIR / "beta_sensitivity.png"
    plot_beta_sensitivity(summary, complete, pruned, plot_path)

    print(f"\nSaved: {summary_path}")
    print(summary.to_string(index=False))
    print(f"Chart: {plot_path}")

    with open(config.CHECKPOINT_DIR / "optuna_best_params.json") as f:
        import json
        best = json.load(f)
    print(f"\nBest trial found: beta={best['params']['beta']:.4f}, "
          f"z_dim={best['params']['z_dim']}, hidden_dim={best['params']['hidden_dim']}, "
          f"lr={best['params']['lr']:.2e}, batch_size={best['params']['batch_size']} "
          f"(test loss = {best['value']:.4f})")

    space_df = pd.DataFrame(SEARCH_SPACE)
    space_path = config.CHECKPOINT_DIR / "hyperparameter_search_space.csv"
    space_df.to_csv(space_path, index=False)
    print(f"\nSaved: {space_path}")
    print(space_df.to_string(index=False))


if __name__ == "__main__":
    main()
