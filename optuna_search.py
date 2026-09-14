"""Busqueda de hiperparametros del FlowVAE con Optuna.

Restriccion pedida por el usuario: z_dim < hidden_dim < input_dim (el vector
plano de embeddings + numericas). Ademas busca beta, lr y batch_size.

z_dim y hidden_dim tienen techo absoluto (64 y 256) para forzar un cuello de
botella real: una primera corrida sin techo mostro que minimizar -ELBO sin
restriccion de tamano absoluto empuja z_dim/hidden_dim a pegarse al
input_dim (~1700), es decir el modelo memoriza la entrada en vez de
comprimirla -- inutil para deteccion de anomalias via error de
reconstruccion.

beta tambien tiene techo (0.5-2.0, ver mas abajo): una segunda corrida con
z_dim/hidden_dim chicos pero beta libre mostro el problema opuesto --
colapso de posterior (beta alto + decoder chico hace que el encoder ignore
la entrada). El decoder ademas pas a tener varianza FIJA (ver model.py) para
que no pueda "perdonarse" el error de reconstruccion inflando sigma.

Cada trial entrena un FlowVAE desde cero (pyro.clear_param_store() al
principio, porque Pyro registra los parametros en un param store global por
nombre de modulo) y usa la mejor test loss (-ELBO/obs) del trial como
objetivo a minimizar. Con MedianPruner, los trials claramente peores que la
mediana de trials anteriores en el mismo epoch se cortan antes de las 100
epochs completas -- mismo criterio de busqueda, menos tiempo de computo.

Uso:
    python3 optuna_search.py --n-trials 100 --epochs 100
"""

import argparse
import json
import math

import optuna
import pyro
import torch

import config
from model import FlowVAE
from train_vae import FlowTensors, build_pyro_optimizer, run_epoch


def make_objective(train_tensors, test_tensors, categorical_cols, numeric_cols,
                    vocab_sizes, embedding_dims, input_dim, device, epochs):
    def objective(trial):
        # Techo absoluto de z_dim: sin esto, minimizar -ELBO solo empuja a
        # z_dim/hidden_dim pegados al input_dim (el modelo memoriza la
        # entrada en vez de comprimirla), lo que inutiliza al VAE para
        # deteccion de anomalias via error de reconstruccion.
        z_dim = trial.suggest_int("z_dim", 2, 64)
        hidden_dim = trial.suggest_int("hidden_dim", z_dim + 1, 256)
        # Techo tambien a beta: sin esto, beta alto + decoder chico
        # colapsaba el posterior (ver docstring de model.py:Decoder).
        beta = trial.suggest_float("beta", 0.5, 2.0)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024,5000])

        pyro.clear_param_store()
        vae = FlowVAE(
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            vocab_sizes=vocab_sizes,
            embedding_dims=embedding_dims,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            beta=beta,
            sparse_cols=(),
        ).to(device)

        optimizer = build_pyro_optimizer(vae, lr)
        svi = pyro.infer.SVI(vae.model, vae.guide, optimizer, loss=pyro.infer.Trace_ELBO())

        best_test_loss = float("inf")
        for epoch in range(1, epochs + 1):
            # Combinaciones de lr/beta/hidden_dim agresivas pueden divergir
            # (pesos -> NaN), lo que hace que Pyro tire ValueError al validar
            # los parametros de la Normal. Un trial asi se poda en vez de
            # matar toda la busqueda.
            try:
                run_epoch(svi, train_tensors, batch_size, train=True, device=device)
                test_loss = run_epoch(svi, test_tensors, batch_size, train=False, device=device)
            except (ValueError, RuntimeError) as e:
                print(f"  trial {trial.number} diverged en epoch {epoch}: {e}")
                raise optuna.TrialPruned()

            if not math.isfinite(test_loss):
                print(f"  trial {trial.number} produjo test_loss no finito en epoch {epoch}")
                raise optuna.TrialPruned()

            best_test_loss = min(best_test_loss, test_loss)

            trial.report(test_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_test_loss

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--study-name", default="flowvae_search")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    with open(config.PROCESSED_DIR / "metadata.json") as f:
        metadata = json.load(f)

    categorical_cols = metadata["categorical_cols"]
    numeric_cols = metadata["numeric_cols"]
    vocab_sizes = metadata["vocab_sizes"]
    embedding_dims = metadata["embedding_dims"]
    input_dim = sum(embedding_dims[c] for c in categorical_cols) + len(numeric_cols)
    print(f"input_dim (embeddings + numericas): {input_dim}")

    train_tensors = FlowTensors(config.PROCESSED_DIR / "train.npz", categorical_cols, device)
    test_tensors = FlowTensors(config.PROCESSED_DIR / "test.npz", categorical_cols, device)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                 study_name=args.study_name)

    objective = make_objective(train_tensors, test_tensors, categorical_cols, numeric_cols,
                                vocab_sizes, embedding_dims, input_dim, device, args.epochs)
    study.optimize(objective, n_trials=args.n_trials)

    print("\nMejor trial:")
    print(f"  test loss (-ELBO/obs) = {study.best_value:.4f}")
    print(f"  params = {study.best_params}")

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = config.CHECKPOINT_DIR / "optuna_best_params.json"
    with open(best_path, "w") as f:
        json.dump({"value": study.best_value, "params": study.best_params, "input_dim": input_dim}, f, indent=2)

    trials_path = config.CHECKPOINT_DIR / "optuna_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)

    print(f"\nMejores hiperparametros: {best_path}")
    print(f"Historial de todos los trials: {trials_path}")


if __name__ == "__main__":
    main()
