"""Entrena el FlowVAE (Pyro, arquitectura simple de una capa -- ver model.py)
sobre processed/train.npz y evalua en processed/test.npz.

Uso:
    python3 train_vae.py --epochs 20 --batch-size 4096 --z-dim 10 --beta 1.1
"""

import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch

import config
from model import FlowVAE


class FlowTensors:
    """Carga train/test enteros en tensores y arma batches con indexado
    vectorizado (torch.index_select), evitando el overhead de Dataset/
    DataLoader fila-por-fila que con millones de filas tardaria horas."""

    def __init__(self, npz_path, categorical_cols, device):
        data = np.load(npz_path)
        self.numeric = torch.from_numpy(data["numeric"]).float().to(device)
        self.categorical = {
            c: torch.from_numpy(data[f"cat__{c}"]).long().to(device) for c in categorical_cols
        }
        self.n = self.numeric.shape[0]
        self.categorical_cols = categorical_cols

    def batches(self, batch_size, shuffle, device):
        idx = torch.randperm(self.n, device=device) if shuffle else torch.arange(self.n, device=device)
        for start in range(0, self.n, batch_size):
            b = idx[start:start + batch_size]
            x_cat = {c: self.categorical[c].index_select(0, b) for c in self.categorical_cols}
            x_num = self.numeric.index_select(0, b)
            yield x_cat, x_num


def build_pyro_optimizer(vae, lr):
    """Pyro crea un optimizador por cada tensor de parametro. Los embeddings
    sparse (dst_ip, src_port) necesitan SparseAdam en vez de Adam denso --
    si no, el optimizador actualizaria la tabla ENTERA en cada paso sin
    importar cuantas filas del batch la usan."""
    _, sparse_params = vae.parameter_groups()
    sparse_ids = {id(p) for p in sparse_params}

    def optim_constructor(params, **kwargs):
        p = params[0]
        if id(p) in sparse_ids:
            return torch.optim.SparseAdam(params, lr=kwargs.get("lr", lr))
        return torch.optim.Adam(params, **kwargs)

    return pyro.optim.PyroOptim(optim_constructor, {"lr": lr})


def run_epoch(svi, tensors, batch_size, train, device):
    total_loss = 0.0
    total_rows = 0
    for x_cat, x_num in tensors.batches(batch_size, shuffle=train, device=device):
        if train:
            loss = svi.step(x_cat, x_num)
        else:
            loss = svi.evaluate_loss(x_cat, x_num)
        total_loss += loss
        total_rows += x_num.shape[0]
    return total_loss / total_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument("--z-dim", type=int, default=10)
    parser.add_argument("--beta", type=float, default=1.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sparse-vocab-threshold", type=int, default=10_000,
                         help="Columnas categoricas con mas categorias que esto usan embeddings sparse + SparseAdam")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    with open(config.PROCESSED_DIR / "metadata.json") as f:
        metadata = json.load(f)

    categorical_cols = metadata["categorical_cols"]
    numeric_cols = metadata["numeric_cols"]
    vocab_sizes = metadata["vocab_sizes"]
    embedding_dims = metadata["embedding_dims"]
    sparse_cols = [c for c in categorical_cols if vocab_sizes[c] > args.sparse_vocab_threshold]

    train_tensors = FlowTensors(config.PROCESSED_DIR / "train.npz", categorical_cols, device)
    test_tensors = FlowTensors(config.PROCESSED_DIR / "test.npz", categorical_cols, device)

    pyro.clear_param_store()
    vae = FlowVAE(
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        beta=args.beta,
        sparse_cols=sparse_cols,
    ).to(device)

    print(f"Dim. del vector de entrada (embeddings + numericas escaladas): {vae.input_dim}")
    if sparse_cols:
        print(f"Columnas con embedding sparse (cardinalidad > {args.sparse_vocab_threshold}): {sparse_cols}")

    optimizer = build_pyro_optimizer(vae, args.lr)
    svi = pyro.infer.SVI(vae.model, vae.guide, optimizer, loss=pyro.infer.Trace_ELBO())

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history_path = config.CHECKPOINT_DIR / "loss_history.csv"
    plot_path = config.CHECKPOINT_DIR / "loss_curve.png"

    def checkpoint_payload():
        return {
            "embedder": vae.embedder.state_dict(),
            "encoder": vae.encoder.state_dict(),
            "decoder": vae.decoder.state_dict(),
            "args": vars(args),
            "metadata": metadata,
        }

    history = []  # (epoch, train_loss, test_loss) -- loss = -ELBO por observacion, se minimiza
    best_test_loss = float("inf")

    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "test_loss"])

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(svi, train_tensors, args.batch_size, train=True, device=device)
        test_loss = run_epoch(svi, test_tensors, args.batch_size, train=False, device=device)
        print(f"epoch {epoch:3d}  train loss (-ELBO/obs) = {train_loss:.4f}  test loss (-ELBO/obs) = {test_loss:.4f}")

        history.append((epoch, train_loss, test_loss))
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, test_loss])

        torch.save(checkpoint_payload(), config.CHECKPOINT_DIR / "vae_last.pt")
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(checkpoint_payload(), config.CHECKPOINT_DIR / "vae_best.pt")
            print(f"  -> nuevo mejor test loss ({test_loss:.4f}), guardado en vae_best.pt")

        epochs_, train_losses, test_losses = zip(*history)
        plt.figure(figsize=(7, 4))
        plt.plot(epochs_, train_losses, label="train")
        plt.plot(epochs_, test_losses, label="test")
        plt.xlabel("epoch")
        plt.ylabel("loss (-ELBO / obs)")
        plt.title("FlowVAE - curva de perdida")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    print(f"Checkpoints en {config.CHECKPOINT_DIR} (vae_last.pt, vae_best.pt)")
    print(f"Historial de perdida: {history_path}")
    print(f"Grafico de perdida: {plot_path}")


if __name__ == "__main__":
    main()
