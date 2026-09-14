"""Utilidades compartidas para las cartas de control T2 de Hotelling y SPE,
construidas sobre el espacio latente del FlowVAE -- el mejor modelo hallado
por optuna_search.py (checkpoints/vae_best.pt: z_dim=64, hidden_dim=112,
beta=0.501...).

T2 mide que tan lejos cae z=E[q(z|x)] del centro de la nube de trafico
benigno (fase 1), en unidades de su covarianza. SPE (error cuadratico de
reconstruccion, suma sobre el vector plano embeddings+numericas) mide cuanto
la fila se aleja de lo que el decoder es capaz de reconstruir a partir de esa
z -- la parte de la variacion que el modelo NO explica.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from model import FlowVAE

CONTROL_CHARTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CONTROL_CHARTS_DIR / "output"
PHASE1_STATS_PATH = OUTPUT_DIR / "phase1_reference.npz"

# Attack-data root: tries the Colab path first (so this still works if run
# there), then falls back to the local path -- auto-detected instead of
# hardcoded to either environment.
_MALWARE_ROOT_CANDIDATES = [
    Path("/content/drive/MyDrive/Modelo VAE SPE/capturas_csv"),
    Path("/home/jproanio/Desktop/Script/Capturas_malignas/Malware/capturas_csv"),
]
MALWARE_ROOT = next((p for p in _MALWARE_ROOT_CANDIDATES if p.exists()), _MALWARE_ROOT_CANDIDATES[0])


def load_artifacts(device="cpu"):
    with open(config.PROCESSED_DIR / "metadata.json") as f:
        metadata = json.load(f)
    with open(config.PROCESSED_DIR / "vocabs.pkl", "rb") as f:
        vocab_maps = pickle.load(f)
    with open(config.PROCESSED_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    ckpt = torch.load(config.CHECKPOINT_DIR / "vae_best.pt", map_location=device, weights_only=False)
    args = ckpt["args"]

    categorical_cols = metadata["categorical_cols"]
    vocab_sizes = metadata["vocab_sizes"]
    embedding_dims = metadata["embedding_dims"]
    sparse_cols = [c for c in categorical_cols if vocab_sizes[c] > args["sparse_vocab_threshold"]]

    vae = FlowVAE(
        categorical_cols=categorical_cols,
        numeric_cols=metadata["numeric_cols"],
        vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        hidden_dim=args["hidden_dim"],
        z_dim=args["z_dim"],
        beta=args["beta"],
        sparse_cols=sparse_cols,
    ).to(device)
    vae.embedder.load_state_dict(ckpt["embedder"])
    vae.encoder.load_state_dict(ckpt["encoder"])
    vae.decoder.load_state_dict(ckpt["decoder"])
    vae.eval()

    return vae, metadata, vocab_maps, scaler


def transform_dataframe(df, metadata, vocab_maps, scaler):
    """Replica preprocess.transform_pass: mapea categoricas a los indices del
    vocabulario de entrenamiento (0 = desconocido/no visto -- p.ej. una IP o
    user-agent que solo aparece en trafico de ataque) y escala numericas con
    el MinMaxScaler ajustado sobre trafico benigno (fase 1)."""
    categorical_cols = metadata["categorical_cols"]
    numeric_cols = metadata["numeric_cols"]
    ts_col = metadata["timestamp_col"]

    cat_arrays = {}
    for c in categorical_cols:
        mapping = vocab_maps[c]
        col = df[c] if c in df.columns else pd.Series([None] * len(df))
        cat_arrays[c] = col.map(mapping).fillna(0).astype(np.int32).to_numpy()

    numeric_df = df[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0.0)
    num_array = scaler.transform(numeric_df.to_numpy(dtype=np.float32)).astype(np.float32)

    ts_array = pd.to_numeric(df[ts_col], errors="coerce").fillna(0).astype(np.int64).to_numpy()

    return cat_arrays, num_array, ts_array


@torch.no_grad()
def encode_batch(vae, cat_arrays, num_array, device="cpu", batch_size=65536):
    """z_loc = media de q(z|x) (encoder), SPE = suma de residuos al cuadrado
    entre el decoder(z_loc) y el vector plano de entrada."""
    n = num_array.shape[0]
    z_out = np.empty((n, vae.z_dim), dtype=np.float32)
    spe_out = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_cat = {c: torch.from_numpy(cat_arrays[c][start:end]).long().to(device) for c in vae.categorical_cols}
        x_num = torch.from_numpy(num_array[start:end]).to(device)
        x = vae.assemble(x_cat, x_num)
        z_loc, _ = vae.encoder(x)
        mu = vae.decoder(z_loc)
        spe = ((mu - x) ** 2).sum(dim=-1)
        z_out[start:end] = z_loc.cpu().numpy()
        spe_out[start:end] = spe.cpu().numpy()

    return z_out, spe_out


def hotelling_t2(z, z_mean, cov_inv):
    diff = z - z_mean
    return np.einsum("ij,jk,ik->i", diff, cov_inv, diff)


def fit_t2_reference(z_phase1):
    """Media y covarianza (con shrinkage Ledoit-Wolf, ya que z_dim=64 con
    solo ~2200 filas de fase 1 hace que la covarianza muestral pura sea
    inestable) del espacio latente de trafico benigno."""
    z_mean = z_phase1.mean(axis=0)
    cov = LedoitWolf().fit(z_phase1 - z_mean).covariance_
    cov_inv = np.linalg.inv(cov)
    return z_mean, cov_inv


def moving_block_bootstrap_ucl(values, block_length, n_boot, quantile, rng):
    """Limite de control superior via bootstrap de bloques moviles.

    A diferencia de un bootstrap i.i.d. (que asume filas independientes),
    remuestrea bloques de `block_length` observaciones consecutivas -- asi
    preserva la autocorrelacion temporal propia de una serie de flujos de
    red. Para cada replica bootstrap se calcula el cuantil `quantile` de la
    serie remuestreada; el UCL final es el promedio de esas B estimaciones
    (correccion de sesgo bootstrap sobre el cuantil, mas estable que tomar
    el cuantil empirico directo de una fase 1 chica)."""
    values = np.asarray(values)
    n = len(values)
    block_length = max(1, min(block_length, n))
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length

    boot_quantiles = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        resample = np.concatenate([values[s:s + block_length] for s in starts])[:n]
        boot_quantiles[b] = np.quantile(resample, quantile)

    return float(np.mean(boot_quantiles)), boot_quantiles


def default_block_length(n):
    return max(5, round(n ** (1 / 3)))


def attack_folders():
    return sorted(p for p in MALWARE_ROOT.iterdir() if p.is_dir())


def attack_csvs(folder):
    """CSVs de primer nivel de la carpeta del ataque (ignora parquet, y la
    subcarpeta 'muestra' que trae capturas de OTROS numeros de captura)."""
    return sorted(folder.glob("*.csv"))
