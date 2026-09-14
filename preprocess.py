"""Preprocesamiento streaming de los parquet de nfstream para el VAE.

No carga los parquet completos (texto) en RAM: los recorre en dos pasadas por
lotes con pyarrow.dataset.

  Pasada 1 (fit):  arma el vocabulario de cada columna categorica y ajusta un
                    MinMaxScaler de forma incremental (partial_fit) sobre las
                    columnas numericas.
  Pasada 2 (transform): vuelve a recorrer los lotes, ahora convirtiendo cada
                    columna categorica a su indice entero (0 = desconocido/
                    faltante) y cada columna numerica a su valor escalado
                    [0, 1], acumulando todo en arrays numpy compactos
                    (int32 / float32) ya en RAM (~3-4 GB, entra sin problema).

Al final ordena por bidirectional_first_seen_ms, separa 80% train / 20% test
sin mezclar (evita data leakage temporal) y guarda:
  processed/train.npz, processed/test.npz, processed/metadata.json,
  processed/scaler.pkl
"""

import json
import pickle

import numpy as np
import pyarrow.dataset as ds
from sklearn.preprocessing import MinMaxScaler

import config


def _batches():
    dataset = ds.dataset([str(p) for p in sorted(config.PARQUET_DIR.glob("*.parquet"))], format="parquet")
    return dataset.to_batches(batch_size=config.BATCH_SIZE, columns=config.READ_COLS)


def fit_pass():
    print("Pasada 1/2: ajustando vocabularios y scaler...")
    vocabs = {c: set() for c in config.CATEGORICAL_COLS}
    scaler = MinMaxScaler()
    total_rows = 0

    for i, batch in enumerate(_batches()):
        df = batch.to_pandas()
        total_rows += len(df)

        for c in config.CATEGORICAL_COLS:
            vocabs[c].update(df[c].dropna().unique().tolist())

        numeric_df = df[config.NUMERIC_COLS].apply(lambda s: __import__("pandas").to_numeric(s, errors="coerce"))
        numeric_df = numeric_df.fillna(0.0).astype(np.float32)
        scaler.partial_fit(numeric_df.values)

        print(f"\r  lote {i + 1}, {total_rows:,} filas vistas...", end="", flush=True)

    print(f"\nTotal filas: {total_rows:,}")

    vocab_maps = {}
    embedding_dims = {}
    for c, values in vocabs.items():
        ordered = sorted(values)
        vocab_maps[c] = {v: idx + 1 for idx, v in enumerate(ordered)}  # 0 = desconocido/faltante
        embedding_dims[c] = config.embedding_dim_for_cardinality(len(ordered) + 1)
        print(f"  {c}: {len(ordered)} categorias -> emb_dim={embedding_dims[c]}")

    return vocab_maps, embedding_dims, scaler, total_rows


def transform_pass(vocab_maps, scaler, total_rows):
    print("Pasada 2/2: transformando y acumulando en memoria...")
    import pandas as pd

    cat_arrays = {c: np.zeros(total_rows, dtype=np.int32) for c in config.CATEGORICAL_COLS}
    num_array = np.zeros((total_rows, len(config.NUMERIC_COLS)), dtype=np.float32)
    ts_array = np.zeros(total_rows, dtype=np.int64)

    offset = 0
    for i, batch in enumerate(_batches()):
        df = batch.to_pandas()
        n = len(df)

        for c in config.CATEGORICAL_COLS:
            mapping = vocab_maps[c]
            cat_arrays[c][offset:offset + n] = df[c].map(mapping).fillna(0).astype(np.int32).values

        numeric_df = df[config.NUMERIC_COLS].apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0.0)
        num_array[offset:offset + n, :] = scaler.transform(numeric_df.values).astype(np.float32)

        ts_array[offset:offset + n] = pd.to_numeric(df[config.TIMESTAMP_COL], errors="coerce").fillna(0).astype(np.int64).values

        offset += n
        print(f"\r  lote {i + 1}, {offset:,}/{total_rows:,} filas...", end="", flush=True)

    print()
    assert offset == total_rows
    return cat_arrays, num_array, ts_array


def sort_split_and_save(cat_arrays, num_array, ts_array, vocab_maps, embedding_dims):
    print("Ordenando por timestamp y dividiendo train/test (80/20, sin mezclar)...")
    order = np.argsort(ts_array, kind="stable")

    ts_array = ts_array[order]
    num_array = num_array[order]
    for c in cat_arrays:
        cat_arrays[c] = cat_arrays[c][order]
    del order

    total_rows = len(ts_array)
    n_train = int(config.TRAIN_FRACTION * total_rows)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def save_split(name, lo, hi):
        payload = {"numeric": num_array[lo:hi]}
        for c in config.CATEGORICAL_COLS:
            payload[f"cat__{c}"] = cat_arrays[c][lo:hi]
        np.savez_compressed(config.PROCESSED_DIR / f"{name}.npz", **payload)
        np.save(config.PROCESSED_DIR / f"{name}_timestamp.npy", ts_array[lo:hi])
        print(f"  {name}: {hi - lo:,} filas -> {config.PROCESSED_DIR / (name + '.npz')}")

    save_split("train", 0, n_train)
    save_split("test", n_train, total_rows)

    metadata = {
        "categorical_cols": config.CATEGORICAL_COLS,
        "numeric_cols": config.NUMERIC_COLS,
        "vocab_sizes": {c: len(v) + 1 for c, v in vocab_maps.items()},  # +1 por el indice 0 (desconocido)
        "embedding_dims": embedding_dims,
        "n_train": n_train,
        "n_test": total_rows - n_train,
        "timestamp_col": config.TIMESTAMP_COL,
    }
    with open(config.PROCESSED_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Guardamos los vocabularios aparte (pueden ser grandes, ej. dst_ip ~11M entradas)
    with open(config.PROCESSED_DIR / "vocabs.pkl", "wb") as f:
        pickle.dump(vocab_maps, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    vocab_maps, embedding_dims, scaler, total_rows = fit_pass()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PROCESSED_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    cat_arrays, num_array, ts_array = transform_pass(vocab_maps, scaler, total_rows)
    sort_split_and_save(cat_arrays, num_array, ts_array, vocab_maps, embedding_dims)
    print("Preprocesamiento completo.")


if __name__ == "__main__":
    main()
