"""Muestreo aleatorio de filas de un CSV de ataque sin cargarlo completo en
RAM (algunos pesan varios GB, ver control_charts/phase2.py). Un solo pasada:
cada chunk aporta una fraccion proporcional al tamano total del archivo
(estimado via `wc -l`), asi el sample final es aprox. uniforme sobre todo el
archivo (no solo las primeras filas) sin necesitar reservoir sampling fila
por fila."""

import subprocess

import numpy as np
import pandas as pd

CHUNK_SIZE = 200_000


def count_rows(path):
    """Numero de filas de datos (excluye el header) via `wc -l`, mucho mas
    rapido que dejar que pandas parsee todo el archivo dos veces."""
    out = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0]) - 1


def sample_csv(path, usecols, target_n, rng, chunk_size=CHUNK_SIZE):
    """Devuelve un DataFrame de hasta `target_n` filas, muestreadas
    aproximadamente uniforme sobre todo el archivo."""
    n_total = count_rows(path)
    if n_total <= target_n:
        frac = 1.0
    else:
        frac = target_n / n_total

    parts = []
    n_sampled = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        if frac >= 1.0:
            parts.append(chunk)
            n_sampled += len(chunk)
            continue
        n_take = rng.binomial(len(chunk), frac)
        if n_take > 0:
            idx = rng.choice(len(chunk), size=min(n_take, len(chunk)), replace=False)
            parts.append(chunk.iloc[idx])
            n_sampled += len(idx)

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
    if len(df) > target_n:
        idx = rng.choice(len(df), size=target_n, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
    return df, n_total
