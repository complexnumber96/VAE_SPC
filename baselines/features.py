"""Representacion de features compartida por TODOS los baselines (Isolation
Forest, One-Class SVM, Autoencoder estandar, Deep SVDD, VAE convencional,
Random Forest).

A diferencia del FlowVAE propuesto -- que aprende una tabla de embeddings por
columna categorica jointly con el resto del modelo -- estos baselines usan
frequency encoding (indice categorico -> frecuencia en el train benigno) mas
el mismo MinMaxScaler ya ajustado (processed/scaler.pkl) para las numericas.
Esto evita:
  - Reusar una representacion APRENDIDA POR EL PROPIO MODELO que se esta
    comparando (sesgaria la comparacion a favor del FlowVAE).
  - El blowup de one-hot en columnas de cardinalidad enorme (dst_ip, con
    ~11.3M valores unicos en el corpus).

El indice 0 (desconocido/no visto, ver preprocess.py) mapea naturalmente a
la frecuencia de "desconocido" en train -- tipicamente muy baja o cero, ya
que el vocabulario se construyo sobre el corpus benigno completo antes del
split train/test. Una categoria que solo aparece en trafico de ataque cae en
ese mismo bucket y por lo tanto recibe un valor de feature cercano a 0, una
señal de "rareza" razonable para un encoding no-aprendido.
"""

import numpy as np


def build_freq_tables(cat_arrays_train, vocab_sizes, categorical_cols):
    """Tabla de frecuencias por columna, ajustada SOLO sobre indices de train
    benigno (cat_arrays_train: dict col -> array de indices int, como los
    guardados en processed/train.npz)."""
    freq_tables = {}
    for c in categorical_cols:
        counts = np.bincount(cat_arrays_train[c], minlength=vocab_sizes[c]).astype(np.float64)
        total = counts.sum()
        freq_tables[c] = (counts / total if total > 0 else counts).astype(np.float32)
    return freq_tables


def encode_features(cat_arrays, num_array, freq_tables, categorical_cols):
    """cat_arrays: dict col -> array de indices int (ya mapeados via
    vocab_maps, como hace control_charts.common.transform_dataframe).
    Devuelve un array (n, len(categorical_cols) + n_numeric) float32."""
    cat_freq = np.stack(
        [freq_tables[c][cat_arrays[c]] for c in categorical_cols], axis=1
    ).astype(np.float32)
    return np.concatenate([cat_freq, num_array.astype(np.float32)], axis=1)
