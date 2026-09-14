"""Baselines de scikit-learn sobre la misma representacion de features
compartida (ver features.py): frequency encoding de categoricas + numericas
escaladas.

  - IsolationForestDetector / OneClassSVMDetector / LOFDetector /
    PCADetector: no-supervisados, entrenados SOLO con trafico benigno
    (train.npz), mismo protocolo que el FlowVAE.

  - LOFDetector agrega densidad local (Local Outlier Factor, Breunig et al.
    2000) -- a diferencia de Isolation Forest/OCSVM/Autoencoder, que miden
    "rareza global" respecto de toda la nube de benigno, LOF mide rareza
    RELATIVA a la densidad del vecindario mas cercano. Utilizado como
    detector de novedad (novelty=True: fit solo en benigno, score sobre
    datos nuevos).

  - PCADetector agrega un baseline lineal clasico (SPE sobre el
    subespacio de componentes principales, Jackson & Mudholkar 1979) --
    la misma idea de "error de reconstruccion" que el FlowVAE/Autoencoder,
    pero con una proyeccion LINEAL en vez de una red neuronal, para separar
    cuanto de la ganancia (si la hay) viene de la no-linealidad.

  - RandomForestBaseline: a diferencia de todo lo anterior, es SUPERVISADO
    (requiere ejemplos de ataque etiquetados durante el entrenamiento). Se
    incluye como techo de referencia -- no es una comparacion justa contra
    metodos no-supervisados, y se reporta por separado, evaluado siempre
    sobre filas de ataque que NO participaron de su entrenamiento.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class IsolationForestDetector:
    def __init__(self, n_estimators=200, seed=0):
        self.clf = IsolationForest(n_estimators=n_estimators, random_state=seed)

    def fit(self, X_train):
        self.clf.fit(X_train)
        return self

    def score(self, X):
        # score_samples: mayor = mas normal -> se invierte para que mayor = mas anomalo
        return -self.clf.score_samples(X)


class OneClassSVMDetector:
    def __init__(self, nu=0.05, gamma="scale"):
        self.clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)

    def fit(self, X_train):
        self.clf.fit(X_train)
        return self

    def score(self, X):
        # decision_function: mayor = mas normal -> se invierte
        return -self.clf.decision_function(X)


class LOFDetector:
    """Local Outlier Factor en modo novedad (Breunig et al. 2000): fit SOLO
    sobre benigno de train, score = densidad local relativa al vecindario
    mas cercano (a diferencia de Isolation Forest/OCSVM, que miden rareza
    global respecto de toda la nube)."""

    def __init__(self, n_neighbors=20):
        self.clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)

    def fit(self, X_train):
        self.clf.fit(X_train)
        return self

    def score(self, X):
        # score_samples: mayor = mas normal -> se invierte
        return -self.clf.score_samples(X)


class PCADetector:
    """Baseline lineal clasico de deteccion de anomalias: SPE (error de
    reconstruccion) sobre el subespacio de componentes principales
    (Jackson & Mudholkar 1979) ajustado SOLO sobre benigno de train.
    n_components se fija para explicar ~95% de la varianza de train."""

    def __init__(self, variance_threshold=0.95, seed=0):
        self.variance_threshold = variance_threshold
        self.pca = PCA(n_components=variance_threshold, svd_solver="full", random_state=seed)

    def fit(self, X_train):
        self.pca.fit(X_train)
        return self

    def score(self, X):
        X_proj = self.pca.inverse_transform(self.pca.transform(X))
        return ((X - X_proj) ** 2).sum(axis=1)


class RandomForestBaseline:
    """Baseline SUPERVISADO: entrena con benigno (label 0) + una muestra de
    ataque etiquetada (label 1) pooled de todas las familias. Se evalua
    SIEMPRE sobre filas de ataque held-out (no vistas en fit)."""

    def __init__(self, n_estimators=300, seed=0):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                           class_weight="balanced", n_jobs=-1)

    def fit(self, X_benign, X_attack_train):
        X = np.concatenate([X_benign, X_attack_train], axis=0)
        y = np.concatenate([np.zeros(len(X_benign)), np.ones(len(X_attack_train))])
        self.clf.fit(X, y)
        return self

    def predict_anomaly(self, X):
        return self.clf.predict(X).astype(bool)

    def score(self, X):
        # probabilidad de clase "ataque" -- score continuo para AUC/umbral best-F1
        return self.clf.predict_proba(X)[:, 1]


class XGBoostBaseline:
    """Baseline SUPERVISADO (gradient boosting, xgboost). Mismo protocolo que
    RandomForestBaseline: benigno (label 0) + muestra de ataque etiquetada
    (label 1), evaluado siempre sobre filas held-out."""

    def __init__(self, n_estimators=300, seed=0):
        from xgboost import XGBClassifier
        self.clf = XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
                                  random_state=seed, eval_metric="logloss", n_jobs=-1)

    def fit(self, X_benign, X_attack_train):
        X = np.concatenate([X_benign, X_attack_train], axis=0)
        y = np.concatenate([np.zeros(len(X_benign)), np.ones(len(X_attack_train))])
        # class_weight no existe en XGBClassifier -- se compensa el desbalance con scale_pos_weight
        n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
        self.clf.set_params(scale_pos_weight=(n_neg / n_pos if n_pos > 0 else 1.0))
        self.clf.fit(X, y)
        return self

    def predict_anomaly(self, X):
        return self.clf.predict(X).astype(bool)

    def score(self, X):
        return self.clf.predict_proba(X)[:, 1]


class LightGBMBaseline:
    """Baseline SUPERVISADO (gradient boosting, lightgbm). Mismo protocolo
    que RandomForestBaseline/XGBoostBaseline."""

    def __init__(self, n_estimators=300, seed=0):
        from lightgbm import LGBMClassifier
        self.clf = LGBMClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
                                   random_state=seed, class_weight="balanced", n_jobs=-1, verbose=-1)

    def fit(self, X_benign, X_attack_train):
        X = np.concatenate([X_benign, X_attack_train], axis=0)
        y = np.concatenate([np.zeros(len(X_benign)), np.ones(len(X_attack_train))])
        self.clf.fit(X, y)
        return self

    def predict_anomaly(self, X):
        return self.clf.predict(X).astype(bool)

    def score(self, X):
        return self.clf.predict_proba(X)[:, 1]
