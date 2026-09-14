"""Baselines de deep learning para deteccion de anomalias, entrenados sobre
la representacion de features compartida (ver features.py) -- frequency
encoding de categoricas + numericas escaladas, NO los embeddings aprendidos
por el FlowVAE propuesto.

Los tres comparten arquitectura de capacidad equivalente a la del FlowVAE
(hidden_dim/z_dim de optuna_best_params.json) para que la diferencia en
resultados sea atribuible al metodo, no a mas/menos parametros.

  - AutoencoderDetector: autoencoder determinista simple (sin muestreo, sin
    KL), entrenado con MSE. Score = error de reconstruccion (SPE-like).

  - DeepSVDDDetector: One-Class Deep SVDD (Ruff et al. 2018), version
    simplificada -- pretrain como autoencoder, centro c = media de
    encoder(train), luego fine-tuning del encoder minimizando la distancia
    al centro. Capas Linear SIN bias (si no, el modelo puede colapsar
    trivialmente aprendiendo bias = c e ignorando la entrada, el "hypersphere
    collapse" que describe el paper original).

  - VanillaVAEDetector: VAE "de manual" (An & Cho, 2015: "Variational
    Autoencoder based Anomaly Detection using Reconstruction Probability"):
    decoder HETEROCEDASTICO (aprende mu Y sigma) y beta=1 (ELBO estandar,
    sin beta-VAE). A diferencia del FlowVAE propuesto -- que fija sigma del
    decoder justamente para evitar que el decoder "infle" sigma y tolere
    cualquier error de reconstruccion sin penalidad (ver docstring de
    model.py) -- esta es la formulacion estandar de la literatura, con esa
    misma patologia potencial. Se incluye tal cual porque es el baseline que
    el revisor pidio explicitamente ("conventional VAE-based anomaly
    detection"), y sirve ademas como validacion empirica de por que el
    FlowVAE propuesto se aparta de este diseno.
    Score = reconstruction probability negativa (NLL gaussiana promediada
    sobre L muestras de z ~ q(z|x)).
"""

import numpy as np
import torch
import torch.nn as nn


def _to_tensor(x, device):
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


class _MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class AutoencoderDetector:
    def __init__(self, input_dim, hidden_dim, z_dim, lr=1e-3, epochs=300,
                 batch_size=256, device="cpu", seed=0):
        torch.manual_seed(seed)
        self.device = device
        self.encoder = _MLPEncoder(input_dim, hidden_dim, z_dim).to(device)
        self.decoder = _MLPEncoder(z_dim, hidden_dim, input_dim).to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )

    def fit(self, X_train):
        X = _to_tensor(X_train, self.device)
        n = X.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                b = perm[start:start + self.batch_size]
                x = X[b]
                x_hat = self.decoder(self.encoder(x))
                loss = ((x_hat - x) ** 2).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        return self

    @torch.no_grad()
    def score(self, X):
        x = _to_tensor(X, self.device)
        x_hat = self.decoder(self.encoder(x))
        return ((x_hat - x) ** 2).sum(dim=-1).cpu().numpy()

    @torch.no_grad()
    def encode(self, X):
        """Bottleneck z -- para el T2 de Hotelling del AE estandar (ver
        baselines/ae_control_chart.py), analogo al z_loc del FlowVAE."""
        x = _to_tensor(X, self.device)
        return self.encoder(x).cpu().numpy()


class DeepSVDDDetector:
    def __init__(self, input_dim, hidden_dim, z_dim, lr=1e-3,
                 pretrain_epochs=150, finetune_epochs=150, batch_size=256,
                 device="cpu", seed=0, eps=0.1):
        torch.manual_seed(seed)
        self.device = device
        self.encoder = _MLPEncoder(input_dim, hidden_dim, z_dim, bias=False).to(device)
        self.decoder = _MLPEncoder(z_dim, hidden_dim, input_dim, bias=False).to(device)
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.eps = eps
        self.center = None

    def _pretrain(self, X):
        n = X.shape[0]
        opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-3
        )
        for _ in range(self.pretrain_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                b = perm[start:start + self.batch_size]
                x = X[b]
                x_hat = self.decoder(self.encoder(x))
                loss = ((x_hat - x) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    @torch.no_grad()
    def _init_center(self, X):
        c = self.encoder(X).mean(dim=0)
        # evita colapso trivial a c=0 (ver Ruff et al. 2018, seccion 4.1)
        c[(c.abs() < self.eps) & (c >= 0)] = self.eps
        c[(c.abs() < self.eps) & (c < 0)] = -self.eps
        return c

    def fit(self, X_train):
        X = _to_tensor(X_train, self.device)
        self._pretrain(X)
        self.center = self._init_center(X)

        n = X.shape[0]
        opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-3, weight_decay=1e-6)
        for _ in range(self.finetune_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                b = perm[start:start + self.batch_size]
                x = X[b]
                z = self.encoder(x)
                loss = ((z - self.center) ** 2).sum(dim=-1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, X):
        x = _to_tensor(X, self.device)
        z = self.encoder(x)
        return ((z - self.center) ** 2).sum(dim=-1).cpu().numpy()


class _VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class _VAEDecoder(nn.Module):
    """Decoder heterocedastico: aprende mu Y log-varianza (a diferencia del
    Decoder de sigma fija en model.py)."""

    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.net(z)
        return self.mu(h), self.logvar(h)


class VanillaVAEDetector:
    def __init__(self, input_dim, hidden_dim, z_dim, lr=1e-3, epochs=300,
                 batch_size=256, device="cpu", seed=0, n_mc_samples=10):
        torch.manual_seed(seed)
        self.device = device
        self.encoder = _VAEEncoder(input_dim, hidden_dim, z_dim).to(device)
        self.decoder = _VAEDecoder(z_dim, hidden_dim, input_dim).to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_mc_samples = n_mc_samples
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )

    @staticmethod
    def _gaussian_nll(x, mu, logvar):
        var = torch.exp(logvar)
        return 0.5 * (logvar + (x - mu) ** 2 / var + np.log(2 * np.pi))

    def fit(self, X_train):
        X = _to_tensor(X_train, self.device)
        n = X.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                b = perm[start:start + self.batch_size]
                x = X[b]
                z_mu, z_logvar = self.encoder(x)
                std = torch.exp(0.5 * z_logvar)
                z = z_mu + std * torch.randn_like(std)
                x_mu, x_logvar = self.decoder(z)

                recon_nll = self._gaussian_nll(x, x_mu, x_logvar).sum(dim=-1)
                kl = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp()).sum(dim=-1)
                loss = (recon_nll + kl).mean()  # beta=1, ELBO estandar

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        return self

    @torch.no_grad()
    def score(self, X):
        """-log p(x) aproximada (reconstruction probability negativa,
        promediada sobre L muestras de z ~ q(z|x)). Mayor score = mas
        anomalo (menor probabilidad de reconstruccion), mismo signo que el
        resto de los detectores."""
        x = _to_tensor(X, self.device)
        z_mu, z_logvar = self.encoder(x)
        std = torch.exp(0.5 * z_logvar)
        nlls = []
        for _ in range(self.n_mc_samples):
            z = z_mu + std * torch.randn_like(std)
            x_mu, x_logvar = self.decoder(z)
            nlls.append(self._gaussian_nll(x, x_mu, x_logvar).sum(dim=-1))
        return torch.stack(nlls, dim=0).mean(dim=0).cpu().numpy()


class NeuralNetBaseline:
    """Red neuronal SUPERVISADA (MLP, BCE) -- mismo protocolo que
    RandomForestBaseline/XGBoostBaseline/LightGBMBaseline: entrena con
    benigno (label 0) + muestra de ataque etiquetada (label 1), evaluada
    siempre sobre filas held-out. Misma capacidad (hidden_dim) que los demas
    baselines de deep learning, con una capa oculta extra para dar mas
    poder de clasificacion supervisada."""

    def __init__(self, input_dim, hidden_dim, lr=1e-3, epochs=200, batch_size=256,
                 device="cpu", seed=0):
        torch.manual_seed(seed)
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        ).to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, X_benign, X_attack_train):
        X = np.concatenate([X_benign, X_attack_train], axis=0)
        y = np.concatenate([np.zeros(len(X_benign)), np.ones(len(X_attack_train))])
        X_t = _to_tensor(X, self.device)
        y_t = _to_tensor(y, self.device)
        # class_weight="balanced" equivalente para BCEWithLogitsLoss
        n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
        pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n = X_t.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                b = perm[start:start + self.batch_size]
                logits = self.net(X_t[b]).squeeze(-1)
                loss = loss_fn(logits, y_t[b])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        return self

    @torch.no_grad()
    def predict_anomaly(self, X):
        return self.score(X) > 0.5

    @torch.no_grad()
    def score(self, X):
        x = _to_tensor(X, self.device)
        return torch.sigmoid(self.net(x).squeeze(-1)).cpu().numpy()
