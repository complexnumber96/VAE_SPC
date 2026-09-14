"""Beta-VAE en Pyro (SVI/Trace_ELBO), arquitectura simple de una capa oculta
por encoder/decoder -- siguiendo el diseño ya validado del usuario en su
propio notebook (modelo_final.ipynb: Encoder/Decoder de una capa, beta-KL via
pyro.poutine.scale).

Decoder de varianza FIJA (no aprendida): una busqueda de hiperparametros con
Optuna mostro que un decoder heterocedastico (que aprende mu Y sigma) puede
hacer "trampa" contra el -ELBO combinando beta alto + z_dim/hidden_dim chicos
-- el encoder colapsa el posterior (ignora la entrada) y el decoder infla
sigma para que la verosimilitud tolere el error de reconstruccion sin
penalidad. El resultado es un modelo que no distingue flujos benignos de
anomalos (justamente lo que este VAE necesita hacer). Con sigma fija, la
unica forma de bajar la loss es que mu se parezca a x, lo que fuerza a usar
z de verdad.

Las columnas categoricas se siguen embebiendo por columna (CategoricalEmbedder,
jointly-trained con el resto del modelo) y se concatenan con las numericas
escaladas para formar un unico vector plano de entrada -- el equivalente al
"features_array" del notebook original. El encoder/decoder generico reconstruye
ESE vector completo con una unica verosimilitud Normal heterocedastica, en vez
de tener una cabeza de clasificacion por columna categorica.

Ventaja practica de este diseño: como el decoder ya no clasifica cada columna
categorica por separado, la cardinalidad de columnas como dst_ip (~11.3M
valores unicos) deja de ser un problema para el decoder (antes pedia una
Linear(hidden_dim, 11.3M), ahora el decoder solo reconstruye el vector de
tamano fijo input_dim*2). Lo unico que sigue requiriendo cuidado es el lado
del ENCODER: esas mismas columnas de cardinalidad grande usan embeddings
sparse (ver sparse_cols) para que el optimizador no tenga que actualizar la
tabla entera en cada paso.

Nota de diseño: como el decoder reconstruye directamente el vector de
embeddings (no los indices categoricos originales), existe en teoria el
riesgo de que el modelo colapse los embeddings a un valor casi constante para
"resolver" la reconstruccion trivialmente. En la practica esto no suele
ocurrir con SGD/Adam sin una presion explicita hacia ese minimo, y es el
mismo patron que ya usa el usuario en produccion.
"""

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn


class CategoricalEmbedder(nn.Module):
    def __init__(self, categorical_cols, vocab_sizes, embedding_dims, sparse_cols=()):
        super().__init__()
        self.categorical_cols = categorical_cols
        self.sparse_cols = set(sparse_cols)
        self.embeddings = nn.ModuleDict({
            c: nn.Embedding(vocab_sizes[c], embedding_dims[c], sparse=(c in self.sparse_cols))
            for c in categorical_cols
        })
        self.output_dim = sum(embedding_dims[c] for c in categorical_cols)

    def forward(self, x_cat):
        return torch.cat([self.embeddings[c](x_cat[c]) for c in self.categorical_cols], dim=-1)


class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim * 2)
        )

    def forward(self, x):
        h = self.encoder(x)
        z_mu, z_logvar = torch.chunk(h, 2, dim=-1)
        return z_mu, torch.nn.functional.softplus(z_logvar)


class Decoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, z):
        return self.decoder(z)


class FlowVAE(nn.Module):
    def __init__(self, categorical_cols, numeric_cols, vocab_sizes, embedding_dims,
                 hidden_dim=50, z_dim=10, beta=1.1, sparse_cols=(), decoder_sigma=1.0):
        super().__init__()
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.n_numeric = len(numeric_cols)
        self.z_dim = z_dim
        self.beta = beta
        self.decoder_sigma = decoder_sigma

        # columnas de cardinalidad grande (dst_ip, src_port): embeddings sparse
        # para que Adam no actualice la tabla entera en cada paso. Siguen
        # siendo parte del modelo -- ninguna variable se descarta.
        self.embedder = CategoricalEmbedder(categorical_cols, vocab_sizes, embedding_dims,
                                             sparse_cols=sparse_cols)
        input_dim = self.embedder.output_dim + self.n_numeric
        self.input_dim = input_dim  # dim del vector plano (features_array): embeddings + numericas escaladas

        self.encoder = Encoder(input_dim, z_dim, hidden_dim)
        self.decoder = Decoder(input_dim, z_dim, hidden_dim)

    def parameter_groups(self):
        """Separa los pesos de embeddings sparse (necesitan SparseAdam) del
        resto de parametros (Adam denso normal)."""
        sparse_params = [self.embedder.embeddings[c].weight for c in self.embedder.sparse_cols]
        sparse_ids = {id(p) for p in sparse_params}
        dense_params = [p for p in self.parameters() if id(p) not in sparse_ids]
        return dense_params, sparse_params

    def assemble(self, x_cat, x_num):
        """Arma el vector plano de entrada (equivalente a 'features_array')."""
        x_embedded = self.embedder(x_cat)
        return torch.cat([x_embedded, x_num], dim=-1)

    def model(self, x_cat, x_num):
        pyro.module("decoder", self.decoder)
        x = self.assemble(x_cat, x_num)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(x.shape[0], self.z_dim)
            z_scale = x.new_ones(x.shape[0], self.z_dim)
            with pyro.poutine.scale(scale=self.beta):
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            mu = self.decoder(z)
            sigma = x.new_full(x.shape, self.decoder_sigma)
            pyro.sample("observed", dist.Normal(mu, sigma).to_event(1), obs=x)

    def guide(self, x_cat, x_num):
        pyro.module("embedder", self.embedder)
        pyro.module("encoder", self.encoder)
        x = self.assemble(x_cat, x_num)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    @torch.no_grad()
    def reconstruction_error(self, x_cat, x_num):
        """Error de reconstruccion (MSE) del vector plano completo, usando la
        media aproximada de q(z|x). Util para deteccion de anomalias."""
        x = self.assemble(x_cat, x_num)
        z_loc, _ = self.encoder(x)
        mu = self.decoder(z_loc)
        return ((mu - x) ** 2).mean(dim=-1)
