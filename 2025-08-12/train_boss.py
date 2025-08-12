"""Train a model."""

import os
import sys
import gc
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import optax
import jaxopt
import pickle
import numpy as np
import h5py as h5
from time import time
from tqdm import tqdm
from functools import partial
from typing import Optional

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


import nufft
import nmf

n_components = 64
wh_iterations = 10_000
xh_iterations = 10_000
epsilon = 1e-12
verbose_frequency = 1_000
learning_rate = 1e-3
penalty = 1e-2
overwrite = True
seed = 42
bit_precision = 64
domain_bounds = (0, jnp.pi)
# n_unique per dimension: (14, 13, 30)
#n_modes = (13, 13, 29)
n_modes = (7, 11, 15)

supported_bit_precisions = (16, 32, 64)
if bit_precision not in supported_bit_precisions:
    raise ValueError(
        f"Unsupported floating point precision: {bit_precision}. Supported values are {supported_bit_precisions}."
    )

# Memory optimization: Enable memory preallocation and limit compilation cache
jax.config.update("jax_enable_x64", bit_precision > 32)

n_gpus = jax.device_count("gpu")
print(f"Number of GPUs detected: {n_gpus}")
default_mesh = jax.make_mesh((n_gpus,), axis_names=("spectra",))
default_sharding = NamedSharding(default_mesh, PartitionSpec("spectra", None))

np.random.seed(seed)
key = jax.random.PRNGKey(seed)

input_kwds = dict(
    n_components=n_components,
    learning_rate=learning_rate,
    wh_iterations=wh_iterations,
    xh_iterations=xh_iterations,
    epsilon=epsilon,
    verbose_frequency=verbose_frequency,
    overwrite=overwrite,
    seed=seed,
    bit_precision=bit_precision,
)


with open(f"../20250812-BOSS-Grid-Test.pkl", "rb") as f:
    meta = pickle.load(f)


# OPTIONS DONE
parameter_names = meta["parameter_names"]
min_parameters = np.min(meta["parameters"], axis=0)
max_parameters = np.max(meta["parameters"], axis=0)

# Transform parameters in chunks to avoid memory explosion
print("Transforming parameters...")
x = meta["parameters"]
absorption = jnp.clip(1 - meta["flux"], 0, None)

x_transformed = (x - min_parameters) / (max_parameters - min_parameters)
x_transformed = x_transformed * (domain_bounds[1] - domain_bounds[0]) + domain_bounds[0]

V = jax.device_put(absorption, default_sharding)

n_spectra, n_pixels = V.shape
n_spectra, n_parameters = x_transformed.shape

H = jax.random.uniform(key, shape=(n_components, n_pixels), minval=epsilon, maxval=1)
W = jax.device_put(
    jax.random.uniform(key, shape=(n_spectra, n_components), minval=epsilon, maxval=1),
    default_sharding,
)
W, H, wh_losses = nmf.multiplicative_updates_WH(
    V,
    W,
    H,
    iterations=wh_iterations,
    verbose_frequency=verbose_frequency,
    epsilon=epsilon,
)

A = nufft.fourier_design_matrix(x_transformed.T, n_modes)
X = jax.vmap(jnp.linalg.lstsq, in_axes=(None, -1))(A, W)[0].T

X, H, W, xh_losses = nmf.update_XH(
    A,
    X,
    W,
    H,
    V,
    iterations=xh_iterations,
    learning_rate=learning_rate,
    epsilon=epsilon,
    verbose_frequency=verbose_frequency,
    penalty=penalty,
)

# Zero-out small values in H, as epsilon was only used for numerical stability
H = jnp.where(H <= epsilon, 0.0, H)

# Make some plots.
import matplotlib.pyplot as plt
m_h_slices = np.sort(np.unique(x[:, 0]))
L = len(m_h_slices)
K = int(np.ceil(np.sqrt(L)))
max_abs_diff = jnp.max(jnp.abs(V - W @ H), axis=1)
mean_abs_diff = jnp.mean(jnp.abs(V - W @ H), axis=1)
fig_mean, axes_mean = plt.subplots(3, 5, figsize=(5*4, 3*4), sharex=True, sharey=True)
fig_worst, axes_worst = plt.subplots(3, 5, figsize=(5*4, 3*4), sharex=True, sharey=True)

for i, m_H in enumerate(m_h_slices):

    mask = x[:, 0] == m_H
    ax_mean = axes_mean.flat[i]
    ax_worst = axes_worst.flat[i]
    scat_worst = ax_worst.scatter(
        x[mask, 2], x[mask, 1],
        c=max_abs_diff[mask],
        vmin=0, vmax=0.25
    )
    scat_mean = ax_mean.scatter(
        x[mask, 2], x[mask, 1],
        c=mean_abs_diff[mask],
        vmin=0, vmax=0.05,
    )
    for ax in (ax_worst, ax_mean):
        ax.set_title(f"$m_H = {m_H:.2f}$")
        ax.set_xlabel(parameter_names[2])
        ax.set_ylabel(parameter_names[1])

cbar = plt.colorbar(scat_mean, ax=(axes_mean.flat[L], ), orientation="vertical")
cbar.set_label("Mean absolute difference")

cbar = plt.colorbar(scat_worst, ax=(axes_worst.flat[L], ), orientation="vertical")
cbar.set_label("Max absolute difference")


for ax in axes_mean.flat[L:]:
    ax.set_visible(False)
for ax in axes_worst.flat[L:]:
    ax.set_visible(False)

fig_mean.tight_layout()
fig_worst.tight_layout()
fig_mean.savefig("boss_mean_abs_diff.png", dpi=300)
fig_worst.savefig("boss_max_abs_diff.png", dpi=300)



with open("boss_model.pkl", "wb") as fp:
    pickle.dump(dict(
        X=np.array(X),
        H=np.array(H),
        W=np.array(W),
        n_modes=n_modes,
        parameter_names= meta["parameter_names"],
        min_parameters=np.array(min_parameters),
        max_parameters=np.array(max_parameters),
        λ=meta["λ"]
    ),
    fp
    )