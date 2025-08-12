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

input_path_prefix = "../korg_grid_subset_7d"
n_components = 64
wh_iterations = 100_000
xh_iterations = 100_000
epsilon = 1e-12
verbose_frequency = 1_000
learning_rate = 1e-3
penalty = 1e-2
overwrite = True
seed = 42
bit_precision = 64
domain_bounds = (0, jnp.pi)
n_modes = (3, 7, 11, 15)
abundance_parameter_names = ("N_m", "C_m")

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
    input_path_prefix=input_path_prefix,
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


with open(f"{input_path_prefix}_meta.pkl", "rb") as f:
    meta = pickle.load(f)
    absorption_memmap_kwds = meta["absorption_memmap_kwds"]
    parameter_memmap_kwds = meta["parameter_memmap_kwds"]


# OPTIONS DONE
absorption_memmap = np.memmap(
    f"{input_path_prefix}_absorption.memmap", mode="r", **absorption_memmap_kwds
)[:-1]

x_memmap = np.memmap(
    f"{input_path_prefix}_parameters.memmap", mode="r", **parameter_memmap_kwds
)[:-1]
print(f"Absorption shape: {absorption_memmap.shape}")

# Sub-select on the absoprtion dimensions that are zero [C/M] = [N/M] = [alpha/M] = 0
x_mask = np.ones(x_memmap.shape[1], dtype=bool)
absorption_mask = np.ones(x_memmap.shape[0], dtype=bool)
for parameter_name in abundance_parameter_names:
    index = meta["parameter_names"].index(parameter_name)
    x_mask[index] = False
    absorption_mask &= (x_memmap[:, index] == 0)

# Memory optimization: Load parameters without keeping full copy in memory
min_parameters, max_parameters = (meta["min_parameters"], meta["max_parameters"])

# Transform parameters in chunks to avoid memory explosion
print("Transforming parameters...")
x_transformed = (x_memmap - min_parameters) / (max_parameters - min_parameters)
x_transformed = x_transformed * (domain_bounds[1] - domain_bounds[0]) + domain_bounds[0]

del x_memmap

print("Apply mask")
x_transformed = x_transformed[absorption_mask][:, x_mask]
absorption_memmap = absorption_memmap[absorption_mask, :]

print("Shapes after masking:")
print(f"x_transformed: {x_transformed.shape}")
print(f"absorption_memmap: {absorption_memmap.shape}")

V = jax.device_put(absorption_memmap, default_sharding)

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

with open("model.pkl", "wb") as fp:
    pickle.dump(dict(
        X=np.array(X),
        H=np.array(H),
        W=np.array(W),
        n_modes=n_modes,
        parameter_names= meta["parameter_names"],
        min_parameters=np.array(min_parameters),
        max_parameters=np.array(max_parameters),
    ),
    fp
    )