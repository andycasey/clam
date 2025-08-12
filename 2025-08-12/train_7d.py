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
n_modes = (5, 3, 7, 11, 15)
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

parameter_names = [pn for pn in meta["parameter_names"] if pn not in abundance_parameter_names]


# prepare absorption differences.


with open(f"{input_path_prefix}_meta.pkl", "rb") as f:
    meta = pickle.load(f)
    absorption_memmap_kwds = meta["absorption_memmap_kwds"]
    parameter_memmap_kwds = meta["parameter_memmap_kwds"]

absorption_memmap = np.memmap(f"{input_path_prefix}_absorption.memmap", mode="r", **absorption_memmap_kwds)
x_memmap = np.memmap(f"{input_path_prefix}_parameters.memmap", mode="r", **parameter_memmap_kwds)

all_parameter_names = meta["parameter_names"]

x_m_args = []
for x_m_parameter_name in ("C_m", "N_m"):

    # Slice on other parameters to be X_m = 0
    subset_mask = np.ones(x_memmap.shape[0], dtype=bool)
    for parameter_name in abundance_parameter_names:
        if parameter_name == x_m_parameter_name:
            continue
        index = all_parameter_names.index(parameter_name)
        subset_mask *= (x_memmap[:, index] == 0.0)

    # These are the ones we must match to.
    subset_has_non_zero_x_m = (
        subset_mask 
    *   (x_memmap[:, all_parameter_names.index(x_m_parameter_name)] != 0.0)
    )
    subset_has_zero_x_m = (
        subset_mask 
    *   (x_memmap[:, all_parameter_names.index(x_m_parameter_name)] == 0.0)
    )
    x = jnp.array(x_memmap)
    match_indices = tuple([all_parameter_names.index(pn) for pn in parameter_names])

    assert match_indices == (2, 3, 4, 5, 6)

    where_subset_has_zero_x_m = jnp.where(subset_has_zero_x_m)[0]

    @jax.jit
    def _get_match(i):
        diff = jnp.sum(jnp.abs(x[:, slice(2, 7)][i] - x[subset_has_zero_x_m, slice(2, 7)]), axis=1)
        index = jnp.argmin(diff)
        return jax.lax.cond(
            diff[index] > 0.01,
            lambda: jnp.array((i, -1)),  # No match found
            lambda: jnp.array((i, where_subset_has_zero_x_m[index]))
        )

    indices = jax.vmap(_get_match)(jnp.where(subset_has_non_zero_x_m)[0])

    indices = indices[indices[:, 1] >= 0]

    delta_absorption = absorption_memmap[indices[:, 0]] - absorption_memmap[indices[:, 1]]
    scaled_delta_absorption = delta_absorption / np.atleast_2d(np.abs(x_memmap[indices[:, 0], all_parameter_names.index(x_m_parameter_name)])).T


    x_indices = tuple([i for i, pn in enumerate(all_parameter_names) if pn in parameter_names or pn == x_m_parameter_name])
    delta_x_transformed = x_memmap[indices[:, 0]][:, x_indices]

    min_parameters = meta["min_parameters"]
    max_parameters = meta["max_parameters"]
    delta_x_transformed = (
        (delta_x_transformed - min_parameters[[x_indices]]) / (max_parameters[[x_indices]] - min_parameters[[x_indices]])
    ) * np.pi


    x_m_n_components = 16
    x_m_n_modes = (5, 3, 3, 3, 3, 15)
    f_modes = nufft.fourier_modes(*x_m_n_modes)
    _A = nufft.fourier_matvec(delta_x_transformed.T, f_modes)

    model = PLSRegression(n_components=x_m_n_components)
    model.fit(_A, scaled_delta_absorption)

    min_x_m = min_parameters[all_parameter_names.index(x_m_parameter_name)]
    max_x_m = max_parameters[all_parameter_names.index(x_m_parameter_name)]
    x_m_args.append((x_m_parameter_name, min_x_m, max_x_m, f_modes, model.x_mean_, model.coef_.T, model.intercept_))
    

raise a


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