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

import nufft
import nmf

input_path_prefix = "../korg_grid_subset_4d"
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

supported_bit_precisions = (16, 32, 64)
if bit_precision not in supported_bit_precisions:
    raise ValueError(
        f"Unsupported floating point precision: {bit_precision}. Supported values are {supported_bit_precisions}."
    )

# Memory optimization: Enable memory preallocation and limit compilation cache
jax.config.update("jax_enable_x64", bit_precision > 32)
#jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

n_gpus = jax.device_count("gpu")
print(f"Number of GPUs detected: {n_gpus}")
default_mesh = jax.make_mesh((n_gpus,), axis_names=("spectra",))
default_sharding = NamedSharding(default_mesh, PartitionSpec("spectra", None))

np.random.seed(seed)
key = jax.random.PRNGKey(seed)

print(os.path.abspath(__file__))

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

n_total_modes = np.prod(n_modes)


absorption_memmap = np.memmap(
    f"{input_path_prefix}_absorption.memmap", mode="r", **absorption_memmap_kwds
)[:-1]

x_memmap = np.memmap(
    f"{input_path_prefix}_parameters.memmap", mode="r", **parameter_memmap_kwds
)[:-1]
print(f"Absorption shape: {absorption_memmap.shape}")

# Memory optimization: Load parameters without keeping full copy in memory
min_parameters, max_parameters = (meta["min_parameters"], meta["max_parameters"])

# Transform parameters in chunks to avoid memory explosion
print("Transforming parameters...")
x_transformed = (x_memmap - min_parameters) / (max_parameters - min_parameters)
x_transformed = x_transformed * (domain_bounds[1] - domain_bounds[0]) + domain_bounds[0]

del x_memmap

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



# TRAINING ENDS
# TEST BEGINS

max_vsini = 200.0  # km/s

parameter_names = list(meta["parameter_names"]) + ["vsini"]
min_labels = jnp.array(meta["min_parameters"])
max_labels = jnp.array(meta["max_parameters"])


import optimistix
import matplotlib.pyplot as plt

from astropy.io import fits
from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.utils.grids import velocity_grid
from model import continuum_design_matrix

vr_array = velocity_grid(22_500, max_vsini)



require_even_n_pixels = True
with fits.open("../mwmStar-0.6.0-0.fits") as image:
    hdu = 3
    λ = image[hdu].data["wavelength"][0]
    flux = image[hdu].data["flux"][0].reshape((1, 8575))
    ivar = image[hdu].data["ivar"][0].reshape((1, 8575))

if require_even_n_pixels and flux.shape[1] % 2 == 1:
    flux = flux[:, :-1]
    ivar = ivar[:, :-1]
    λ = λ[:-1]
    H = H[:, :-1]

flux, ivar = jnp.array(flux), jnp.array(ivar)
n_stellar_parameters = len(parameter_names)

A = continuum_design_matrix(λ, n_modes=7)
f_modes = nufft.fourier_modes(*n_modes)



def create_periodic_transform_functions(min_parameters, max_parameters, min_domain=0, max_domain=jnp.pi):
    """ Create functions to transform parameters to the domain [0, pi] and back. """
    
    def to_domain(θ):
        return ((θ - min_parameters) / (max_parameters - min_parameters)) * (max_domain - min_domain) + min_domain

    domain_edge = jnp.pi - 0.5 * jnp.abs(max_domain - min_domain)

    def from_domain(θ):
        θ %= 2 * jnp.pi
        θ = jnp.where(θ > (max_domain + domain_edge), θ - 2 * jnp.pi, θ) # wrap around
        θ = (θ - min_domain) / (max_domain - min_domain)
        return θ * (max_parameters - min_parameters) + min_parameters

    return (to_domain, from_domain)

def create_vsini_transform_functions(min_vsini=0.0, max_vsini=200.0):
    """ Create functions to transform vsini to the domain [0, 1] and back. """

    def to_domain(θ):
        return (θ - min_vsini) / (max_vsini - min_vsini)
    
    def from_domain(θ):
        return jnp.abs(θ) * (max_vsini - min_vsini) + min_vsini
    
    return (to_domain, from_domain)
    

to_label_domain, from_label_domain = create_periodic_transform_functions(min_labels, max_labels, *domain_bounds)
to_vsini_domain, from_vsini_domain = create_vsini_transform_functions(max_vsini=max_vsini)

def to_domain(θ):
    """Transform parameters to the optimisation domain."""
    return jnp.hstack([to_label_domain(θ[:-1]), to_vsini_domain(θ[-1])])

def from_domain(θ):
    """Transform parameters from the optimisation domain to the original parameter space."""
    return jnp.hstack([from_label_domain(θ[:-1]), from_vsini_domain(θ[-1])])


def emulate_spectrum(θ, f_modes, X, H):
    A = nufft.fourier_matvec(θ.reshape((4, -1)), f_modes)
    W = nmf.basis_weights(A, X, epsilon=0)
    return (1 - W @ H).flatten()

def predict_spectrum(θ, vr_array, A, f_modes, X, H):
    continuum = θ[:-n_stellar_parameters] @ A.T
    rectified_flux = emulate_spectrum(θ[-n_stellar_parameters:-1], f_modes, X, H)
    vsini = from_vsini_domain(θ[-1])
    broadened_flux = convolve_rigid_rotation(rectified_flux.flatten(), vr_array, vsini)
    return (broadened_flux, continuum)

def residual(θ, args):
    flux, ivar, vr_array, A, f_modes, X, H = args
    rectified_flux, continuum = predict_spectrum(θ, vr_array, A, f_modes, X, H)
    model_flux = rectified_flux * continuum
    return (model_flux - flux) * jnp.sqrt(ivar)

def nll_spectrum(θ, flux, ivar, vr_array, A, f_modes, X, H):
    return 0.5 * jnp.sum(jnp.square(residual(θ, (flux, ivar, vr_array, A, f_modes, X, H))))

def approximate_vmic(logg):
    coeffs = jnp.array([0.372160, -0.090531, -0.000802, 0.001263, -0.027321])
    DM = jnp.array([1, logg, logg**2, logg**3, 0])
    return jnp.clip(10**(DM @ coeffs), 0.32, 5.0)


@jax.jit
def fit_spectrum(flux, ivar, initial_parameters, vr_array, A, f_modes, X, H):
    θ_stellar_0 = to_domain(initial_parameters)

    initial_model_rectified_flux = emulate_spectrum(θ_stellar_0[:-1], f_modes, X, H).flatten()
    pseudo_rectified_flux = flux / initial_model_rectified_flux
    pseudo_rectified_ivar = ivar * initial_model_rectified_flux**2
    
    use = (ivar > 0) * jnp.any(A != 0, axis=1)
    Y = jnp.where(use, pseudo_rectified_flux, 0.0)
    C_inv = jnp.diag(jnp.where(use, pseudo_rectified_ivar, 0.0))

    θ_continuum_0, *extras = jnp.linalg.lstsq(A.T @ C_inv @ A, A.T @ C_inv @ Y, rcond=None)
    θ_1 = jnp.hstack([θ_continuum_0, θ_stellar_0])

    args = (vr_array, A, f_modes, X, H)
    r = optimistix.compat.minimize(
        nll_spectrum, 
        θ_1, 
        args=(flux, ivar, *args), 
        method="bfgs", 
        options=dict(maxiter=10_000)
    )
    r = optimistix.least_squares(
        residual, 
        args=(flux, ivar, *args),
        solver=optimistix.LevenbergMarquardt(rtol=1e-8, atol=1e-8),
        y0=r.x,
        max_steps=256,
        tags={},
        throw=False
    )
    rectified_flux, continuum = predict_spectrum(r.value, *args)
    initial_rectified_flux, initial_continuum = predict_spectrum(θ_1, *args)
    return (r.value, rectified_flux, continuum, initial_rectified_flux, initial_continuum, initial_model_rectified_flux, r.result)
    

args = (
    vr_array,
    jnp.array(A), 
    jnp.array(f_modes), 
    jnp.array(X), 
    jnp.array(H), 
)


initial_parameters_dict=dict(Teff=5750, logg=4.3, m_H=0.0, vmic=0.8, vsini=20.0)
initial_parameters = jnp.array([initial_parameters_dict[p] for p in parameter_names])
t_solar = -time()
solar = fit_spectrum(flux[0], ivar[0], initial_parameters, *args)  # Test the function to ensure it works
t_solar += time()

print(f"Fitting solar spectrum took {t_solar:.2f} seconds.")

t_solar = -time()
solar = fit_spectrum(flux[0], ivar[0], initial_parameters, *args)  # Test the function to ensure it works
t_solar += time()

print(f"Fitting solar spectrum took {t_solar:.2f} seconds.")

solar_parameters = dict(zip(parameter_names, from_domain(solar[0][-n_stellar_parameters:])))

for key, value in solar_parameters.items():
    print(f"{key}: {value:.4f}")

fig, axes = plt.subplots(3, 1, figsize=(20, 10))

for i, ax in enumerate(axes.flat):
    ax.plot(λ, flux[0] / solar[2], c="k", label="Data")
    ax.plot(
        λ,
        solar[1],
        c="tab:red",
        label=f"Model: Teff={solar_parameters['Teff']:.0f} K logg={solar_parameters['logg']:.2f} [M/H]={solar_parameters['m_H']:.2f} vmic={solar_parameters['vmic']:.2f} km/s vsini={solar_parameters['vsini']:.1f} km/s",
    )
    ax.plot(
        λ, 
        solar[3], 
        c="tab:blue", 
        label=f"Initial model: " + (" ".join([f"{k}={v:.2f}" for k, v in initial_parameters_dict.items()])),
        zorder=-1,
        )
    ax.set_ylim(0, 1.2)

regions = [(15120.0, 15820.0), (15840.0, 16440.0), (16450.0, 16960.0)]
for region, ax in zip(regions, axes.flat):
    ax.set_xlim(region)

axes[0].legend(frameon=False, loc="lower left")
fig.savefig("solar.png", dpi=300, bbox_inches="tight")



# Test clusters

with open("../20240517_spectra.pkl", "rb") as fp:
    flux, ivar, all_meta = pickle.load(fp)
    λ = 10**(4.179 + 6e-6 * np.arange(flux.shape[1]))

if require_even_n_pixels and flux.shape[1] % 2 == 1:
    flux = flux[:, :-1]
    ivar = ivar[:, :-1]
    λ = λ[:-1]

flux, ivar = jnp.array(flux), jnp.array(ivar)

from astropy.io import fits
image = fits.open("../astraAllStarApogeeNet-0.6.0.fits")


initial_parameters = []
for row in all_meta:
    match = np.where(image[2].data["sdss_id"] == row["sdss_id"])[0]    
    match = match[np.argmax(image[2].data["snr"][match])]

    d = dict(
        Teff=np.clip(image[2].data["teff"][match], 3000, 8000),
        logg=np.clip(image[2].data["logg"][match], 0, 5.5),
        m_H=np.clip(image[2].data["fe_h"][match], -2.5, 1.0),
        vsini=20.0,
    )
    d["vmic"] = approximate_vmic(d["logg"])
    initial_parameters.append([d[k] for k in parameter_names])

initial_parameters = jnp.array(initial_parameters)

t_fitting = -time()
results = []
for i in tqdm(range(flux.shape[0])):
    results.append(fit_spectrum(flux[i], ivar[i], initial_parameters[i], *args))
t_fitting += time()

print(f"Time to fit {n} spectra: {t_fitting:.2f} seconds.")

stellar_parameters = jnp.array([from_domain(r[0][-n_stellar_parameters:]) for r in results])

rectified_flux = jnp.array([r[1] for r in results])
continuum = jnp.array([r[2] for r in results])
initial_rectified_flux = jnp.array([r[3] for r in results])
initial_continuum = jnp.array([r[4] for r in results])

rchi2 = jnp.sum((rectified_flux * continuum - flux)**2 * ivar, axis=1) / 8575
log2_flag = {
    0: "M92",
    1: "M15",
    2: "M53",
    3: "NGC 5466",
    4: "NGC 4147",
    5: "M2",
    6: "M13",
    7:  "M3",
    8:  "M5",
    9:  "M12",
    10: "M107",
    11: "M71",
    12: "NGC 2243",
    13: "Be29",
    14:  "NGC 2158",
    15:  "M35",
    16:  "NGC 2420",
    17: "NGC 188",
    18: "M67",
    19: "NGC 7789",
    20: "Pleiades",
    21: "NGC 6819",
    22: "Coma Berenices",
    23: "NGC 6791",
    24: "NGC 5053",
    25: "M68",
    26: "NGC 6397",
    27: "M55",
    28: "NGC 5634",
    29: "M22",
    30: "M79",
    31: "NGC 3201",
    32: "M10",
    33:"NGC 6752",
    34: "Omega Centauri",
    35: "M54",
    36: "NGC 6229",
    37: "Pal5",
    38: "NGC 6544",
    39: "NGC 6522",
    40: "NGC 288",
    41: "NGC 362",
    42: "NGC 1851",
    43: "M4",
    44: "NGC 2808",
    45: "Pal6",
    46: "47 Tucane",
    47: "Pal1",
    48: "NGC 6539",
    49: "NGC 6388",
    50: "NGC 6441",
    51: "NGC 6316",
    52: "NGC 6760",
    53: "NGC 6553",
    54: "NGC 6528",
    55: "Draco",
    56: "Ursa Minor",
    57: "Bootes 1",
    58: "Sextans",
    59: "Fornax",
    60: "Sculptor",
    61: "Carina",
}
import h5py as h5
with h5.File("20250729_cluster_results_bfgs+lm.h5", "w") as fp:
    group = fp.create_group("results")

    for i, parameter_name in enumerate(list(parameter_names)):
        group.create_dataset(parameter_name, data=stellar_parameters[:, i])

    group.create_dataset("initial_teff", data=np.array(initial_parameters[:, -2]))
    group.create_dataset("initial_logg", data=np.array(initial_parameters[:, -3]))
    group.create_dataset("initial_m_h", data=np.array(initial_parameters[:, -4]))
    group.create_dataset("initial_vmic", data=np.array(initial_parameters[:, 0]))
    
    group.create_dataset("rchi2", data=rchi2)

    group.create_dataset("sdss4_apogee_member_flags", data=np.hstack([m.get("sdss4_apogee_member_flags", 0) for m in all_meta]))
    group.create_dataset("snr", data=np.hstack([m.get("snr", np.nan) for m in all_meta]))
    group.create_dataset("comp_teff", data=np.hstack([m.get("Teff", np.nan) for m in all_meta]))
    group.create_dataset("comp_logg", data=np.hstack([m.get("logg", np.nan) for m in all_meta]))
    group.create_dataset("comp_vmic", data=np.hstack([m.get("vmic", np.nan) for m in all_meta]))
    group.create_dataset("comp_m_h", data=np.hstack([m.get("m_h", np.nan) for m in all_meta]))
    group.create_dataset("sdss_id", data=np.hstack([r.get("sdss_id", -1) for r in all_meta]))
    for bit, cluster_name in log2_flag.items():
        group.create_dataset(cluster_name.replace(" ", "_").lower(), data=np.hstack([int(np.log2(m.get("sdss4_apogee_member_flags", 0))) == bit for m in all_meta]).astype(bool))
    
    fp.create_dataset("flux", data=flux)
    fp.create_dataset("ivar", data=ivar)
    fp.create_dataset("model_flux", data=rectified_flux * continuum)
    fp.create_dataset("initial_model_flux", data=initial_rectified_flux)
    fp.create_dataset("initial_continuum", data=initial_continuum)
    fp.create_dataset("continuum", data=continuum)
    fp.create_dataset("sdss_id", data=np.hstack([r.get("sdss_id", -1) for r in all_meta]))


sdss_ids = np.hstack([r.get("sdss_id", -1) for r in all_meta])

sdss_id = 104414216
sdss_id = 129059176
sdss_id = 107550634
sdss_id = 77513476
sdss_id = 107565472
sdss_id = 104416297
sdss_id = 104416050
sdss_id = 104413460
sdss_id = 104417075
sdss_id = 107547609
sdss_id = 107558123
sdss_id = 107033877
sdss_id = 107546863
sdss_id = 107568299
sdss_id = 115039403
sdss_id = 61279382

index = np.where(sdss_ids == sdss_id)[0][0] # problematic one

init = {}
init.update(initial_parameters[index])
for key in meta["parameter_names"]:
    init[key] = np.clip(
        init[key],
        min_parameters[meta["parameter_names"].index(key)],
        max_parameters[meta["parameter_names"].index(key)],
    )
star = fit_spectrum(flux[index], ivar[index], init, *args)


params = dict(zip(parameter_names, from_domain(star[0][-n_stellar_parameters:])))

for key, value in params.items():
    print(f"{key}: {value:.4f}")


bfgs_rchi2 = jnp.sum((star[1] * star[2] - flux[index])**2 * ivar[index]) / 8575
print(f"BFGS rchi2: {bfgs_rchi2:.4f}")

fig, axes = plt.subplots(3, 1, figsize=(20, 10))

for i, ax in enumerate(axes.flat):
    ax.plot(λ, flux[index] / star[2], c="k", label="Data")
    ax.plot(
        λ,
        star[1],
        c="tab:red",
        label=f"Model (BFGS): Teff={params['Teff']:.0f} K logg={params['logg']:.2f} [M/H]={params['m_H']:.2f} vmic={params['vmic']:.2f} km/s vsini={params['vsini']:.1f} km/s",
    )
    ax.plot(
        λ, 
        star[3], 
        c="tab:blue", 
        label=f"Initial model: " + (" ".join([f"{k}={v:.2f}" for k, v in init.items()])),
        zorder=-1,
        )
    ax.set_ylim(0, 1.2)

regions = [(15120.0, 15820.0), (15840.0, 16440.0), (16450.0, 16960.0)]
for region, ax in zip(regions, axes.flat):
    ax.set_xlim(region)

axes[0].legend(frameon=False, loc="lower left")
fig.savefig(f"star_{sdss_id}_{index}.png", dpi=300, bbox_inches="tight")
print(sdss_id, index)