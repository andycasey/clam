
import pickle
import jax.numpy as jnp
import spectrum_model
from scalars import PeriodicScalar


with open("boss_model.pkl", "rb") as fp:
    serialized = pickle.load(fp)

λ, X, H, W, n_modes, parameter_names, min_parameters, max_parameters = (
    serialized["λ"],
    serialized["X"],
    serialized["H"],
    serialized["W"],
    serialized["n_modes"],
    serialized["parameter_names"],
    serialized["min_parameters"],
    serialized["max_parameters"],
)

# Interpolate to the BOSS wavelength grid just for ease of use.
from scipy.interpolate import interp1d
import numpy as np

λ_boss = 10**(3.5523 + 1e-4 * np.arange(4648))
H_interp = np.array([
    interp1d(λ, H[i], bounds_error=False, fill_value=0.0)(λ_boss)
    for i in range(H.shape[0])
])

parameter_scaler = PeriodicScalar(min_parameters, max_parameters, domain_maximum=jnp.pi)



model = spectrum_model.create_stellar_spectrum_model(
    λ=λ_boss,
    parameter_names=parameter_names,
    parameter_scalar=parameter_scaler,
    H=H_interp,
    X=X,
    n_modes=n_modes,
    spectral_resolution=5_000,
    max_vsini=200.0,
    continuum_regions=(
        (3750.0, 6250.0),
        (6350.0, 12_000.0),
    ),
    continuum_n_modes=1
)

solar = jnp.array([0.0, 4.4, 5777])
θ = jnp.hstack([
    parameter_scaler.transform(solar),
    1.0,
    jnp.ones(model.continuum_model.design_matrix.shape[1]),
])

flux = model(θ)



import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(30, 2), layout="compressed")
ax.plot(model.λ, flux.flatten(), c="k")
ax.set_xlabel("Wavelength (Å)")
ax.set_ylabel("Rectified Flux")
ax.set_xlim(model.λ[0], model.λ[-1])
fig.savefig("boss_solar.png", dpi=300)




θ = jnp.hstack([
    parameter_scaler.transform(jnp.array([0.0, 4, 8050])),
    1.0,
    jnp.ones(model.continuum_model.design_matrix.shape[1]),
])

flux = model(θ)



import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(30, 2), layout="compressed")
ax.plot(model.λ, flux.flatten(), c="k")
ax.set_xlabel("Wavelength (Å)")
ax.set_ylabel("Rectified Flux")
ax.set_xlim(model.λ[0], model.λ[-1])
fig.savefig("boss_hot_star.png", dpi=300)


