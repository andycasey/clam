
import jax
import jax.numpy as jnp
import equinox as eqx
from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.utils.grids import velocity_grid as _velocity_grid
from typing import Tuple, Optional
from rectified_flux import RectifiedFluxModel

class ConvolvedFluxModel(eqx.Module):
    """
    Convolved flux model for continuum-normalized spectra.
    """

    parameter_names: Tuple[str, ...]
    n_parameters: int
    max_vsini: float
    velocity_grid: jnp.ndarray
    rectified_flux_model: RectifiedFluxModel

    def __init__(
        self,
        rectified_flux_model: RectifiedFluxModel,
        spectral_resolution: float,
        max_vsini: float,
    ):
        self.rectified_flux_model = rectified_flux_model
        self.velocity_grid = _velocity_grid(spectral_resolution, max_vsini)
        self.max_vsini = max_vsini
        self.parameter_names = tuple(list(rectified_flux_model.parameter_names) + ["vsini"])
        self.n_parameters = rectified_flux_model.n_parameters + 1

    def transform(self, θ: jnp.ndarray) -> jnp.ndarray:
        """Transform the parameters."""
        return jnp.hstack([
            self.rectified_flux_model.transform(θ[:-1]),
            jnp.clip(θ[-1], 0, self.max_vsini)
        ])

    def inverse_transform(self, θ: jnp.ndarray) -> jnp.ndarray:
        """Inverse transform the parameters."""
        return jnp.hstack([
            self.rectified_flux_model.inverse_transform(θ[:-1]),
            jnp.clip(θ[-1], 0, self.max_vsini)
        ])
    
    def __call__(self, θ: jnp.ndarray, vsini: float, epsilon: float = 0.0) -> jnp.ndarray:
        rectified_flux = self.rectified_flux_model(θ, epsilon=epsilon)
        vsini = jnp.clip(vsini, 0, self.max_vsini)
        return jax.lax.cond(
            vsini > 0,
            lambda: self.convolve_flux(rectified_flux, vsini),
            lambda: rectified_flux
        )

    def convolve_flux(self, rectified_flux, vsini):
        return (
            convolve_rigid_rotation(rectified_flux.flatten(), self.velocity_grid, vsini)
            .reshape((1, -1))
        )

def create_convolved_flux_model(
    rectified_flux_model: RectifiedFluxModel,
    spectral_resolution: float,
    max_vsini: float = 200.0, # km/s
) -> ConvolvedFluxModel:
    """Factory function to create a convolved flux model.
    
    :param rectified_flux_model:
        Rectified flux model instance.
    
    :param spectral_resolution:
        Spectral resolution of the model.
    
    :param max_vsini:
        Maximum vsini value for the convolution.
    """
    return ConvolvedFluxModel(
        rectified_flux_model=rectified_flux_model,
        spectral_resolution=spectral_resolution,
        max_vsini=max_vsini
    )