
import jax
import jax.numpy as jnp
import equinox as eqx
from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.utils.grids import velocity_grid as _velocity_grid

from rectified_flux import RectifiedFluxModel

class ConvolvedFluxModel(eqx.Module):
    """
    Convolved flux model for continuum-normalized spectra.
    """

    rectified_flux_model: RectifiedFluxModel
    velocity_grid: jnp.ndarray
    max_vsini: float
    n_parameters: int

    def __init__(
        self,
        rectified_flux_model: RectifiedFluxModel,
        spectral_resolution: float,
        max_vsini: float,
    ):
        self.rectified_flux_model = rectified_flux_model
        self.velocity_grid = _velocity_grid(spectral_resolution, max_vsini)
        self.max_vsini = max_vsini
        self.n_parameters = rectified_flux_model.n_parameters + 1

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