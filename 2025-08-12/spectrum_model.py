"""Forward model stellar spectra."""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import Tuple, Optional
from nufft import fourier_modes, fourier_matvec

from convolved_flux import ConvolvedFluxModel, create_convolved_flux_model
from rectified_flux import RectifiedFluxModel, create_rectified_flux_model
from continuum import ContinuumModel, create_continuum_model

from scalers import BaseScaler

class StellarSpectrumModel(eqx.Module):

    """Model for stellar spectra."""
    
    λ: jnp.ndarray
    parameter_names: Tuple[str, ...]
    n_parameters: int
    convolved_flux_model: ConvolvedFluxModel
    continuum_model: ContinuumModel

    def __init__(
        self,
        λ: jnp.ndarray,
        continuum_model: ContinuumModel,
        convolved_flux_model: ConvolvedFluxModel,
    ):
        self.λ = λ
        self.continuum_model = continuum_model
        self.convolved_flux_model = convolved_flux_model
        self.parameter_names = tuple(
            list(self.convolved_flux_model.parameter_names) +
            list(self.continuum_model.parameter_names)
        )
        self.n_parameters = len(self.parameter_names)

    def _separate_parameters(self, θ: jnp.array) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
        θ_convolved_flux = θ[:self.convolved_flux_model.n_parameters]
        θ_continuum = θ[self.convolved_flux_model.n_parameters:]
        return (θ_convolved_flux[:-1], θ_convolved_flux[-1], θ_continuum)

    def __call__(self, θ: jnp.array) -> jnp.ndarray:
        """Predict the stellar spectrum given parameters."""
        convolved_flux, continuum = self._predict(*self._separate_parameters(θ))
        return convolved_flux * continuum

    def _predict(self, θ_stellar: jnp.array, vsini: float, θ_continuum: jnp.array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        convolved_flux = self.convolved_flux_model(θ_stellar, vsini)
        continuum = self.continuum_model(θ_continuum)
        return (convolved_flux, continuum)

    def transform(self, θ: jnp.array) -> jnp.array:
        """Transform the parameters."""
        return jnp.hstack([
            self.convolved_flux_model.transform(θ[:self.convolved_flux_model.n_parameters]),
            θ[self.convolved_flux_model.n_parameters:],
        ])

    def inverse_transform(self, θ: jnp.array) -> jnp.array:
        """Inverse transform the parameters."""
        return jnp.hstack([
            self.convolved_flux_model.inverse_transform(θ[:self.convolved_flux_model.n_parameters]),
            θ[self.convolved_flux_model.n_parameters:]
        ])


    def inverse_transform_as_dict(self, θ: jnp.array) -> dict:
        """Inverse transform the parameters and return as a dictionary."""
        return dict(zip(self.parameter_names, self.inverse_transform(θ)))


def create_stellar_spectrum_model(
    λ: jnp.ndarray,
    parameter_names: Tuple[str, ...],
    parameter_scaler: BaseScaler,
    H: jnp.ndarray,
    X: jnp.ndarray,
    n_modes: Tuple[int, ...],
    spectral_resolution: Optional[float] = None,
    max_vsini: Optional[float] = None,
    continuum_regions: Optional[Tuple[float, float]] = None,
    continuum_n_modes: Optional[int] = None,
    delta_flux_models: Optional[Tuple['DeltaFluxModel', ...]] = None,
) -> StellarSpectrumModel:
    """Factory function to create a stellar spectrum model.
    
    :param λ:
        Wavelength grid.
    
    :param parameter_names:
        Names of the model parameters.

    :param parameter_scaler:
        Scaler for the model parameters.

    :param H:
        Rectified flux matrix.
    
    :param X:
        Fourier coefficients for the basis functions.
    
    :param n_modes:
        Number of Fourier modes for each dimension.
    
    :param spectral_resolution:
        Spectral resolution of the model.
    
    :param max_vsini:
        Maximum vsini value for the model.
    
    :param continuum_regions:
        Tuple of (start, end) wavelength continuum regions.

    :param continuum_n_modes:
        Number of Fourier modes per continuum region.

    :param delta_flux_models:
        Optional tuple of DeltaFluxModel instances to model flux changes due to
        abundances.
    
    :return:
        A StellarSpectrumModel instance.
    """

    rectified_flux_model = create_rectified_flux_model(
        parameter_names=parameter_names, 
        parameter_scaler=parameter_scaler,
        H=H, 
        X=X, 
        n_modes=n_modes,
        delta_flux_models=delta_flux_models,
    )
    convolved_flux_model = create_convolved_flux_model(
        rectified_flux_model=rectified_flux_model,
        spectral_resolution=spectral_resolution,
        max_vsini=max_vsini,
    )    
    continuum_model = create_continuum_model(
        λ=λ,
        continuum_regions=continuum_regions,
        continuum_n_modes=continuum_n_modes,
    )
    
    return StellarSpectrumModel(
        λ=λ,
        convolved_flux_model=convolved_flux_model,
        continuum_model=continuum_model,
        
    )

