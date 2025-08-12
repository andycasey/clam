"""Forward model continuum-normalized spectra."""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import Tuple, Optional
from nufft import fourier_modes, fourier_matvec

from scalers import BaseScaler

class RectifiedFluxModel(eqx.Module):
    """
    Rectified flux model for continuum-normalized spectra.
    """

    parameter_names: Tuple[str, ...]
    parameter_scaler: BaseScaler
    n_parameters: int
    H: jnp.ndarray
    X: jnp.ndarray
    f_modes: jnp.ndarray
    delta_flux_models: Optional[Tuple['DeltaFluxModel', ...]] = None

    def __init__(
        self, 
        parameter_names: Tuple[str, ...],
        parameter_scaler: BaseScaler,
        H: jnp.ndarray, 
        X: jnp.ndarray, 
        n_modes: Tuple[int, ...], 
        delta_flux_models: Optional[Tuple['DeltaFluxModel', ...]] = None
    ):
        self.H = H
        self.X = X
        self.f_modes = fourier_modes(*n_modes)
        if delta_flux_models is None:
            self.delta_flux_models = ()
        else:
            self.delta_flux_models = delta_flux_models
        self.parameter_names = list(parameter_names)
        for model in self.delta_flux_models:
            self.parameter_names.extend(model.parameter_name)
        self.parameter_names = tuple(self.parameter_names)
        self.parameter_scaler = parameter_scaler
        self.n_parameters = len(self.parameter_names)
    
    def inverse_transform(self, θ: jnp.ndarray) -> jnp.ndarray:
        items = [self.parameter_scaler.inverse_transform(θ[:self.f_modes.shape[1]])]
        for i, model in enumerate(self.delta_flux_models):
            items.append(model.parameter_scaler.inverse_transform(θ[self.f_modes.shape[1] + i]))
        return jnp.hstack(items)

    def transform(self, θ: jnp.ndarray) -> jnp.ndarray:
        items = [self.parameter_scaler(θ[:self.f_modes.shape[1]])]
        for i, model in enumerate(self.delta_flux_models):
            items.append(model.parameter_scaler(θ[self.f_modes.shape[1] + i]))
        return jnp.hstack(items)

    def basis_weights(self, ϴ: jnp.ndarray, epsilon: float) -> jnp.ndarray:
        ϴ = ϴ.reshape((-1, self.f_modes.shape[1])).T
        A = fourier_matvec(ϴ, self.f_modes)
        return jnp.clip(A @ self.X, epsilon, None)

    def __call__(self, θ: jnp.ndarray, epsilon: float = 0.0) -> jnp.ndarray:
        W = self.basis_weights(θ[:self.f_modes.shape[1]], epsilon=epsilon)   
        absorption = W @ self.H
        
        if len(self.delta_flux_models) > 0:
            delta_contributions = []
            for i, model in enumerate(self.delta_flux_models):
                delta_contributions.append(model(θ[self.f_modes.shape[1] + i]))
            
            # Sum all contributions at once
            absorption = absorption + jnp.sum(jnp.stack(delta_contributions), axis=0)
        
        absorption = jnp.clip(absorption, 0, None)
        return 1 - absorption


def create_rectified_flux_model(
    parameter_names: Tuple[str, ...],
    parameter_scaler: BaseScaler,
    H: jnp.ndarray, 
    X: jnp.ndarray,
    n_modes: Tuple[int, ...],
    delta_flux_models: Optional[Tuple['DeltaFluxModel', ...]] = None
) -> RectifiedFluxModel:
    """Factory function to create a rectified flux model.
    
    :param parameter_names:
        Names of the parameters for the model.

    :param parameter_scaler:
        Scaler for the model parameters.

    :param H:
        Rectified flux basis matrix.

    :param X:
        Fourier coefficients for the basis functions.
    
    :param n_modes:
        Number of Fourier modes for each dimension.
    
    :param delta_flux_models:
        Optional tuple of DeltaFluxModel instances to model flux changes due to
    """
    assert H.shape[1] % 2 == 0, "H must have an even number of pixels"
    return RectifiedFluxModel(
        parameter_names=parameter_names,
        parameter_scaler=parameter_scaler,
        H=H, 
        X=X, 
        n_modes=n_modes,
        delta_flux_models=delta_flux_models
    )



class DeltaFluxModel(eqx.Module):
    """
    Model for the difference in stellar spectra due to an abundance change.    
    """

    parameter_name: str
    parameter_scaler: BaseScaler
    f_modes: jnp.ndarray
    mean: jnp.ndarray
    coefficients: jnp.ndarray
    intercept: jnp.ndarray
    
    def __init__(
        self,
        parameter_name: str,
        parameter_scaler: BaseScaler,
        f_modes: jnp.ndarray,
        mean: jnp.ndarray,
        coefficients: jnp.ndarray,
        intercept: jnp.ndarray,
    ):
        self.parameter_name = parameter_name
        self.parameter_scaler = parameter_scaler
        self.f_modes = f_modes
        self.mean = mean
        self.coefficients = coefficients
        self.intercept = intercept


    def __call__(self, θ: jnp.array) -> jnp.ndarray:
        """Predict the difference between two stellar spectra."""
        
        A = nufft.fourier_matvec(θ.reshape((-1, 1)), self.f_modes)
        abs_x_m = jnp.abs(self.parameter_scaler.inverse_transform(ϴ))
        return abs_x_m * ((A - self.mean) @ self.coefficients + self.intercept)
