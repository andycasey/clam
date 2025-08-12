"""Forward model continuum-normalized spectra."""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import Tuple, Optional
from nufft import fourier_modes, fourier_matvec

class RectifiedFluxModel(eqx.Module):
    """
    Rectified flux model for continuum-normalized spectra.
    """

    H: jnp.ndarray
    X: jnp.ndarray
    f_modes: jnp.ndarray
    n_parameters: int

    def __init__(self, H: jnp.ndarray, X: jnp.ndarray, n_modes: Tuple[int, ...]):
        self.H = H
        self.X = X
        self.f_modes = fourier_modes(*n_modes)
        self.n_parameters = len(n_modes)

    def basis_weights(self, ϴ: jnp.ndarray, epsilon: float) -> jnp.ndarray:
        ϴ = ϴ.reshape((-1, self.f_modes.shape[1])).T
        A = fourier_matvec(ϴ, self.f_modes)
        return jnp.clip(A @ self.X, epsilon, None)

    def __call__(self, θ: jnp.ndarray, epsilon: float = 0.0) -> jnp.ndarray:
        W = self.basis_weights(θ, epsilon=epsilon)
        return 1 - W @ self.H


def create_rectified_flux_model(
    H: jnp.ndarray, 
    X: jnp.ndarray,
    n_modes: Tuple[int, ...]
) -> RectifiedFluxModel:
    """Factory function to create a rectified flux model.
    
    :param H:
        Rectified flux basis matrix.

    :param X:
        Fourier coefficients for the basis functions.
    
    :param n_modes:
        Number of Fourier modes for each dimension.
    """
    assert H.shape[1] % 2 == 0, "H must have an even number of pixels"
    return RectifiedFluxModel(H=H, X=X, n_modes=n_modes)


