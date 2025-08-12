"""Equinox module for continuum forward modeling."""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import Tuple, Optional


def get_fourier_mode_indices(P):
    return jnp.arange(P) - P // 2

def fourier_modes(*n_modes):
    indices = list(map(get_fourier_mode_indices, n_modes))    
    return -1j * jnp.array(jnp.meshgrid(*indices, indexing='ij')).T.astype(jnp.complex64)


def fourier_design_matrix(x, n_modes):
    x = jnp.atleast_2d(x)
    n_dim, n_data = x.shape
    n_modes = jnp.atleast_1d(n_modes)
    assert n_dim == n_modes.size

    A = jnp.exp(fourier_modes(*n_modes) @ x).T.reshape(n_data, -1)
    m = A.shape[1]

    # make a dense matrix by taking the matrix-matrix product with an identity matrix
    # that has been made hermitian
    cols = [0.5 * (A[:, i].real + A[:, m - i - 1].real) for i in range(m // 2 + 1)]
    for i in range(m // 2 + 1, m):
        cols.append((0.5j * A[:, i] - 0.5j * A[:, m - i - 1]).real)
    return jnp.vstack(cols).T


class ContinuumModel(eqx.Module):
    """
    Module for modeling stellar continuum using Fourier basis functions.    
    """
    
    continuum_regions: Tuple[Tuple[float, float], ...]
    continuum_n_modes: int
    design_matrix: jnp.ndarray
    
    def __init__(
        self, 
        λ: jnp.array,
        continuum_regions: Tuple[Tuple[float, float], ...] = ((15120.0, 15820.0), (15840.0, 16440.0), (16450.0, 16960.0)),
        continuum_n_modes: int = 7,
    ):
        """Initialize the continuum model.
        
        Args:
            continuum_regions: Tuple of (start, end) λ continuum_regions
            continuum_n_modes: Number of Fourier modes per region
            key: Random key for coefficient initialization
        """
        self.continuum_regions = continuum_regions
        self.continuum_n_modes = continuum_n_modes
        self.design_matrix = self.create_design_matrix(λ)
            
    def create_design_matrix(self, λ: jnp.ndarray) -> jnp.ndarray:
        """Create the design matrix for the given λ array.
        
        Args:
            λ: λ array
            
        Returns:
            Design matrix A such that continuum = coefficients @ A.T
        """
        A = jnp.zeros((λ.size, len(self.continuum_regions) * self.continuum_n_modes))
        
        for i, (start_wave, end_wave) in enumerate(self.continuum_regions):
            # Find λ indices for this region
            start_idx = jnp.searchsorted(λ, start_wave, side='left')
            end_idx = jnp.searchsorted(λ, end_wave, side='right')
            
            # Normalize λ to [0, 1] within the region
            wave_region = λ[start_idx:end_idx]
            if wave_region.size > 0:
                x_norm = (wave_region - wave_region[0]) / (wave_region[-1] - wave_region[0] + 1e-10)
                
                # Create Fourier design matrix for this region
                start_coeff = i * self.continuum_n_modes
                end_coeff = (i + 1) * self.continuum_n_modes
                A = A.at[start_idx:end_idx, start_coeff:end_coeff].set(
                    fourier_design_matrix(x_norm.reshape(1, -1), self.continuum_n_modes)
                )
        
        return A
    
    def __call__(self, θ) -> jnp.ndarray:
        """Forward model the continuum.
        
        Args:
            θ: Model parameters

        Returns:
            Continuum values at the given λs
        """
        return self.design_matrix @ θ
    


def create_continuum_model(
    λ: jnp.ndarray,
    continuum_regions: Tuple[Tuple[float, float], ...] = ((15120.0, 15820.0), (15840.0, 16440.0), (16450.0, 16960.0)),
    continuum_n_modes: int = 7,
) -> ContinuumModel:
    """Factory function to create a continuum model.
    
    Args:
        λ: λ array (used to validate continuum_regions)
        continuum_regions: λ continuum_regions for piecewise modeling
        continuum_n_modes: Number of Fourier modes per region
        key: Random key for initialization
        
    Returns:
        Initialized ContinuumModel
    """
    return ContinuumModel(
        λ=λ, 
        continuum_regions=continuum_regions, 
        continuum_n_modes=continuum_n_modes
    )
    
