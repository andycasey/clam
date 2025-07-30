import numpy as np
import jax
import jax.numpy as jnp

import jax.numpy as jnp
import jax
from typing import Sequence, Union

def get_fourier_mode_indices(P):
    return np.arange(P) - P // 2

def get_fourier_mode_indices_jax(P):
    return jnp.arange(P) - P // 2

def fourier_modes(*n_modes):
    indices = list(map(get_fourier_mode_indices, n_modes))    
    return -1j * np.array(np.meshgrid(*indices, indexing='ij')).T.astype(np.complex64)

def fourier_modes_jax(*n_modes):
    indices = list(map(get_fourier_mode_indices_jax, n_modes))    
    return -1j * jnp.array(jnp.meshgrid(*indices, indexing='ij')).T.astype(jnp.complex64)

def fourier_design_matrix(x, n_modes):    
    x = np.atleast_2d(x)
    n_dim, n_data = x.shape
    n_modes = np.atleast_1d(n_modes)
    assert n_dim == n_modes.size

    A = np.exp(fourier_modes(*n_modes) @ x).T.reshape(n_data, -1)
    m = A.shape[1]    

    # make a dense matrix by taking the matrix-matrix product with an identity matrix
    # that has been made hermitian
    cols = [0.5 * (A[:, i].real + A[:, m - i - 1].real  ) for i in range(m // 2 + 1)]
    for i in range(m // 2 + 1, m):
        cols.append((0.5j * A[:, i] - 0.5j * A[:, m - i - 1]).real)
    return np.vstack(cols).T

@jax.jit
def fourier_design_matrix_jax(x, n_modes):
    x = jnp.atleast_2d(x)
    n_dim, n_data = x.shape
    n_modes = jnp.atleast_1d(n_modes)
    assert n_dim == n_modes.size

    A = jnp.exp(fourier_modes_jax(*n_modes) @ x).T.reshape(n_data, -1)
    m = A.shape[1]

    # make a dense matrix by taking the matrix-matrix product with an identity matrix
    # that has been made hermitian
    cols = [0.5 * (A[:, i].real + A[:, m - i - 1].real) for i in range(m // 2 + 1)]
    for i in range(m // 2 + 1, m):
        cols.append((0.5j * A[:, i] - 0.5j * A[:, m - i - 1]).real)
    return jnp.vstack(cols).T

@jax.jit
def update_H(W, H, V, epsilon, full_output=False):
    numerator = W.T @ V
    denominator = W.T @ W @ H + epsilon
    scale = (numerator / denominator)
    H = jnp.clip(H * scale, epsilon, None)
    if full_output:
        return (H, jnp.max(jnp.abs(scale - 1)))
    else:
        return H

@jax.jit
def update_H_nd(W, H, V, epsilon):
    W_flat = W.reshape((-1, W.shape[-1]))
    numerator = W_flat.T @ V.reshape((-1, V.shape[-1]))
    denominator = W_flat.T @ W_flat @ H + epsilon
    return jnp.clip(H * (numerator / denominator), epsilon, None)


@jax.jit
def update_W_nd(W, H, V, epsilon):
    numerator = V @ H.T
    denominator = W @ (H @ H.T) + epsilon
    scale = (numerator / denominator)
    return jnp.clip(W * scale, 0, None)

@jax.jit
def update_W(W, H, V, epsilon):
    numerator = V @ H.T
    denominator = W @ (H @ H.T)
    scale = (numerator / denominator)
    W = jnp.clip(W * scale, epsilon, None)
    return W


def basis_weights(A, X, epsilon):
    return jnp.clip(A @ X, epsilon, None)

@jax.jit
def update_X(A, X, H, V, learning_rate, epsilon):
    W = basis_weights(A, X, epsilon)
    return X - learning_rate * (A.T @ (W * ((1 - V / (W @ H)) @ H.T)))


def continuum_design_matrix(
    λ, 
    regions = ((15120.0, 15820.0), (15840.0, 16440.0), (16450.0, 16960.0)),
    n_modes=7
):
    A = np.zeros((λ.size, len(regions) * n_modes))
    for i, (sr, er) in enumerate(regions):
        si, ei = λ.searchsorted((sr, er))
        x = (λ[si:ei] - λ[si]) / np.ptp(λ[si:ei])
        sj, ej = (i * n_modes, (i + 1) * n_modes)
        A[si:ei, sj:ej] = fourier_design_matrix(x, n_modes)
    return jnp.array(A)



def irdftn_basis_weights(X, t, s=None, axes=None):
    """
    Compute the basis weights at location `t` using the inverse real discrete Fourier transform.

    Args:
        X: Fourier coefficients with shape (..., n1, n2, ..., nk, m), where m is the last axis.
        t: Evaluation points with shape (..., k).
        s: Shape of the original real signal. If None, inferred from X shape.
        axes: Which axes of X correspond to the Fourier dimensions. If None, uses the last len(t.shape[-1]) axes.

    Returns:
        Real values of the inverse transform evaluated at points t.
    """
    # Vectorize the evaluation of `evaluate_irdftn_at_t` over the last axis of X
    evaluate_irdftn_vmap = jax.vmap(
        lambda x: evaluate_irdftn_at_t(x, t, s=s, axes=axes), 
        in_axes=-1
    )
    return evaluate_irdftn_vmap(X)
    

def prepare_irdftn(t: jnp.ndarray, fourier_shape: Union[Sequence[int]], s: Union[Sequence[int], None] = None, axes: Union[Sequence[int], None] = None) -> jnp.ndarray:

    if axes is None:
        # Assume the last dimensions of X are the Fourier dimensions
        ndim = t.shape[-1]
        axes = tuple(range(-ndim, 0))
    else:
        ndim = len(axes)
        axes = tuple(axes)

    # Determine original signal shape
    if s is None:
        # For real DFT, the last dimension is truncated
        original_shape = list(fourier_shape)
        original_shape[-1] = 2 * (fourier_shape[-1] - 1)
        original_shape = tuple(original_shape)
    else:
        original_shape = tuple(s)
    
    # Reshape t to broadcast properly
    batch_shape = t.shape[:-1]
    t = jnp.reshape(t, (-1, ndim))  # Flatten batch dimensions
    n_points = t.shape[0]
    
    # Create all possible index combinations for Fourier modes
    index_arrays = jnp.meshgrid(*[jnp.arange(s) for s in fourier_shape], indexing='ij')
    indices = jnp.stack([idx.flatten() for idx in index_arrays], axis=1)
        
    # Calculate phases for all modes and all evaluation points
    phases = jnp.zeros((n_points, len(indices)))
    
    for dim_idx in range(ndim):
        mode_indices = indices[:, dim_idx]
        
        if dim_idx == ndim - 1:  # Last dimension (real DFT)
            # Frequencies are 0, 1, 2, ..., n//2
            freqs = mode_indices
        else:  # Other dimensions (standard DFT)
            # Standard DFT frequencies: 0, 1, ..., n//2, -(n//2-1), ..., -1
            n = original_shape[dim_idx]
            freqs = jnp.where(mode_indices <= n//2, 
                            mode_indices, 
                            mode_indices - n)
        
        # Add contribution to phase
        phases += jnp.outer(t[:, dim_idx], freqs) * (2 * jnp.pi / original_shape[dim_idx])
    
    # Calculate trigonometric functions
    cos_phases = jnp.cos(phases)
    sin_phases = jnp.sin(phases)
        
    # For real DFT, modes with k > 0 in the last dimension need to be doubled
    # to account for the missing negative frequency components
    last_mode_indices = indices[:, -1]
    is_dc_or_nyquist = (last_mode_indices == 0) | (last_mode_indices == fourier_shape[-1] - 1)    

    return (cos_phases, sin_phases, is_dc_or_nyquist, batch_shape)



def evaluate_irdftn_at_t_single_prepared(X, cos_phases, sin_phases, is_dc_or_nyquist, batch_shape):

    coeffs = X.flatten()
    real_parts = jnp.real(coeffs)
    imag_parts = jnp.imag(coeffs)

    contributions = real_parts[None, :] * cos_phases - imag_parts[None, :] * sin_phases
    
    # Double the non-DC/Nyquist components for real DFT
    contributions = jnp.where(is_dc_or_nyquist[None, :], contributions, 2 * contributions)
    
    result = jnp.sum(contributions, axis=1)
    
    # Reshape back to original batch shape
    result = jnp.reshape(result, batch_shape)
    return result


def evaluate_irdftn_at_t_single(X, cos_phases, sin_phases, is_dc_or_nyquist, batch_shape):
    # Vectorize the evaluation of `evaluate_irdftn_at_t` over the last axis of X
    evaluate_irdftn_vmap = jax.vmap(
        lambda x: evaluate_irdftn_at_t_single_prepared(x, cos_phases, sin_phases, is_dc_or_nyquist, batch_shape), 
        in_axes=-1
    )
    return evaluate_irdftn_vmap(X)
    

def evaluate_irdftn_at_t(X: jnp.ndarray, t: jnp.ndarray, 
                        s: Union[Sequence[int], None] = None,
                        axes: Union[Sequence[int], None] = None) -> jnp.ndarray:
    """
    Evaluate the inverse real discrete Fourier transform at arbitrary points.
    
    This computes the multi-dimensional trigonometric polynomial represented by
    the Fourier coefficients X at the specified coordinates t.
    
    Args:
        X: Complex Fourier coefficients with shape (..., n1, n2, ..., nk)
           where the last dimensions correspond to the Fourier modes.
           For real DFT, the last axis should have size (original_size // 2 + 1).
        t: Evaluation points with shape (..., k) where k is the number of dimensions
           to transform. Coordinates should be in the same units as array indices.
        s: Shape of the original real signal. If None, inferred from X shape.
        axes: Which axes of X correspond to the Fourier dimensions. 
              If None, uses the last len(t.shape[-1]) axes.
    
    Returns:
        Real values of the inverse transform evaluated at points t.
    """
    
    if axes is None:
        # Assume the last dimensions of X are the Fourier dimensions
        ndim = t.shape[-1]
        axes = tuple(range(-ndim, 0))
    else:
        ndim = len(axes)
        axes = tuple(axes)
    
    # Get the shape of the Fourier dimensions
    fourier_shape = tuple(X.shape[ax] for ax in axes)
    
    # Determine original signal shape
    if s is None:
        # For real DFT, the last dimension is truncated
        original_shape = list(fourier_shape)
        original_shape[-1] = 2 * (fourier_shape[-1] - 1)
        original_shape = tuple(original_shape)
    else:
        original_shape = tuple(s)
    
    # Reshape t to broadcast properly
    batch_shape = t.shape[:-1]
    t = jnp.reshape(t, (-1, ndim))  # Flatten batch dimensions
    n_points = t.shape[0]
    
    # Create all possible index combinations for Fourier modes
    index_arrays = jnp.meshgrid(*[jnp.arange(s) for s in fourier_shape], indexing='ij')
    indices = jnp.stack([idx.flatten() for idx in index_arrays], axis=1)
    
    # Get all coefficients
    coeffs = X.flatten()
    
    # Calculate phases for all modes and all evaluation points
    phases = jnp.zeros((n_points, len(indices)))
    
    for dim_idx in range(ndim):
        mode_indices = indices[:, dim_idx]
        
        if dim_idx == ndim - 1:  # Last dimension (real DFT)
            # Frequencies are 0, 1, 2, ..., n//2
            freqs = mode_indices
        else:  # Other dimensions (standard DFT)
            # Standard DFT frequencies: 0, 1, ..., n//2, -(n//2-1), ..., -1
            n = original_shape[dim_idx]
            freqs = jnp.where(mode_indices <= n//2, 
                            mode_indices, 
                            mode_indices - n)
        
        # Add contribution to phase
        phases += jnp.outer(t[:, dim_idx], freqs) * (2 * jnp.pi / original_shape[dim_idx])
    
    # Calculate trigonometric functions
    cos_phases = jnp.cos(phases)
    sin_phases = jnp.sin(phases)
    
    # Handle real DFT conjugate symmetry
    real_parts = jnp.real(coeffs)
    imag_parts = jnp.imag(coeffs)
    
    # For real DFT, modes with k > 0 in the last dimension need to be doubled
    # to account for the missing negative frequency components
    last_mode_indices = indices[:, -1]
    is_dc_or_nyquist = (last_mode_indices == 0) | (last_mode_indices == fourier_shape[-1] - 1)
    
    # Calculate contributions
    contributions = real_parts[None, :] * cos_phases - imag_parts[None, :] * sin_phases
    
    # Double the non-DC/Nyquist components for real DFT
    contributions = jnp.where(is_dc_or_nyquist[None, :], contributions, 2 * contributions)
    
    result = jnp.sum(contributions, axis=1)
    
    # Reshape back to original batch shape
    result = jnp.reshape(result, batch_shape)
    
    return result