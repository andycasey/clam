import numpy as np
import jax
import jax.numpy as jnp

def get_initial_continuum_theta(flux, ivar, A):
    n_spectra, n_pixels = flux.shape
    θ = np.zeros((n_spectra, A.shape[1]))
    for i in range(n_spectra):
        keep = (ivar[i] > 0) & (flux[i] > 0)
        ATCinv = (A.T * ivar[i])[:, keep]
        θ[i] = np.linalg.lstsq(ATCinv @ A[keep], ATCinv @ flux[i][keep], rcond=None)[0]
    return jnp.array(θ)

@jax.jit
def fit_continuum_given_no_absorption(flux, ivar, A):
    """
    Return the continuum coefficients for a spectrum, given no absorption.

    :param flux: 
        The flux of the spectrum.
    
    :param ivar:
        The inverse variance of the spectrum.
    
    :param A:
        The design matrix for the continuum.
    """
    ATCinv = A.T * ivar
    θ, *extras = jnp.linalg.lstsq(ATCinv @ A, ATCinv @ flux, rcond=None)
    return θ




def get_bounds(n_spectra, n_stellar_parameters, n_continuum_coefficients):
    lower_bounds = jnp.hstack([
        -jnp.inf * jnp.ones(n_spectra * n_continuum_coefficients),
        jnp.zeros(n_spectra * n_stellar_parameters)
    ])
    upper_bounds = jnp.hstack([
        jnp.inf * jnp.ones(n_spectra * n_continuum_coefficients),
        jnp.ones(n_spectra * n_stellar_parameters)
    ])
    return jnp.vstack([lower_bounds, upper_bounds]).T
