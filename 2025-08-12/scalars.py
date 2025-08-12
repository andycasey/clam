import equinox as eqx
import jax.numpy as jnp
from jax import jit

class BaseScaler(eqx.Module):
    pass

class StandardScaler(BaseScaler):
    minimum: jnp.ndarray
    maximum: jnp.ndarray

    def __init__(self, minimum=None, maximum=None):
        self.minimum = minimum
        self.maximum = maximum

    def fit(self, X):
        minimum = jnp.min(X, axis=0)
        maximum = jnp.max(X, axis=0)
        return StandardScaler(minimum=minimum, maximum=maximum)

    def fit_transform(self, X):
        self = self.fit(X)
        return self(X)
    
    def transform(self, X):
        return self(X)

    def inverse_transform(self, X):
        return X * (self.maximum - self.minimum) + self.minimum

    def __call__(self, X):
        return (X - self.minimum) / (self.maximum - self.minimum)
    


class PeriodicScalar(BaseScaler):

    minimum: jnp.ndarray
    maximum: jnp.ndarray
    domain_minimum: float
    domain_maximum: float

    def __init__(
        self, 
        minimum=None, 
        maximum=None, 
        domain_minimum=0.0, 
        domain_maximum=jnp.pi
    ):
        self.minimum = minimum
        self.maximum = maximum
        self.domain_minimum = domain_minimum
        self.domain_maximum = domain_maximum
    
    def fit(self, X):
        minimum = jnp.min(X, axis=0)
        maximum = jnp.max(X, axis=0)
        return PeriodicScalar(
            minimum=minimum, 
            maximum=maximum, 
            domain_minimum=self.domain_minimum, 
            domain_maximum=self.domain_maximum
        )
    
    def transform(self, X):
        return self(X)
    
    def fit_transform(self, X):
        self = self.fit(X)
        return self(X)
    
    def inverse_transform(self, X):
        domain_edge = (
            self.domain_maximum 
        -   0.5 * jnp.abs(self.domain_maximum - self.domain_minimum)
        )
        
        X %= 2 * jnp.pi
        X = jnp.where(X > (self.domain_maximum + domain_edge), X - 2 * jnp.pi, X) # wrap around
        X = (X - self.domain_minimum) / (self.domain_maximum - self.domain_minimum)
        return (
            X * (self.max_parameters - self.min_parameters) 
        +   self.min_parameters
        )
    
    def __call__(self, X):
        X_scaled = (X - self.minimum) / (self.maximum - self.minimum)
        X_scaled *= (self.domain_maximum - self.domain_minimum) + self.domain_minimum
        return X_scaled

