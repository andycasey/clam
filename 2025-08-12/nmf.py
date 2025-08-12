import jax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm

@jax.jit
def loss(W, H, V):
    return jnp.sum(jnp.square(W @ H - V)) / V.shape[0]

@jax.jit
def max_abs_diff(W, H, V):
    return jnp.max(jnp.abs(W @ H - V))

@jax.jit
def mean_abs_diff(W, H, V):
    return jnp.mean(jnp.abs(W @ H - V))

@partial(jax.jit, donate_argnums=(0,), static_argnames=("iterations", ))
def nmf_multiplicative_update_H(H, W, V, iterations, epsilon):
    def f(carry, x):
        H = carry

        H_new = jnp.clip(H * ((W.T @ V) / (W.T @ W @ H + epsilon)), epsilon, None)
        return (H_new, None)

    H_new, _ = jax.lax.scan(f, H, None, length=iterations)
    return H_new

@jax.jit
def update_H(W, H, V, epsilon):
    return jnp.clip(H * ((W.T @ V) / (W.T @ W @ H + epsilon)), epsilon, None)

@partial(jax.jit, static_argnums=(3,)) # iterations should be static for scan to unroll
def _multiplicative_update_WH(W, H, V, iterations, epsilon=1e-12):
    def f(carry, x):
        # carry is (W, H) from the previous iteration
        W, H = carry
        H = jnp.clip(H * ((W.T @ V) / (W.T @ W @ H + epsilon)), epsilon, None)
        W = jnp.clip(W * ((V @ H.T) / (W @ (H @ H.T) + epsilon)), epsilon, None)
        return ((W, H), None)

    (W, H), _ = jax.lax.scan(f, (W, H), None, length=iterations)
    return (W, H)

def multiplicative_updates_H(V, W, H, iterations=10_000, verbose_frequency=0, epsilon=1e-12):

    if verbose_frequency == 0:
        epochs, n_steps_per_epoch = (1, iterations)
    else:
        epochs, n_steps_per_epoch = (iterations // verbose_frequency, verbose_frequency)

    desc = "Minimising C(H|W,V)"
    losses = (_, last_loss, *_) = [0, loss(W, H, V), max_abs_diff(W, H, V), mean_abs_diff(W, H, V)]
    with tqdm(total=iterations, desc=f"{desc}: loss={last_loss}") as pb:
        for epoch in range(1, 1 + epochs):
            H = nmf_multiplicative_update_H(H, W, V, n_steps_per_epoch, epsilon=epsilon)

            this_loss, this_max_abs_diff, this_mean_abs_diff = (loss(W, H, V), max_abs_diff(W, H, V), mean_abs_diff(W, H, V))
            losses.append((n_steps_per_epoch * epoch, this_loss, this_max_abs_diff, this_mean_abs_diff))
            pb.set_description(f"{desc}: loss={this_loss:.2e} ({this_loss - last_loss:+.2e}); max_abs_diff={this_max_abs_diff:.2f}; mean_abs_diff={this_mean_abs_diff:.2e}")
            last_loss = this_loss
            pb.update(n_steps_per_epoch)
    
    return (H, losses)



def multiplicative_updates_WH(V, W, H, iterations=10_000, verbose_frequency=0, epsilon=1e-12):

    if verbose_frequency == 0:
        epochs, n_steps_per_epoch = (1, iterations)
    else:
        epochs, n_steps_per_epoch = (iterations // verbose_frequency, verbose_frequency)

    desc = "Minimising C(W,H|V)"
    losses = (_, last_loss, *_) = [0, loss(W, H, V), max_abs_diff(W, H, V), mean_abs_diff(W, H, V)]
    with tqdm(total=iterations, desc=f"{desc}: loss={last_loss}") as pb:
        for epoch in range(1, 1 + epochs):
            W, H = _multiplicative_update_WH(W, H, V, n_steps_per_epoch, epsilon=epsilon)

            this_loss, this_max_abs_diff, this_mean_abs_diff = (loss(W, H, V), max_abs_diff(W, H, V), mean_abs_diff(W, H, V))
            losses.append((n_steps_per_epoch * epoch, this_loss, this_max_abs_diff, this_mean_abs_diff))
            pb.set_description(f"{desc}: loss={this_loss:.2e} ({this_loss - last_loss:+.2e}); max_abs_diff={this_max_abs_diff:.2f}; mean_abs_diff={this_mean_abs_diff:.2e}")
            last_loss = this_loss
            pb.update(n_steps_per_epoch)
    
    return (W, H, losses)




@partial(jax.jit, static_argnames=("epsilon", ))
def basis_weights(A, X, epsilon=1e-12):
    return jnp.clip(A @ X, epsilon, None)
    
@jax.jit
def residual(W, H, V):
    return (W @ H - V)


# Define the loss function
@partial(jax.jit, static_argnames=("epsilon", "penalty"))
def loss_X(X, A, H, V, epsilon=1e-12, penalty=0.0):
    W = A @ X
    # Count elements of W that are negative
    non_negative_penalty = jnp.sum(penalty * jnp.where(W < 0, 1, 0) * -W)
    W = jnp.clip(W, epsilon, None)
    #W = jnp.abs(W)
    return jnp.sum(jnp.square(residual(W, H, V))) + non_negative_penalty

grad_loss_X = jax.grad(loss_X, argnums=0)

@partial(jax.jit, donate_argnums=(1, 2, 3), static_argnums=(5, 6, 7, 8)) # iterations should be static for scan to unroll
def _update_XH(A, X, W, H, V, iterations, learning_rate, epsilon=1e-12, penalty=0.0):

    def f(carry, x):
        # carry is (X, H, W) from the previous iteration
        X, H, W = carry
        X -= learning_rate * grad_loss_X(X, A, H, V, epsilon, penalty)
        W = basis_weights(A, X, epsilon)
        H = jnp.clip(H * ((W.T @ V) / (W.T @ W @ H + epsilon)), epsilon, None)
        return ((X, H, W), None)

    (X, H, W), _ = jax.lax.scan(f, (X, H, W), None, length=iterations)
    return (X, H, W)    

def update_XH(A, X, W, H, V, iterations, verbose_frequency=0, learning_rate=1e-8, epsilon=1e-12, penalty=0.0):        
    desc = "Minimising C(X,H|A,V)"
    last_loss = loss(basis_weights(A, X, epsilon), H, V)
    if verbose_frequency == 0:
        epochs, n_steps_per_epoch = (1, iterations)
    else:
        epochs, n_steps_per_epoch = (iterations // verbose_frequency, verbose_frequency)

    with tqdm(total=iterations, desc=f"{desc}: loss={last_loss:.2e}") as pb:
        for epoch in range(epochs):
            X, H, W = _update_XH(A, X, W, H, V, n_steps_per_epoch, learning_rate, epsilon=epsilon, penalty=penalty)
            this_loss = loss(W, H, V)
            pb.set_description(f"{desc}: loss={this_loss:.2e} ({this_loss - last_loss:+.2e}), min(AX)={jnp.min(A @ X):.2e} max_abs_diff={max_abs_diff(W, H, V):.2f}, mean_abs_diff={mean_abs_diff(W, H, V):.2e}")
            last_loss = this_loss
            pb.update(n_steps_per_epoch)
    
    return (X, H, W, [])



def basis_weights_AsXs(As, Xs):
    result = As[0] @ Xs[0]
    for A, X in zip(As[1:], Xs[1:]):
        result = jnp.concatenate([result, A @ X], axis=1)
    return result#, epsilon, None)


@jax.jit
def loss_Xs_H(Xs, As, H, V, epsilon, penalty):

    W = basis_weights_AsXs(As, Xs)
    # Count elements of W that are negative
    non_negative_penalty = jnp.sum(penalty * jnp.where(W < 0, 1, 0) * -W)
    W = jnp.clip(W, epsilon, None)
    #W = jnp.abs(W)
    return jnp.sum(jnp.square(W @ H - V)) + non_negative_penalty


grad_loss_Xs_H = jax.grad(loss_Xs_H, argnums=0)

@partial(jax.jit, static_argnames=("iterations", "learning_rate", "epsilon", "penalty"))
def update_Xs_H(As, Xs, H, V, iterations, learning_rate, epsilon, penalty):
    def f(carry, x):
        *Xs, H = carry
        grad = grad_loss_Xs_H(Xs, As, H, V, epsilon, penalty)
        new_Xs = []        
        for X, g in zip(Xs, grad):
            #assert jnp.isfinite(g).all()
            new_Xs.append(X - learning_rate * g)
        W = jnp.clip(basis_weights_AsXs(As, new_Xs), epsilon, None)

        H_new = jnp.clip(H * ((W.T @ V) / (W.T @ W @ H + epsilon)), epsilon, None)
        return ((*new_Xs, H), None)

    (*Xs, H), _ = jax.lax.scan(f, (*Xs, H), None, length=iterations)
    return (Xs, H)
