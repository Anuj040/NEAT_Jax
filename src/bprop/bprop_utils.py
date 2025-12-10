import jax.numpy as jnp
import jax
import optax

from src.jax_neat.policy import jax_forward_brpop

def classification_loss(raw_outputs: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the sparse softmax cross-entropy loss.

    raw_outputs: (N, 2) array of scores/logits from the network.
    Y: (N,) array of true class labels (0 or 1).

    Returns: Scalar loss value.
    """
    # JAX's softmax_cross_entropy_with_logits requires raw logits and
    # performs softmax internally, which is more numerically stable.
    # The loss is averaged over all N samples.
    # loss = jnp.mean(
    #     jax.nn.softmax_cross_entropy_with_logits(
    #         logits=raw_outputs, 
    #         labels=jax.nn.one_hot(Y, num_classes=raw_outputs.shape[-1])
    #     )
    # )
    loss = optax.softmax_cross_entropy_with_integer_labels(raw_outputs, Y).mean()
    return loss

def compute_loss_for_genome(conn_weight: jnp.ndarray, conn_enabled: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray, 
    static_genome_params: dict, n_output: int = 2, max_nodes: int = 0
) -> jnp.ndarray:
    """A wrapper to run the forward pass and calculates loss"""
    
    # Reconstruct a temporary genome dictionary for jax_forward
    temp_gen_params = {**static_genome_params, 
                       "conn_weight": conn_weight, 
                       "conn_enabled": conn_enabled}
    jax_forward_vmap_obs = jax.vmap(jax_forward_brpop, in_axes=(None, 0, None, None))
    raw_outputs = jax_forward_vmap_obs(temp_gen_params, X, n_output, max_nodes) 
    return classification_loss(raw_outputs, Y)

# Use jax.grad to compute gradients ONLY w.r.t. the first argument (conn_weight)
grad_fn = jax.grad(compute_loss_for_genome, argnums=0)

@jax.jit
def sgd_update(current_weights, gradients, learning_rate: float) -> jnp.ndarray:
    """Applies a simple SGD step."""
    # Only update weights corresponding to ENABLED connections (optional but cleaner)
    # Note: jax_forward already gates connections by conn_enabled, but updating
    # disabled weights is harmless.
    return current_weights - learning_rate * gradients