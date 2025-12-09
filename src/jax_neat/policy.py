# jax_neat/policy.py
from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.lax as lax
from src.neat_core.genome import NodeType, NODE_TYPE_MAP

@dataclass
class JAXGenome:
    node_type: jnp.ndarray      # int32, shape (MAX_NODES,)
    node_activation: jnp.ndarray  # int32, shape (MAX_NODES,)
    conn_in: jnp.ndarray        # int32, shape (MAX_CONNS,)
    conn_out: jnp.ndarray       # int32, shape (MAX_CONNS,)
    conn_weight: jnp.ndarray    # float32, shape (MAX_CONNS,)
    conn_enabled: jnp.ndarray   # bool, shape (MAX_CONNS,)
    n_input: int
    n_output: int
    n_nodes: int
    n_conns: int


# This function would be placed in your JAX policy file

def jax_activate(z, act_id):
    """JAX-compatible activation dispatcher using lax.switch."""
    
    # Define a list of activation functions (must be the same order as ACT_TYPE_MAP)
    def identity(x): return x
    
    # Note: Tanh ID 1, ReLU ID 2, Sigmoid ID 3, Linear ID 4 (as per ACT_TYPE_MAP)
    activation_funcs = [
        lambda x: x,                      # ID 0 (unused/placeholder)
        jnp.tanh,                         # ID 1: TANH
        lambda x: jnp.maximum(0.0, x),    # ID 2: RELU
        lambda x: 1.0 / (1.0 + jnp.exp(-x)), # ID 3: SIGMOID
        identity                          # ID 4: LINEAR
    ]

    # lax.switch requires a 0-indexed integer. Since our IDs start at 1,
    # and we use 0 as a placeholder, we can use the ID directly as the index.
    return lax.switch(act_id, activation_funcs, z)

# In your JAX forward pass loop:
# current_activation_id = jax_genome.node_activation[node_idx]
# values[node_idx] = jax_activate(sum_of_weights, current_activation_id)
def jax_forward(gen: JAXGenome, obs: jnp.ndarray, n_output:int) -> jnp.ndarray:
    """Feed-forward a NEAT genome in JAX.

    gen: JAXGenome
    obs: shape (n_input,)
    returns: shape (n_output,)
    """
    # Values for all nodes, initialize with zeros.
    # Fill inputs, bias, then propagate hidden+output.
    if isinstance(gen, dict):
        from types import SimpleNamespace
        gen = SimpleNamespace(**gen)
    values = jnp.zeros_like(gen.node_type, dtype=jnp.float32)

    # Set input node values.
    # Assume inputs are nodes [0 .. n_input-1]
    values = jax.lax.dynamic_update_slice(values, obs, start_indices=(0,))


    # Set bias nodes (if any) to 1.0
    # Let's say bias nodes are type == 4
    bias_mask = (gen.node_type == NODE_TYPE_MAP[NodeType.BIAS])
    values = jnp.where(bias_mask, 1.0, values)

    # Topological eval:
    # We loop node_id from 0..n_nodes-1, but **skip inputs and bias** as
    # they are already set. For each node, we sum all enabled incoming
    # connections.
    def node_body(node_id, values):
        # Skip inputs and bias
        ntype = gen.node_type[node_id]
        is_input_or_bias = jnp.logical_or(
            ntype ==  NODE_TYPE_MAP[NodeType.INPUT],
            ntype ==  NODE_TYPE_MAP[NodeType.BIAS],
        )
        def skip(values):
            return values

        def compute(values):
            # All connections where out == node_id
            # We consider only the first n_conns; others are garbage.
            conn_mask = jnp.logical_and(
                gen.conn_enabled,
                gen.conn_out == node_id,
            )
            # Optionally: also mask by index < n_conns, if you want:
            # idx = jnp.arange(gen.conn_out.shape[0])
            # conn_mask = jnp.logical_and(conn_mask, idx < gen.n_conns)

            # Gather inputs and weights
            in_ids = gen.conn_in
            w = gen.conn_weight

            contrib = jnp.where(
                conn_mask,
                values[in_ids] * w,
                0.0,
            )
            s = jnp.sum(contrib)

            # activation
            act_id = gen.node_activation[node_id]
            activated_s = jax_activate(s, act_id)
            values = values.at[node_id].set(activated_s)
            return values

        return jax.lax.cond(is_input_or_bias, skip, compute, values)

    # Loop over nodes in order
    values = jax.lax.fori_loop(0, gen.n_nodes, node_body, values)

    # Collect outputs.
    # Assume outputs are the last n_output nodes: [n_nodes - n_output .. n_nodes)
    start = gen.n_nodes - gen.n_output
    return jax.lax.dynamic_slice(values, (start,), (n_output,))

def jax_genome_flatten(jg):
    children = (
        jg.node_type, jg.node_activation, jg.conn_in, jg.conn_out, jg.conn_weight, jg.conn_enabled
    )
    aux = {
        "n_input": jg.n_input,
        "n_output": jg.n_output,
        "n_nodes": jg.n_nodes,
        "n_conns": jg.n_conns,
    }
    return children, aux

def jax_genome_unflatten(aux, children) -> JAXGenome:
    return JAXGenome(
        node_type=children[0],
        node_activation=children[1],
        conn_in=children[2],
        conn_out=children[3],
        conn_weight=children[4],
        conn_enabled=children[5],
        n_input=aux["n_input"],
        n_output=aux["n_output"],
        n_nodes=aux["n_nodes"],
        n_conns=aux["n_conns"],
    )

jax.tree_util.register_pytree_node(JAXGenome, jax_genome_flatten, jax_genome_unflatten)
jax_forward_jit = jax.jit(jax_forward)