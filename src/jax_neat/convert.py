from __future__ import annotations
import numpy as np
import jax.numpy as jnp

from src.neat_core.genome import Genome, NodeType
from src.jax_neat.policy import JAXGenome
from src.jax_neat.config import NeatHyperParams

MODEL_CONFIG = NeatHyperParams()
NODE_TYPE_MAP = {
    NodeType.INPUT: 1,
    NodeType.HIDDEN: 2,
    NodeType.OUTPUT: 3,
    NodeType.BIAS: 4,
}

def genome_to_jax(gen: Genome, obs_dim: int, act_dim: int) -> JAXGenome:
    # Sort node IDs so we have a consistent order.
    node_ids = sorted(gen.nodes.keys())
    n_nodes = len(node_ids)

    if n_nodes > MODEL_CONFIG.MAX_NODES:
        raise ValueError(f"Too many nodes ({n_nodes}) for MAX_NODES={MODEL_CONFIG.MAX_NODES}")

    # Build node_type array
    node_type_arr = np.zeros((MODEL_CONFIG.MAX_NODES,), dtype=np.int32)
    id_to_idx = {}  # map old node id -> new index [0..n_nodes-1]

    for idx, nid in enumerate(node_ids):
        id_to_idx[nid] = idx
        node_type_arr[idx] = NODE_TYPE_MAP[gen.nodes[nid].type]

    # Count inputs and outputs from types
    input_mask = (node_type_arr[:n_nodes] == NODE_TYPE_MAP[NodeType.INPUT])
    output_mask = (node_type_arr[:n_nodes] == NODE_TYPE_MAP[NodeType.OUTPUT])

    n_input = int(input_mask.sum())
    n_output = int(output_mask.sum())

    # Sanity check with expected obs_dim / act_dim
    assert n_input == obs_dim, f"Expected {obs_dim} inputs, got {n_input}"
    assert n_output == act_dim, f"Expected {act_dim} outputs, got {n_output}"

    # Connections
    conns = gen.connections
    n_conns = len(conns)
    if n_conns > MODEL_CONFIG.MAX_CONNS:
        raise ValueError(f"Too many connections ({n_conns}) for MAX_CONNS={MODEL_CONFIG.MAX_CONNS}")

    conn_in = np.zeros((MODEL_CONFIG.MAX_CONNS,), dtype=np.int32)
    conn_out = np.zeros((MODEL_CONFIG.MAX_CONNS,), dtype=np.int32)
    conn_weight = np.zeros((MODEL_CONFIG.MAX_CONNS,), dtype=np.float32)
    conn_enabled = np.zeros((MODEL_CONFIG.MAX_CONNS,), dtype=bool)

    for i, c in enumerate(conns):
        conn_in[i] = id_to_idx[c.in_id]
        conn_out[i] = id_to_idx[c.out_id]
        conn_weight[i] = c.weight
        conn_enabled[i] = c.enabled

    # Convert to JAX arrays
    return JAXGenome(
        node_type=jnp.array(node_type_arr),
        conn_in=jnp.array(conn_in),
        conn_out=jnp.array(conn_out),
        conn_weight=jnp.array(conn_weight),
        conn_enabled=jnp.array(conn_enabled),
        n_input=n_input,
        n_output=n_output,
        n_nodes=n_nodes,
        n_conns=n_conns,
    )
