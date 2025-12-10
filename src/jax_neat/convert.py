from __future__ import annotations
import numpy as np
import jax.numpy as jnp

from src.neat_core.genome import Genome, NodeType, NODE_TYPE_MAP, ACT_TYPE_MAP
from src.jax_neat.policy import JAXGenome
from src.jax_neat.config import NeatHyperParams

MODEL_CONFIG = NeatHyperParams()

def genome_to_jax(gen: Genome, obs_dim: int, act_dim: int) -> JAXGenome:
    # Sort node IDs so we have a consistent order.
    node_ids = sorted(gen.nodes.keys())
    n_nodes = len(node_ids)

    if n_nodes > MODEL_CONFIG.MAX_NODES:
        raise ValueError(f"Too many nodes ({n_nodes}) for MAX_NODES={MODEL_CONFIG.MAX_NODES}")

    # Build node_type array
    node_type_arr = np.zeros((MODEL_CONFIG.MAX_NODES,), dtype=np.int32)
    node_activation_arr = np.ones((MODEL_CONFIG.MAX_NODES,), dtype=np.int32)
    id_to_idx = {}  # map old node id -> new index [0..n_nodes-1]

    for idx, nid in enumerate(node_ids):
        id_to_idx[nid] = idx
        node_type_arr[idx] = NODE_TYPE_MAP[gen.nodes[nid].type]
        # ADDED: Map Python ActivationType to JAX integer ID
        if gen.nodes[nid].type in (NodeType.HIDDEN, NodeType.OUTPUT):
            # Only HIDDEN and OUTPUT nodes need an activation
            node_activation_arr[idx] = ACT_TYPE_MAP[gen.nodes[nid].activation]

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
        node_activation=jnp.array(node_activation_arr),
        conn_in=jnp.array(conn_in),
        conn_out=jnp.array(conn_out),
        conn_weight=jnp.array(conn_weight),
        conn_enabled=jnp.array(conn_enabled),
        n_input=n_input,
        n_output=n_output,
        n_nodes=n_nodes,
        n_conns=n_conns,
    )

def genomes_to_params_batch(genomes: list, obs_dim: int, act_dim: int) -> dict[str, jnp.ndarray]:
    """Convert a list of Python Genomes into batched JAX params.

    Returns a dict of JAX arrays with leading dimension P = len(genomes).
    """
    max_nodes = MODEL_CONFIG.MAX_NODES
    max_conns = MODEL_CONFIG.MAX_CONNS
    P = len(genomes)

    node_type   = np.zeros((P, max_nodes), dtype=np.int32)
    node_activation = np.ones((P, max_nodes), dtype=np.int32)
    conn_in     = np.zeros((P, max_conns), dtype=np.int32)
    conn_out    = np.zeros((P, max_conns), dtype=np.int32)
    conn_weight = np.zeros((P, max_conns), dtype=np.float32)
    conn_enabled= np.zeros((P, max_conns), dtype=bool)
    n_input     = np.zeros((P,), dtype=np.int32)
    n_output    = np.zeros((P,), dtype=np.int32)
    n_nodes     = np.zeros((P,), dtype=np.int32)
    n_conns     = np.zeros((P,), dtype=np.int32)

    for i, g in enumerate(genomes):
        # ----- nodes -----
        node_ids = sorted(g.nodes.keys())
        id_to_idx = {nid: j for j, nid in enumerate(node_ids)}
        nn = len(node_ids)
        if nn > max_nodes:
            raise ValueError(f"Genome {i}: n_nodes {nn} > MAX_NODES {max_nodes}")
        n_nodes[i] = nn

        # fill node_type row
        for nid, jidx in id_to_idx.items():
            try:
                nd_type = g.nodes[nid].type
                node_type[i, jidx] = NODE_TYPE_MAP[nd_type]
                if nd_type in (NodeType.HIDDEN, NodeType.OUTPUT):
                    node_activation[i, jidx] = ACT_TYPE_MAP[g.nodes[nid].activation]
            except KeyError:
                raise KeyError(f"Unknown node type: {nd_type}")

        # count inputs/outputs
        n_input[i]  = sum(1 for nid in node_ids if g.nodes[nid].type == NodeType.INPUT)
        n_output[i] = sum(1 for nid in node_ids if g.nodes[nid].type == NodeType.OUTPUT)
        # Sanity check with expected obs_dim / act_dim
        assert n_input[i] == obs_dim, f"Expected {obs_dim} inputs, got {n_input}"
        assert n_output[i] == act_dim, f"Expected {act_dim} outputs, got {n_output}"

        # ----- connections -----
        conns = list(g.connections.values()) if isinstance(g.connections, dict) else list(g.connections)
        nc = len(conns)
        if nc > max_conns:
            raise ValueError(f"Genome {i}: n_conns {nc} > MAX_CONNS {max_conns}")
        n_conns[i] = nc

        for k, c in enumerate(conns):
            conn_in[i, k]      = id_to_idx[c.in_id]
            conn_out[i, k]     = id_to_idx[c.out_id]
            conn_weight[i, k]  = float(c.weight)
            conn_enabled[i, k] = bool(c.enabled)

    # convert to JAX arrays
    params_batch = {
        "node_type":   jnp.array(node_type),
        "node_activation": jnp.array(node_activation),
        "conn_in":     jnp.array(conn_in),
        "conn_out":    jnp.array(conn_out),
        "conn_weight": jnp.array(conn_weight),
        "conn_enabled":jnp.array(conn_enabled),
        "n_input":     jnp.array(n_input),
        "n_output":    jnp.array(n_output),
        "n_nodes":     jnp.array(n_nodes),
        "n_conns":     jnp.array(n_conns),
    }
    return params_batch
