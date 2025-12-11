from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import random

class NodeType(Enum):
    INPUT = auto()
    HIDDEN = auto()
    OUTPUT = auto()
    BIAS = auto()

NODE_TYPE_MAP = {
    NodeType.INPUT: 1,
    NodeType.HIDDEN: 2,
    NodeType.OUTPUT: 3,
    NodeType.BIAS: 4,
}

class ActivationType(str, Enum):
    NULL = "null"
    TANH = "tanh"
    RELU = "relu"
    SIGMOID = "sigmoid"
    LINEAR = "linear"   # identity

ACT_TYPE_MAP = {
    ActivationType.NULL: 0,
    ActivationType.TANH: 1,
    ActivationType.RELU: 2,
    ActivationType.SIGMOID: 3,
    ActivationType.LINEAR: 4,
}

@dataclass
class NodeGene:
    id: int
    type: NodeType
    activation: ActivationType = ActivationType.SIGMOID  # default; ignored for INPUT/BIAS


@dataclass
class ConnectionGene:
    in_id: int
    out_id: int
    weight: float
    enabled: bool
    innovation: int


@dataclass
class Genome:
    """NEAT genome: nodes + connections, feed-forward evaluation only."""
    nodes: dict[int, NodeGene]
    connections: list[ConnectionGene]

    def _build_eval_order(self) -> list[int]:
        """Return an evaluation order for non-input nodes.

        Simple version: sort all HIDDEN + OUTPUT node IDs.
        (DAG structure: only adding edges from lower -> higher ID.)
        """
        hidden_and_output = [
            (nid, n) for nid, n in self.nodes.items()
            if n.type in (NodeType.HIDDEN, NodeType.OUTPUT)
        ]
        return sorted(hidden_and_output)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute network output for a single observation.

        x: shape (num_inputs,)
        returns: shape (num_outputs,)
        """
        # Assign input (and bias) values.
        values: dict[int, float] = {}

        input_ids = sorted(
            nid for nid, n in self.nodes.items()
            if n.type == NodeType.INPUT
        )
        assert x.shape[0] == len(input_ids), (
            f"Expected {len(input_ids)} inputs, got {x.shape[0]}"
        )

        for nid, val in zip(input_ids, x):
            values[nid] = float(val)

        # Bias nodes (if any) always output 1.0
        for nid, n in self.nodes.items():
            if n.type == NodeType.BIAS:
                values[nid] = 1.0

        # Compute hidden and output nodes in order.
        eval_order = self._build_eval_order()

        for nid, n in eval_order:
            incoming = [
                c for c in self.connections
                if c.enabled and c.out_id == nid
            ]
            s = 0.0
            for c in incoming:
                # Missing in_id value should not happen if DAG is respected.
                s += values.get(c.in_id, 0.0) * c.weight
            # Activation
            values[nid] = self.activate(s, n.activation)

        # Collect outputs in sorted ID order.
        output_ids = sorted(
            nid for nid, n in self.nodes.items()
            if n.type == NodeType.OUTPUT
        )
        outputs = np.array([values[nid] for nid in output_ids], dtype=np.float32)
        return outputs
    
    @staticmethod
    def activate(z: float, act: ActivationType) -> float:
        # Define activation dispatcher
        if act == ActivationType.TANH:
            return float(np.tanh(z))
        elif act == ActivationType.RELU:
            return float(max(0.0, z))
        elif act == ActivationType.SIGMOID:
            return float(1.0 / (1.0 + np.exp(-z)))
        elif act == ActivationType.LINEAR:
            return float(z)
        else:
            return float(z)

def make_minimal_genome(obs_dim: int, act_dim: int, hyps) -> Genome:
    """Create a minimal NEAT genome: inputs + bias fully connected to outputs.

    No hidden nodes. This matches classic NEAT's 'start simple' strategy.
    """
    nodes: dict[int, NodeGene] = {}
    node_id = 0

    # Inputs
    input_ids: list[int] = []
    for _ in range(obs_dim):
        node_id += 1
        nodes[node_id] = NodeGene(id=node_id, type=NodeType.INPUT)
        input_ids.append(node_id)

    # Bias
    node_id += 1
    bias_id = node_id
    nodes[bias_id] = NodeGene(id=bias_id, type=NodeType.BIAS)

    # Outputs
    output_base = hyps.MAX_NODES - act_dim
    output_ids = list(range(output_base, hyps.MAX_NODES))
    for out_id in output_ids:
        nodes[out_id] = NodeGene(id=out_id, type=NodeType.OUTPUT)

    # Fully connect (inputs + bias) → outputs with random weights.
    connections: list[ConnectionGene] = []
    innovation = 0
    for in_id in input_ids + [bias_id]:
        for out_id in output_ids:
            innovation += 1
            connections.append(ConnectionGene(
                in_id=in_id,
                out_id=out_id,
                weight=float(np.random.randn() * 0.5),
                enabled=True,
                innovation=innovation,
            ))

    return Genome(nodes=nodes, connections=connections)
