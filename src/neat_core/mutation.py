from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from copy import deepcopy
from typing import Optional

from src.neat_core.genome import Genome, NodeType, ConnectionGene, NodeGene, ActivationType, ACT_TYPE_MAP


@dataclass
class InnovationTracker:
    """Global innovation counter shared across the whole population."""
    current: int = 0

    def next(self) -> int:
        self.current += 1
        return self.current


def mutate_weights(
    genome: Genome,
    rng: np.random.Generator,
    prob_perturb: float = 0.9,
    sigma: float = 0.5,
    reset_scale: float = 1.0,
) -> None:
    """Mutate connection weights in-place.

    - With prob_perturb, add N(0, sigma) noise.
    - Otherwise, reset weight ~ N(0, reset_scale).
    """
    for c in genome.connections:
        if rng.random() < prob_perturb:
            c.weight += float(rng.normal(0.0, sigma))
        else:
            c.weight = float(rng.normal(0.0, reset_scale))


def _connection_exists(genome: Genome, in_id: int, out_id: int) -> bool:
    for c in genome.connections:
        if c.in_id == in_id and c.out_id == out_id:
            return True
    return False


def mutate_add_connection(
    genome: Genome,
    innov: InnovationTracker,
    rng: np.random.Generator,
    max_tries: int = 20,
    weight_scale: float = 0.5,
) -> bool:
    """Try to add a new connection.

    Returns True if a connection was added, False otherwise.
    We enforce DAG-ness by only allowing in_id < out_id.
    """
    node_ids = sorted(genome.nodes.keys())

    for _ in range(max_tries):
        in_id = int(rng.choice(node_ids))
        out_id = int(rng.choice(node_ids))

        # Enforce direction and type constraints
        if in_id >= out_id:
            continue

        in_node = genome.nodes[in_id]
        out_node = genome.nodes[out_id]

        # Valid sources: INPUT, HIDDEN, BIAS
        if in_node.type not in (NodeType.INPUT, NodeType.HIDDEN, NodeType.BIAS):
            continue

        # Valid targets: HIDDEN, OUTPUT
        if out_node.type not in (NodeType.HIDDEN, NodeType.OUTPUT):
            continue

        # No duplicate connection
        if _connection_exists(genome, in_id, out_id):
            continue

        # Success: create new connection
        new_conn = ConnectionGene(
            in_id=in_id,
            out_id=out_id,
            weight=float(rng.normal(0.0, weight_scale)),
            enabled=True,
            innovation=innov.next(),
        )
        genome.connections.append(new_conn)
        return True

    # Failed to find a legal pair
    return False


def mutate_add_node(
    genome: Genome,
    innov: InnovationTracker,
    rng: np.random.Generator,
) -> bool:
    """Add a new node by splitting an existing enabled connection.

    Returns True if a node was added, False otherwise.
    """
    enabled_conns = [c for c in genome.connections if c.enabled]
    if not enabled_conns:
        return False

    # Pick a random connection to split
    conn = rng.choice(enabled_conns)
    conn.enabled = False  # disable old connection

    # New node id = max existing id + 1
    new_id = max(genome.nodes.keys()) + 1

    genome.nodes[new_id] = NodeGene(id=new_id, type=NodeType.HIDDEN)

    # Two new connections:
    # in -> new (weight = 1.0)
    # new -> out (weight = old weight)
    in_to_new = ConnectionGene(
        in_id=conn.in_id,
        out_id=new_id,
        weight=1.0,
        enabled=True,
        innovation=innov.next(),
    )
    new_to_out = ConnectionGene(
        in_id=new_id,
        out_id=conn.out_id,
        weight=conn.weight,  # preserve old behavior initially
        enabled=True,
        innovation=innov.next(),
    )
    genome.connections.append(in_to_new)
    genome.connections.append(new_to_out)

    return True


def mutate_activation(genome: Genome, rng: np.random.Generator, prob_mutate: float = 0.1) -> None:
    """Mutate the activation function of a node with probability `prob_mutate`."""
    
    # Select nodes that are mutable (HIDDEN and OUTPUT)
    mutable_nodes = [
        node for node in genome.nodes.values()
        if node.type in (NodeType.HIDDEN, NodeType.OUTPUT)
    ]

    if not mutable_nodes:
        return

    # Iterate through mutable nodes and apply mutation probability
    for node in mutable_nodes:
        if rng.random() < prob_mutate:
            # Filter choices to those that are NOT the current activation
            available_choices = [
                act for act in ACT_TYPE_MAP.keys()
                if act != node.activation
                and act != ActivationType.NULL
            ]
            
            if not available_choices:
                continue 

            # Select and assign the new activation
            idx = rng.integers(0, len(available_choices))
            node.activation = available_choices[idx]


def crossover_genomes(dominant: Genome, submissive: Genome, rng: np.random.Generator) -> Genome:
    """
    Performs NEAT crossover to create a child genome.
    The parent with the higher fitness is chosen as the dominant parent.
    but here we rely on the caller to determine dominance if fitness is equal.
    """
    
    child = deepcopy(dominant) 
    child.connections = []
    
    all_nodes = {**dominant.nodes, **submissive.nodes}
    child.nodes = all_nodes
    # Inherit Connections (Genes)
    # Maps for easy lookup by innovation number
    dominant_conn_map = {c.innovation: c for c in dominant.connections}
    submissive_conn_map = {c.innovation: c for c in submissive.connections}
    
    all_innovations = set(dominant_conn_map.keys()) | set(submissive_conn_map.keys())
    
    for innov_num in sorted(list(all_innovations)):
        conn_dominant = dominant_conn_map.get(innov_num)
        conn_submissive = submissive_conn_map.get(innov_num)
        
        chosen_conn: Optional[ConnectionGene] = None
        
        if conn_dominant and conn_submissive:
            if rng.random() < 0.5:
                chosen_conn = conn_dominant
            else:
                chosen_conn = conn_submissive
            
            # NEAT typically has a rule: if one is disabled, the disabled one is inherited 
            # with 75% probability, regardless of fitness. For simplicity, we stick to 
            # random choice of the two connection copies, assuming they have the same structure.
            
        elif conn_dominant:
            # Disjoint/Excess Gene in Dominant Parent: Always inherited
            chosen_conn = conn_dominant
        
        elif conn_submissive:
            # Disjoint/Excess Gene in Submissive Parent: Never inherited
            pass
            
        if chosen_conn:
            # Create a deep copy of the gene for the child
            child.connections.append(deepcopy(chosen_conn))
    
    # Re-verify node set (optional, but safer)
    all_node_ids = set(child.nodes.keys())
    for conn in child.connections:
        if conn.in_id not in all_node_ids:
            # This should only happen if disjoint genes were allowed from submissive parent
            # or if the genome structure is incomplete. We skip explicit node merging 
            # for now, assuming your `Genome` ensures node completeness upon creation.
            pass

    return child