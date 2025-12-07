import numpy as np

from src.neat_core.genome import make_minimal_genome
from src.neat_core.mutation import (
    InnovationTracker,
    mutate_weights,
    mutate_add_connection,
    mutate_add_node,
)


def main():
    rng = np.random.default_rng(0)
    innov = InnovationTracker()

    obs_dim = 12
    act_dim = 3
    g = make_minimal_genome(obs_dim, act_dim)

    print("Initial:")
    print("  num_nodes:", len(g.nodes))
    print("  num_conns:", len(g.connections))

    # Weight mutation
    mutate_weights(g, rng)
    print("After weight mutation:")
    print("  first 3 weights:", [c.weight for c in g.connections[:3]])

    # Add connection
    added_conn = mutate_add_connection(g, innov, rng)
    print("Add connection success:", added_conn)
    print("  num_conns:", len(g.connections))

    # Add node
    added_node = mutate_add_node(g, innov, rng)
    print("Add node success:", added_node)
    print("  num_nodes:", len(g.nodes))
    print("  num_conns:", len(g.connections))

    # Sanity: forward still works
    x = rng.normal(size=(obs_dim,), loc=0.0, scale=1.0)
    y = g.forward(x)
    print("Forward output shape:", y.shape)


if __name__ == "__main__":
    main()
