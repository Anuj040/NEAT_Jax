import numpy as np
from src.neat_core.genome import make_minimal_genome
from src.jax_neat.convert import genome_to_jax
from src.jax_neat.policy import jax_forward_jit

def test():
    obs_dim = 12
    act_dim = 4
    g = make_minimal_genome(obs_dim, act_dim)

    jg = genome_to_jax(g, obs_dim, act_dim)
    x = np.random.randn(obs_dim).astype(np.float32)

    y = jax_forward_jit(jg, x)
    print("Output:", y)

if __name__ == "__main__":
    test()
