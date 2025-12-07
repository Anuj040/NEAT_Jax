import numpy as np
from src.neat_core.genome import make_minimal_genome

def main():
    obs_dim = 12   # SlimeVolley-like
    act_dim = 3    # 3 binary actions

    g = make_minimal_genome(obs_dim, act_dim)

    x = np.random.randn(obs_dim).astype(np.float32)
    y = g.forward(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Output values:", y)

    # Example mapping to MultiBinary(3) action:
    action = (y > 0.0).astype(np.int8)
    print("Binary action:", action)

if __name__ == "__main__":
    main()
