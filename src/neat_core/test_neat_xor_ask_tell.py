import numpy as np
from src.neat_core.neat import Neat, NeatHyperParams
from src.neat_core.genome import Genome

XOR_INPUTS = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=np.float32)

XOR_TARGETS = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)


def xor_fitness(genome: Genome) -> float:
    total_sq_err = 0.0
    for x, t in zip(XOR_INPUTS, XOR_TARGETS):
        y = genome.forward(x)[0]
        y01 = 0.5 * (y + 1.0)
        total_sq_err += (y01 - t) ** 2
    mse = total_sq_err / len(XOR_INPUTS)
    return -float(mse)


def main():
    hyp = NeatHyperParams(
        pop_size=50,
        elite_frac=0.1,
        parent_frac=0.5,
        p_add_conn=0.1,
        p_add_node=0.1,
    )
    neat = Neat(obs_dim=2, act_dim=1, hyp=hyp, seed=42)

    for gen in range(1, 101):
        genomes = neat.ask()
        fitnesses = np.array([xor_fitness(g) for g in genomes], dtype=np.float32)
        neat.tell(fitnesses)

        best = neat.get_best()
        best_mse = -best.fitness
        if gen % 10 == 0:
            print(f"Gen {gen:03d}: best_MSE={best_mse:.4f}, "
                  f"nodes={len(best.genome.nodes)}, conns={len(best.genome.connections)}")

    best = neat.get_best()
    print("\nFinal best:")
    for x, t in zip(XOR_INPUTS, XOR_TARGETS):
        y = best.genome.forward(x)[0]
        y01 = 0.5 * (y + 1.0)
        print(f"x={x}, target={t}, pred01={y01:.3f}")

if __name__ == "__main__":
    main()
