import numpy as np

from src.neat_core.population import (
    make_initial_population,
    evaluate_population,
    reproduce,
)
from src.neat_core.mutation import InnovationTracker
from src.neat_core.genome import Genome


# XOR dataset: inputs (2,) -> target in {0, 1}
XOR_INPUTS = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=np.float32)

XOR_TARGETS = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)


def xor_fitness(genome: Genome, rng: np.random.Generator) -> float:
    """Higher fitness = lower MSE on XOR."""
    total_sq_err = 0.0
    for x, t in zip(XOR_INPUTS, XOR_TARGETS):
        y = genome.forward(x)[0]
        # Normalizing [-1, 1] -> [0, 1]
        y01 = 0.5 * (y + 1.0)
        total_sq_err += (y01 - t) ** 2
    mse = total_sq_err / len(XOR_INPUTS)
    return -float(mse)  # NEAT maximizes fitness, so we negate error


def main():
    rng = np.random.default_rng(42)
    innov = InnovationTracker()

    obs_dim = 2
    act_dim = 1
    pop_size = 50
    generations = 500

    population = make_initial_population(pop_size, obs_dim, act_dim, rng)
    print("\nInitial Random network:")
    for x, t in zip(XOR_INPUTS, XOR_TARGETS):
        y = population[0].genome.forward(x)[0]
        y01 = 0.5 * (y + 1.0)
        print(f"  x={x}  target={t}  pred01={y01:.3f}")

    for gen in range(1, generations + 1):
        evaluate_population(population, xor_fitness, rng)

        best = max(population, key=lambda ind: ind.fitness)
        best_mse = -best.fitness
        if gen % 50 == 0 or gen == 1:
            print(f"Gen {gen:03d}: best_fitness={best.fitness:.4f}  best_MSE={best_mse:.4f}  "
                f"nodes={len(best.genome.nodes)}  conns={len(best.genome.connections)}")

        # Simple stopping condition (optional)
        if best_mse < 0.02:
            print("Reached good XOR solution, stopping early.")
            break

        population = reproduce(population, pop_size, rng, innov, p_add_node=0.2, p_add_conn= 0.3)

    # Show final best predictions
    best = max(population, key=lambda ind: ind.fitness)
    print("\nFinal best network:")
    for x, t in zip(XOR_INPUTS, XOR_TARGETS):
        y = best.genome.forward(x)[0]
        y01 = 0.5 * (y + 1.0)
        print(f"  x={x}  target={t}  pred01={y01:.3f}")

if __name__ == "__main__":
    main()
