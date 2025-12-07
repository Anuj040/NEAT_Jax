from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import copy
import numpy as np

from src.neat_core.genome import Genome, make_minimal_genome
from src.neat_core.mutation import (
    mutate_weights,
    mutate_add_connection,
    mutate_add_node,
    InnovationTracker,
)


@dataclass
class Individual:
    genome: Genome
    fitness: float = 0.0


def make_initial_population(
    pop_size: int,
    obs_dim: int,
    act_dim: int,
    rng: np.random.Generator,
) -> list[Individual]:
    """Create a population of minimal genomes (no hidden nodes yet)."""
    population: list[Individual] = []
    for _ in range(pop_size):
        g = make_minimal_genome(obs_dim, act_dim)
        # Small random tweak to avoid identical clones
        mutate_weights(g, rng, prob_perturb=1.0, sigma=0.1, reset_scale=0.5)
        population.append(Individual(genome=g))
    return population


def evaluate_population(
    population: list[Individual],
    fitness_fn: Callable[[Genome, np.random.Generator], float],
    rng: np.random.Generator,
) -> None:
    """Compute fitness for each individual in-place."""
    for ind in population:
        ind.fitness = fitness_fn(ind.genome, rng)


def reproduce(
    population: list[Individual],
    pop_size: int,
    rng: np.random.Generator,
    innov: InnovationTracker,
    elite_frac: float = 0.1,
    parent_frac: float = 0.5,
    p_add_conn: float = 0.05,
    p_add_node: float = 0.03,
) -> list[Individual]:
    """Create the next generation via selection + mutation only.

    - Keep top elite_frac as-is (copied).
    - Sample parents from top parent_frac for mutation-only offspring.
    """
    assert 0 < elite_frac <= parent_frac <= 1.0

    # Sort by fitness descending
    population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

    n_elite = max(1, int(elite_frac * pop_size))
    n_parent_pool = max(1, int(parent_frac * pop_size))

    new_population: list[Individual] = []

    # Keep elites (deepcopy to prevent overwrite genomes later)
    for ind in population[:n_elite]:
        new_population.append(Individual(genome=copy.deepcopy(ind.genome),
                                         fitness=ind.fitness))

    # Filling the rest with mutated children
    while len(new_population) < pop_size:
        parent = rng.choice(population[:n_parent_pool])
        child_genome = copy.deepcopy(parent.genome)

        # Always mutate weights
        mutate_weights(child_genome, rng)

        # Sometimes add a connection
        if rng.random() < p_add_conn:
            mutate_add_connection(child_genome, innov, rng)

        # Sometimes add a node
        if rng.random() < p_add_node:
            mutate_add_node(child_genome, innov, rng)

        new_population.append(Individual(genome=child_genome, fitness=0.0))

    return new_population
