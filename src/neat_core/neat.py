from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import copy
import numpy as np

from src.neat_core.genome import Genome, make_minimal_genome
from src.neat_core.mutation import (
    mutate_weights,
    mutate_add_connection,
    mutate_add_node,
    mutate_activation, 
    crossover_genomes,
    InnovationTracker,
)


@dataclass
class Individual:
    genome: Genome
    fitness: float = 0.0


@dataclass
class NeatHyperParams:
    pop_size: int = 50
    elite_frac: float = 0.1
    parent_frac: float = 0.5
    p_add_conn: float = 0.05
    p_add_node: float = 0.03
    p_mutate_activation: float = 0.03

    # later: speciation params, compatibility coeffs, etc.


class Neat:
    """PrettyNEAT-style NEAT engine with ask/tell API."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hyp: Optional[NeatHyperParams] = None,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hyp = hyp or NeatHyperParams()
        self.rng = np.random.default_rng(seed)
        self.innov = InnovationTracker()

        self.generation = 0
        self.population: list[Individual] = self._init_population()

    def _init_population(self) -> list[Individual]:
        pop: list[Individual] = []
        for _ in range(self.hyp.pop_size):
            g = make_minimal_genome(self.obs_dim, self.act_dim)
            mutate_weights(g, self.rng, prob_perturb=1.0, sigma=0.1, reset_scale=0.5)
            pop.append(Individual(genome=g, fitness=0.0))
        return pop

    # -----------------------
    # ask / tell interface
    # -----------------------
    def ask(self) -> list[Genome]:
        """Return genomes to evaluate for this generation."""
        # Just return in current order
        return [ind.genome for ind in self.population]

    def tell(self, fitnesses: np.ndarray) -> None:
        """Receive fitness values for the last ask() population and evolve.

        fitnesses: shape (pop_size,)
        """
        assert fitnesses.shape[0] == len(self.population)
        # Attach fitness
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness = float(fit)

        # Optionally: rank-based fitness scaling here later
        self.population = self._reproduce()
        self.generation += 1

    # -----------------------
    # internal evolution
    # -----------------------
    def _reproduce(self) -> list[Individual]:
        """Simple mutation-only reproduction."""
        hyp = self.hyp
        pop_size = hyp.pop_size

        # Sort by fitness descending
        population = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)

        n_elite = max(1, int(hyp.elite_frac * pop_size))
        n_parent_pool = max(1, int(hyp.parent_frac * pop_size))

        new_population: list[Individual] = []

        # 1) carry elites
        for ind in population[:n_elite]:
            new_population.append(Individual(
                genome=copy.deepcopy(ind.genome),
                fitness=ind.fitness,
            ))

        # 2) mutated children from parent pool
        while len(new_population) < pop_size:
            # parent = self.rng.choice(population[:n_parent_pool])
            parent_pool = population[:n_parent_pool]
            idx1, idx2 = self.rng.choice(len(parent_pool), size=2, replace=False)
            parent1 = parent_pool[idx1]
            parent2 = parent_pool[idx2]

            # Determine the dominant (fitter) and non-dominant parent
            is_parent1_dominant = (parent1.fitness > parent2.fitness) or \
                                (parent1.fitness == parent2.fitness and self.rng.random() < 0.5)

            dominant = parent1 if is_parent1_dominant else parent2
            submissive = parent2 if is_parent1_dominant else parent1

            # The function handles dominance based on fitness
            child_genome = crossover_genomes(dominant.genome, submissive.genome, self.rng)
            # child_genome = copy.deepcopy(dominant.genome)

            # Always mutate weights
            mutate_weights(child_genome, self.rng)
            # Sometimes add connection / node
            if self.rng.random() < hyp.p_add_conn:
                mutate_add_connection(child_genome, self.innov, self.rng)
            if self.rng.random() < hyp.p_add_node:
                mutate_add_node(child_genome, self.innov, self.rng)
            if self.rng.random() < hyp.p_mutate_activation:
                mutate_activation(child_genome, self.rng, prob_mutate=0.5)

            new_population.append(Individual(genome=child_genome, fitness=0.0))

        return new_population

    # convenience: get current best
    def get_best(self) -> Individual:
        return max(self.population, key=lambda ind: ind.fitness)
