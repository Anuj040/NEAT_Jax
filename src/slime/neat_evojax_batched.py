import time
import numpy as np
import jax.random as random
import jax.numpy as jnp
import os
import jax
from evojax.task.slimevolley import SlimeVolley

from src.jax_neat.convert import genomes_to_params_batch, genome_to_jax
from src.slime.neat_cpu_jax import slime_policy_jax
from src.neat_core.neat import Neat, NeatHyperParams
from src.neat_core.genome import Genome
from src.slime.neat_evojax import eval_with_render_evojax
OBS_DIM = 12  # SlimeVolley state observation size (fixed)
ACT_DIM = 3   # SlimeVolley action size (MultiBinary(3))

neat_policy_batched = jax.vmap(
    slime_policy_jax,
    in_axes=(0, 0),   # params_batch[0], obs_batch[0] go together, etc.
    out_axes=0,
)

def evaluate_genome_slime_evojax(
    genomes,
    episodes: int = 3,
    max_steps: int = 3000,
    rng_seed: int | None = None,
    pop_size: int | None = None,
) -> np.ndarray:
    """
    Evaluate a single NEAT genome using EvoJAX's SlimeVolley task.

    - Uses the built-in simple AI opponent (test=False).
    - Returns average episodic reward over `episodes`.
    """

    # JAX PRNG
    if rng_seed is None:
        rng_seed = int(time.time() * 1e6) & 0xFFFFFFFF
    key = random.PRNGKey(rng_seed)

    # Create EvoJAX task (train mode)
    env = SlimeVolley(test=False, max_steps=max_steps)

    total_return = np.zeros(pop_size, dtype=np.float32)

    keys = random.split(key, num=pop_size*episodes)
    state = env.reset(keys)
    ep_return = jnp.zeros(pop_size * episodes, dtype=jnp.float32)
    step = 0
    genomes_epoch_batched = {key: jnp.repeat(values, episodes, axis=0) for key, values in genomes.items()}
    for step in range(max_steps):
        action = neat_policy_batched(genomes_epoch_batched, state.obs)
        # Step only for active indices
        state, reward, _ = env.step(state, action)
        # Update ep_return for active indices
        ep_return += reward

    total_return = np.array(ep_return).reshape(pop_size, episodes)
    return total_return.mean(axis=1)


def evaluate_population_evojax(
    genomes: list,
    episodes: int = 3,
    max_steps: int = 3000,
    base_seed: int | None = None,
    pop_size: int | None = None,
) -> np.ndarray:
    """
    Evaluate a list of genomes using EvoJAX SlimeVolley.

    Returns:
        fitnesses: np.ndarray of shape (len(genomes),)
    """
    if base_seed is None:
        base_seed = int(time.time() * 1e6) & 0xFFFFFFFF

    return evaluate_genome_slime_evojax(
            genomes=(genomes_to_params_batch(genomes, OBS_DIM, ACT_DIM)),
            episodes=episodes,
            max_steps=max_steps,
            rng_seed=base_seed,
            pop_size=pop_size
        )

def train_neat_on_slime(generations: int = 20, episodes_per_genome: int = 3, pop_size: int = 20):
    hyparams = NeatHyperParams(
        pop_size=pop_size,
        elite_frac=0.1,
        parent_frac=0.5,
        p_add_conn=0.1,
        p_add_node=0.05,
    )
    neat = Neat(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hyp=hyparams,
        seed=42,
    )

    for gen in range(generations):
        genomes = neat.ask()  # list[Genome]
        fitnesses = evaluate_population_evojax(genomes, episodes=episodes_per_genome, max_steps=10, pop_size=pop_size)#00)

        neat.tell(fitnesses)

        best = neat.get_best()
        print(
            f"Gen {gen:03d}  best_fit={best.fitness:.3f}  "
            f"nodes={len(best.genome.nodes)}  conns={len(best.genome.connections)}"
        )

    # After training, you can test best genome with rendering if SlimeVolley supports it.
    best = neat.get_best()
    # Visuailize best genome
    eval_with_render_evojax(best.genome, episodes=1, max_steps=1000)

if __name__ == "__main__":
    import time
    start = time.time()
    train_neat_on_slime(
        generations=5,
        episodes_per_genome=2,
        pop_size=10,
        )
    print(f"Total training time: {time.time() - start} seconds")
