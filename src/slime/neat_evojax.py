import time
import numpy as np
import jax.random as random
import jax.numpy as jnp
import os
import jax
from evojax.task.slimevolley import SlimeVolley

from src.jax_neat.convert import genome_to_jax
from src.slime.neat_cpu_jax import slime_policy_jax
from src.neat_core.neat import Neat, NeatHyperParams
from src.neat_core.genome import Genome

OBS_DIM = 12  # SlimeVolley state observation size (fixed)
ACT_DIM = 3   # SlimeVolley action size (MultiBinary(3))


def evaluate_genome_slime_evojax(
    genome,
    episodes: int = 3,
    max_steps: int = 3000,
    rng_seed: int | None = None,
) -> float:
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

    # Convert NEAT genome to JAX representation
    jg = genome_to_jax(genome, obs_dim=OBS_DIM, act_dim=ACT_DIM)

    total_return = 0.0
    keys = random.split(key, num=episodes)

    for ep in range(episodes):
        ep_key = keys[ep:ep+1]
        state = env.reset(ep_key)   # state is a JAX pytree with .obs, .done, etc.
        done = False
        ep_return = 0.0
        step = 0

        # Note: we stay in Python loop here. That's okay for now.
        while not done and step < max_steps:
            # state.obs should be a numpy or JAX array of shape (OBS_DIM,)
            obs = np.array(state.obs, dtype=np.float32)
            action = np.array(slime_policy_jax(jg, obs[0]))  # numpy int32 (3,)
            # SlimeVolley expects actions as numpy / jax arrays; both usually work.
            state, reward, done = env.step(state, jnp.expand_dims(action, 0))
            ep_return += float(reward[0])
            step += 1

        total_return += ep_return

    return total_return / float(episodes)


def evaluate_population_evojax(
    genomes: list,
    episodes: int = 3,
    max_steps: int = 3000,
    base_seed: int | None = None,
) -> np.ndarray:
    """
    Evaluate a list of genomes using EvoJAX SlimeVolley.

    Returns:
        fitnesses: np.ndarray of shape (len(genomes),)
    """
    if base_seed is None:
        base_seed = int(time.time() * 1e6) & 0xFFFFFFFF

    fitnesses = []
    for idx, g in enumerate(genomes):
        seed = base_seed + idx
        fit = evaluate_genome_slime_evojax(
            genome=g,
            episodes=episodes,
            max_steps=max_steps,
            rng_seed=seed,
        )
        fitnesses.append(fit)

    return np.array(fitnesses, dtype=np.float32)

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
        fitnesses = evaluate_population_evojax(genomes, episodes=episodes_per_genome, max_steps=10)

        neat.tell(fitnesses)

        best = neat.get_best()
        print(
            f"Gen {gen:03d}  best_fit={best.fitness:.3f}  "
            f"nodes={len(best.genome.nodes)}  conns={len(best.genome.connections)}"
        )

    # After training, you can test best genome with rendering if SlimeVolley supports it.
    best = neat.get_best()
    # Visuailize best genome
    eval_with_render_evojax(best.genome, episodes=1, max_steps=10)

def eval_with_render_evojax(genome:Genome, episodes: int = 1, max_steps:int = 1000) -> None:
    jg = genome_to_jax(genome, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    test_env = SlimeVolley(test=True, max_steps=max_steps)

    total_return = 0.0
    log_dir = './log/slimevolley'
    rng_seed = 0xFFFFFFFF
    key = random.PRNGKey(rng_seed)
    keys = random.split(key, num=episodes)
    ep_num = 0
    ep_return = 0.0
    state = test_env.reset(keys[ep_num:ep_num+1])

    screens = []
    for _ in range(max_steps):
        obs = np.array(state.obs, dtype=np.float32)
        action = slime_policy_jax(jg, obs[0])
        state, reward, done = test_env.step(state, jnp.expand_dims(action, 0))
        single_task_state = jax.tree_util.tree_map(lambda x: x[0], state)
        screens.append(SlimeVolley.render(single_task_state))
        ep_return += float(reward[0])
        if done:
            print(f"Episode {ep_num} return: {ep_return}")
            ep_num += 1
            total_return +=  ep_return
            ep_return = 0.0
            if ep_num >= episodes:
                break
            state = test_env.reset(keys[ep_num:ep_num+1])
    gif_file = os.path.join(log_dir, 'slimevolley.gif')
    screens[0].save(gif_file, save_all=True, append_images=screens[1:],
                    duration=40, loop=0)
    print(f'GIF saved to {gif_file}.')

if __name__ == "__main__":
    import time
    start = time.time()
    train_neat_on_slime(
        generations=5,
        episodes_per_genome=2,
        pop_size=10,
        )
    print(f"Total training time: {time.time() - start} seconds")
