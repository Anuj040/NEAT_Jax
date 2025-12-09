import time
import numpy as np
import jax.random as random
import jax.numpy as jnp
import jax
from evojax.task.slimevolley import SlimeVolley
from pathlib import Path

from src.jax_neat.convert import genomes_to_params_batch
from src.slime.neat_cpu_jax import slime_policy_jax
from src.neat_core.neat import Neat, NeatHyperParams
from src.neat_core.genome import Genome
from src.slime.neat_evojax import eval_with_render_evojax
from src.neat_core.visualize import GenomeEvolutionRecorder

OBS_DIM = 12  # SlimeVolley state observation size (fixed)
ACT_DIM = 3   # SlimeVolley action size (MultiBinary(3))

neat_policy_batched = jax.vmap(
    slime_policy_jax,
    in_axes=(0, 0),   # params_batch[0], obs_batch[0] go together, etc.
    out_axes=0,
)


def rollout_batched(
    params_batch: dict,     # dict of arrays with shape (B, ...)
    episodes: int,
    max_steps: int,
    rng_seed: int | None = None,
) -> jnp.ndarray:
    """
    Fully JAXed batched rollout for NEAT policies in SlimeVolley.

    params_batch: params for P genomes, shape (P, ...)
    episodes: number of episodes per genome (parallelized as env slots)
    max_steps: time horizon
    key: PRNG key

    Returns:
        fitnesses: (P,) average return per genome over episodes.
    """
    # Number of genomes
    P = params_batch["n_nodes"].shape[0]
    B = P * episodes  # total env slots

    # Repeat params for episodes axis: shape (B, ...)
    params_slots = {
        k: jnp.repeat(v, repeats=episodes, axis=0)
        for k, v in params_batch.items()
    }
    if rng_seed is None:
        rng_seed = int(time.time() * 1e6) & 0xFFFFFFFF
    key = random.PRNGKey(rng_seed)
    env = SlimeVolley(test=False, max_steps=max_steps)
    keys = random.split(key, num=B)
    state = env.reset(keys)

    def step_fn(carry, t):
        state, returns, done_carry = carry  # done_carry: (B,) bool
        target_dtype = state.game_state.action_right.dtype
        # If done, we want to skip stepping those envs
        obs = state.obs  # (B, obs_dim)

        # Compute actions for all envs
        actions = neat_policy_batched(params_slots, obs)  # (B, 3)

        # Step env
        next_state, reward, done = env.step(state, actions)
        reward = jnp.squeeze(reward)  # (B,)
        done = jnp.squeeze(done)      # (B,)
        # Only accumulate reward if not already done
        # (This allows env to keep stepping but we freeze returns per episode.)
        # active = ~done_carry
        returns = returns + reward#jnp.where(active, reward, 0)

        # Once done, stay done
        done_carry = jnp.logical_or(done_carry, done)
        corrected_action_right = next_state.game_state.action_right.astype(target_dtype)
        new_game_state = next_state.game_state.replace(action_right=corrected_action_right)
        next_state = next_state.replace(game_state=new_game_state)
        new_carry = (next_state, returns, done_carry)
        return new_carry, None

    B_bool = jnp.zeros((B,), dtype=bool)
    init_returns = jnp.zeros((B,), dtype=jnp.int32)
    (_, final_returns, _), _ = jax.lax.scan(
        step_fn,
        (state, init_returns, B_bool),
        jnp.arange(max_steps),
    )

    returns_matrix = final_returns.reshape(P, episodes).astype(jnp.float32) # (P, episodes)
    return jnp.mean(returns_matrix, axis=1)

def evaluate_genome_slime_evojax(
    genomes,
    episodes: int = 3,
    max_steps: int = 3000,
    rng_seed: int | None = None,
    pop_size: int | None = None,
) -> np.ndarray:
    """
    Evaluate all NEAT genomes in the batch using EvoJAX's SlimeVolley task.

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

    # return evaluate_genome_slime_evojax(
    #         genomes=(genomes_to_params_batch(genomes, OBS_DIM, ACT_DIM)),
    #         episodes=episodes,
    #         max_steps=max_steps,
    #         rng_seed=base_seed,
    #         pop_size=pop_size
    #     )
    fitnesses_jax = rollout_batched(
        params_batch=genomes_to_params_batch(genomes, OBS_DIM, ACT_DIM),
        episodes=episodes,
        max_steps=max_steps,
        rng_seed=base_seed,
    )
    return np.array(fitnesses_jax, dtype=np.float32)

def train_neat_on_slime(generations: int = 20, episodes_per_genome: int = 3, pop_size: int = 20):
    hyparams = NeatHyperParams(
        pop_size=pop_size,
        elite_frac=0.1,
        parent_frac=0.5,
        p_add_conn=0.1,
        p_add_node=0.05,
        p_mutate_activation=0.03,
    )
    neat = Neat(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hyp=hyparams,
        seed=42,
    )
    
    log_dir = Path('./log/slimevolley')
    gif_path = log_dir / "snapshots_nocross"
    rec = GenomeEvolutionRecorder(gif_path)
    for gen in range(generations):
        genomes:list[Genome] = neat.ask()
        fitnesses = evaluate_population_evojax(genomes, episodes=episodes_per_genome, max_steps=1000, pop_size=pop_size)
        neat.tell(fitnesses)

        best = neat.get_best()
        print(
            f"Gen {gen:03d}  best_fit={best.fitness:.3f}  "
            f"nodes={len(best.genome.nodes)}  conns={len(best.genome.connections)}"
        )
        if gen % 10 == 0 or gen == generations - 1:
            rec.save_genome_frame(best.genome, label=f"gen {gen}")

    # After training, you can test best genome with rendering if SlimeVolley supports it.
    best = neat.get_best()
    final_gif_path = gif_path / "neat_evolution.gif"
    rec.make_gif(final_gif_path, duration_ms=500)
    # Visuailize best genome
    eval_with_render_evojax(best.genome, episodes=1, max_steps=1000, log_dir='./log/slimevolley')

if __name__ == "__main__":
    import time
    start = time.time()
    train_neat_on_slime(
        generations=50,
        episodes_per_genome=5,
        pop_size=200,
        )
    print(f"Total training time: {time.time() - start} seconds")
