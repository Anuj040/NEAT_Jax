import gym
import slimevolleygym
import numpy as np

from src.neat_core.neat import Neat, NeatHyperParams
from src.neat_core.genome import Genome


def slime_policy(genome: Genome, obs: np.ndarray) -> np.ndarray:
    """NEAT policy: obs -> raw -> binarized action (MultiBinary(3))."""
    raw = genome.forward(obs.astype(np.float32))  # shape (3,)
    # Threshold at 0 → 1 if positive, 0 otherwise
    return (raw > 0.0).astype(np.int8)

def slime_fitness_vs_builtin(
    genome: Genome,
    episodes: int = 3,
    max_steps: int = 1000,
) -> float:
    """Evaluate a genome vs the built-in AI in SlimeVolley-v0.

    Returns average total reward over 'episodes' episodes.
    """
    env = gym.make("SlimeVolley-v0")
    total_reward = 0.0

    for ep in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0

        while not done and steps < max_steps:
            action = slime_policy(genome, obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1

        total_reward += ep_reward

    env.close()
    return total_reward / episodes

def train_neat_slime(
    generations: int = 20,
    episodes_per_genome: int = 3,
    pop_size: int = 20,
):
    # Create a temporary env to read obs/action dims
    env = gym.make("SlimeVolley-v0")
    obs_dim = env.observation_space.shape[0]   # 12
    act_dim = env.action_space.shape[0]        # MultiBinary(3) -> shape (3,)
    env.close()

    hyp = NeatHyperParams(
        pop_size=pop_size,
        elite_frac=0.1,
        parent_frac=0.5,
        p_add_conn=0.1,
        p_add_node=0.05,
    )
    neat = Neat(obs_dim=obs_dim, act_dim=act_dim, hyp=hyp, seed=123)

    for gen in range(1, generations + 1):
        genomes = neat.ask()

        # Evaluate each genome (sequentially; fine for small pop)
        fitnesses = []
        for g in genomes:
            fit = slime_fitness_vs_builtin(
                g,
                episodes=episodes_per_genome,
                max_steps=1000,
            )
            fitnesses.append(fit)

        fitnesses = np.array(fitnesses, dtype=np.float32)
        neat.tell(fitnesses)

        best = neat.get_best()
        print(
            f"Gen {gen:03d}: "
            f"best_fit={best.fitness:.3f}, "
            f"avg_fit={fitnesses.mean():.3f}, "
            f"nodes={len(best.genome.nodes)}, "
            f"conns={len(best.genome.connections)}"
        )

    # After training, run one evaluation with render
    best = neat.get_best()
    print("\nEvaluating best genome with render=True...")
    eval_with_render(best.genome, episodes=episodes_per_genome)


def eval_with_render(genome: Genome, episodes: int = 1, max_steps: int = 1000):
    env = gym.make("SlimeVolley-v0")
    for ep in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        while not done and steps < max_steps:
            # env.render()
            action = slime_policy(genome, obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        print(f"[Render] Episode {ep} total_reward={total_reward}")
    env.close()


if __name__ == "__main__":
    train_neat_slime(
        generations=10,          # keep small at first
        episodes_per_genome=10,   # small for speed
        pop_size=10,             # small pop for a quick test
    )
