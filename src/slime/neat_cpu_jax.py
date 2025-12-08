import slimevolleygym
import numpy as np
import jax.numpy as jnp
from src.jax_neat.convert import genome_to_jax
from src.jax_neat.policy import jax_forward
from src.neat_core.neat import Neat, NeatHyperParams
from src.neat_core.genome import Genome

try:
    import gym
except:
    pass

def slime_policy_jax(jg, obs_np: np.ndarray) -> np.ndarray:
    """Wrapper used by gym loop."""
    obs = jnp.array(obs_np, dtype=jnp.float32)
    raw = jax_forward(jg, obs)  # (3,)
    # Binary actions (MultiBinary(3))
    act = (raw > 0.0).astype(jnp.int32)
    return np.array(act)  # back to numpy for slimevolleygym


def eval_genome_slime_jax(g:Genome, episodes:int=3, max_steps:int = 1000, render: bool=False) -> float:

    env = gym.make("SlimeVolley-v0")
    jg = genome_to_jax(g, obs_dim=12, act_dim=3)  # fixed dims


    total = 0.0
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < max_steps:
            action = slime_policy_jax(jg, obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            steps += 1
        total += ep_reward
    env.close()
    return total / episodes  # fitness

def train_neat_slime_jax(
        generations: int = 20,
        episodes_per_genome: int = 3,
        pop_size: int = 20,
    ):
    env = gym.make("SlimeVolley-v0")
    obs_dim = env.observation_space.shape[0] #12
    act_dim = env.action_space.shape[0]  # MutltiBinary(3) -> Shape (3,)
    env.close()

    hyparams = NeatHyperParams(
        pop_size=pop_size,
        elite_frac=0.1,
        parent_frac=0.5,
        p_add_conn=0.1,
        p_add_node=0.05,
    )
    neat = Neat(obs_dim, act_dim, hyparams, seed=42)
    for gen in range(1, generations + 1):
        genomes = neat.ask()
        fitnesses = []
        for g in genomes:
            fit = eval_genome_slime_jax(g, episodes=episodes_per_genome, render=False)
            fitnesses.append(fit)
    
        fitnesses = jnp.asarray(fitnesses, dtype=jnp.float32)   
        neat.tell(fitnesses)
        best = neat.get_best()
        print(f"Generation {gen}: Best fitness: {best.fitness} Average fitness: {fitnesses.mean():}, Genome size: {len(best.genome.nodes)} nodes, {len(best.genome.connections)} connections")



if __name__ == "__main__":
    train_neat_slime_jax(
        generations=5,
        episodes_per_genome=2,
        pop_size=10,
    )
