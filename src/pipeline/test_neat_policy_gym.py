# test_neat_policy_gym.py
import gym
import slimevolleygym # Ensure the SlimeVolley env is registered
import numpy as np
from src.neat_core.genome import make_minimal_genome

def neat_policy(genome, obs: np.ndarray) -> np.ndarray:
    raw = genome.forward(obs)
    # Threshold → MultiBinary(3)
    return (raw > 0.0).astype(np.int8)

def main(episodes: int = 3, render: bool = False):
    env = gym.make("SlimeVolley-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n  # MultiBinary(3) but Gym wraps it differently; if this errors, just set act_dim = 3

    genome = make_minimal_genome(obs_dim, act_dim)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if render:
                env.render()
            action = neat_policy(genome, obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        print(f"Episode {ep} total_reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    main(render=False)
