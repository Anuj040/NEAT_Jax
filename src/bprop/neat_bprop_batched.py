import time
import numpy as np
import jax.numpy as jnp
import jax
from pathlib import Path
import json

from src.jax_neat.convert import genomes_to_params_batch
from src.neat_core.neat import Neat, NeatHyperParams
from src.neat_core.genome import Genome
from src.bprop.viz import BpropGenomeEvolutionRecorder
from src.jax_neat.policy import jax_forward


def load_classification_data(file_path: str = "neat_dataset.json") -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Loads data from the generated JSON file and converts it to JAX arrays.
    
    Returns:
        X: (N, OBS_DIM) JAX array of inputs.
        Y: (N,) JAX array of labels (0 or 1).
    """
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        data: list[dict] = json.load(f)

    # Separate inputs and outputs
    inputs: list[list[float]] = [[item['x'], item['y']] for item in data]
    # Assuming single classification output: [0] or [1]
    outputs: list[int] = [int(item['l']) for item in data] 

    # Convert to JAX arrays
    X = jnp.array(inputs, dtype=jnp.float32)
    Y = jnp.array(outputs, dtype=jnp.int32)
    
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, Y
X_DATA, Y_DATA = load_classification_data("data/neat_dataset_train.json")

# Update dimensions based on the dataset
OBS_DIM = X_DATA.shape[1]
ACT_DIM = 1*2  # Since we will have activation mutations, better to classify by argmax rather than thresholding

# 2. Define the VMAP that processes the POPULATION (P) and OBSERVATIONS (N)
# P = Population size, N = Number of data points
# This Vmap runs `jax_forward` over all P genomes and all N observations.

# Final shape: (P, N, ACT_DIM) -> (P, N) after thresholding
@jax.jit(static_argnums=(3))
def classify_batched(
    params_batch: dict,  # dict of arrays with shape (P, ...)
    X: jnp.ndarray,      # (N, OBS_DIM) dataset inputs
    Y: jnp.ndarray,       # (N,) dataset labels
    n_output: int = ACT_DIM
) -> jnp.ndarray:
    
    # 2a. VMAP over Observations (N) first
    jax_forward_vmap_obs = jax.vmap(jax_forward, in_axes=(None, 0, None)) # Genome fixed, Obs batched

    # 2b. VMAP over Population (P) second
    # Input X (N, D) is shared across the population, so in_axes=(0, None)
    jax_forward_vmap_pop_obs = jax.vmap(jax_forward_vmap_obs, in_axes=(0, None, None))

    # raw_outputs_batch shape: (P, N, ACT_DIM)
    raw_outputs_batch = jax_forward_vmap_pop_obs(params_batch, X, n_output)

    # prediction_batch shape: (P, N)
    predictions_batch: jnp.ndarray = jnp.argmax(raw_outputs_batch, axis=2).astype(jnp.int32)

    # 5. Calculate accuracy (P,)
    correct_predictions = (predictions_batch == jnp.expand_dims(Y, 0))
    accuracies: jnp.ndarray = jnp.mean(correct_predictions.astype(jnp.float32), axis=1)

    return accuracies

def evaluate_population_classification(
    genomes: list,
    pop_size: int | None = None,
) -> np.ndarray:
    """
    Evaluate a list of genomes using the classification dataset.

    Returns:
        fitnesses: np.ndarray of shape (len(genomes),) containing accuracies.
    """
    # Convert list of Genome objects to a JAX-friendly batch of parameters
    params_batch = genomes_to_params_batch(genomes, OBS_DIM, ACT_DIM)
    fitnesses_jax = classify_batched(params_batch=params_batch, X=X_DATA, Y=Y_DATA)
    return np.array(fitnesses_jax, dtype=np.float32)

def train_neat_on_classification(generations: int = 200, pop_size: int = 200):
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
    
    # Use a simpler directory name
    log_dir = Path('./log/classification')
    rec = BpropGenomeEvolutionRecorder(log_dir) 
    # Global variables for the dataset
    X_DATA_TEST, Y_DATA_TEST = load_classification_data("data/neat_dataset_test.json")
    for gen in range(generations):
        genomes: list[Genome] = neat.ask()
        fitnesses = evaluate_population_classification(genomes, pop_size=pop_size)
        neat.tell(fitnesses)

        best = neat.get_best()
        print(
            f"Gen {gen:03d}  best_fit={best.fitness:.3f} (Accuracy)  "
            f"nodes={len(best.genome.nodes)}  conns={len(best.genome.connections)}"
        )
        
        # Save a visualization of the best network structure periodically
        if gen % 5 == 0 or gen == generations - 1:
            rec.save_combined_frame(best, label=f"gen {gen}", X_data=X_DATA_TEST, Y_data=Y_DATA_TEST, n_input=OBS_DIM, n_output=ACT_DIM)

    # Remove GIF/rendering logic, as it's SlimeVolley-specific.
    final_gif_path = log_dir / "neat_evolution.gif"
    rec.make_gif(final_gif_path, duration_ms=500)
    print(f"Final best genome structure saved.")


if __name__ == "__main__":
    import time
    start = time.time()
    train_neat_on_classification(generations=100, pop_size=100)
    print(f"Total training time: {time.time() - start} seconds")
