import jax
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from src.jax_neat.convert import genome_to_jax
from src.neat_core.genome import Genome
from src.neat_core.neat import Individual
from src.jax_neat.policy import jax_forward
from src.neat_core.visualize import GenomeEvolutionRecorder, draw_genome

@jax.jit(static_argnums=(2))
def generate_prediction_grid(params: dict, X_grid: jnp.ndarray, n_output: int) -> jnp.ndarray:
    """
    Generates predictions for an entire grid using a single genome.

    params: JAXGenome parameters (dict).
    X_grid: (GridSize, 2) array of (x, y) coordinates.
    n_output: Static integer for the output size (must be 1).

    Returns:
        Z_pred: (GridSize,) array of binary predictions (0 or 1).
    """
    # Vmap jax_forward over the grid observations
    # jax_forward_grid(genome, X_grid, n_output) -> (GridSize, n_output)
    jax_forward_vmap_grid = jax.vmap(jax_forward, in_axes=(None, 0, None))
    
    # Raw output shape: (GridSize, 1)
    raw_outputs = jax_forward_vmap_grid(params, X_grid, n_output)

    # Apply threshold and squeeze to (GridSize,)
    return jnp.argmax(raw_outputs, axis=-1).astype(jnp.int32)


def visualize_decision_boundary(genome: Genome, fitness:float, X_data: jnp.ndarray, Y_data: jnp.ndarray, n_input: int, n_output: int) -> Image.Image:
    """
    Plots the dataset and the decision boundary for the given genome.
    Returns a PIL Image object.
    """
    # 1. Setup Grid for Prediction
    x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
    y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
    h = 0.05  # step size in the mesh
    xx, yy = jnp.meshgrid(jnp.arange(x_min, x_max, h), jnp.arange(y_min, y_max, h))
    
    # Grid data for JAX
    X_grid = jnp.c_[xx.ravel(), yy.ravel()]
    
    # 2. Get JAX Parameters and Predictions
    params = genome_to_jax(genome, obs_dim=n_input, act_dim=n_output) # Use n_output here too
    Z = generate_prediction_grid(params, X_grid, n_output)
    
    # Put result back into a grid structure for plotting
    Z = Z.reshape(xx.shape)
    
    # 3. Plotting with Matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot decision boundaries (shaded area)
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    
    # Plot training data points (scatter)
    # Convert JAX arrays to NumPy for plotting
    ax.scatter(np.array(X_data[:, 0]), np.array(X_data[:, 1]), 
               c=np.array(Y_data), cmap=plt.cm.RdBu, edgecolors='k')
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(f"Decision Boundary (Accuracy: {fitness:.3f})")
    
    # 4. Save to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img


class BpropGenomeEvolutionRecorder(GenomeEvolutionRecorder):
    """Extends GenomeEvolutionRecorder to add JAX visualization capabilities."""

    def save_network_structure(self, genome: Genome, fitness:float, label: str | None = None) -> Image.Image:
            """
            Renders the network structure and returns a PIL Image object.
            (This replaces the original save_genome_frame's network drawing part)
            """
            fig, ax = plt.subplots(figsize=(8, 6))
            # Assuming draw_genome is defined and available
            draw_genome(genome, ax=ax) 

            if label is not None:
                # Append extra info (e.g., generation, fitness)
                ax.set_title(f"{ax.get_title()} | {label} | Fitness: {fitness:.3f}", fontsize=10)

            # Save to buffer and return PIL Image
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=120)
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)
    
    def save_combined_frame(self, best: Individual, label: str, X_data: jnp.ndarray, Y_data: jnp.ndarray, n_input: int, n_output: int) -> Path:
            """
            Generates the network structure, the decision boundary visualization, 
            combines them, and saves the final frame.
            """
            genome = best.genome
            fitness = best.fitness
            network_img = self.save_network_structure(genome, fitness, label=label)
            boundary_img = visualize_decision_boundary(genome, fitness, X_data, Y_data, n_input, n_output)
            new_height = 500
            
            # Resize both images to the target height
            network_img = network_img.resize((int(network_img.width * new_height / network_img.height), new_height))
            boundary_img = boundary_img.resize((int(boundary_img.width * new_height / boundary_img.height), new_height))

            # Create the combined canvas
            new_width = network_img.width + boundary_img.width
            combined_img = Image.new('RGB', (new_width, new_height))
            
            # Paste images
            combined_img.paste(network_img, (0, 0))
            combined_img.paste(boundary_img, (network_img.width, 0))
            
            fname = f"{self.prefix}_{self.frame_idx:05d}.png"
            fpath = self.out_dir / fname
            combined_img.save(fpath)

            self.frames.append(fpath)
            self.frame_idx += 1
            return fpath
