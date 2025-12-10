import os
from pathlib import Path
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.neat_core.genome import Genome, NodeType

def draw_genome(genome: Genome, ax: plt.Axes = None) -> plt.Axes:
    """
    Visualize a NEAT Genome, drawing ONLY nodes connected to at least one edge.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    G = nx.DiGraph()

    # Add nodes with type info
    for nid, node in genome.nodes.items():
        G.add_node(nid, node_type=node.type, activation=node.activation)

    # Add edges with weight + enabled info
    for c in genome.connections:
        G.add_edge(
            c.in_id,
            c.out_id,
            weight=c.weight,
            enabled=c.enabled,
        )

    # --- Identify nodes that are part of any edge ---
    connected_nodes = set()
    for u, v in G.edges():
        connected_nodes.add(u)
        connected_nodes.add(v)
        
    # Filter genome.nodes to include only connected nodes for layout/drawing
    connected_genome_nodes = {
        nid: node for nid, node in genome.nodes.items() if nid in connected_nodes
    }

    # ---- Layout: place connected nodes by type in vertical columns ----
    type_to_x = {
        NodeType.INPUT: 0.0,
        NodeType.BIAS: 0.5,
        NodeType.HIDDEN: 1.5,
        NodeType.OUTPUT: 2.5,
    }

    # Group *connected* nodes by type
    # FIX: Use NodeType objects (keys of type_to_x) for groups initialization
    groups: dict[Any, List[int]] = {t: [] for t in type_to_x.keys()}
    for nid, node in connected_genome_nodes.items():
        groups[node.type].append(nid)

    # Assign positions
    pos: Dict[int, Tuple[float, float]] = {}
    for ntype, ids in groups.items():
        if not ids:
            continue
        ids = sorted(ids)
        # Spread them vertically
        # Note: np.linspace is fine, but need to ensure it's imported
        ys = np.linspace(0.0, 1.0, len(ids) + 2)[1:-1] 
        x = type_to_x[ntype]
        for nid, y in zip(ids, ys):
            pos[nid] = (x, y)

    # ---- Draw connected nodes (by type so we can change shapes) ----
    node_colors = {
        NodeType.INPUT: "#8dd3c7",   # mint-ish
        NodeType.HIDDEN: "#ffffb3",  # pale yellow
        NodeType.OUTPUT: "#bebada",  # lavender
        NodeType.BIAS: "#fdb462",    # orange
    }

    # Circle for input/hidden/output, square for bias
    for ntype, shape in [
        (NodeType.INPUT, "o"),
        (NodeType.HIDDEN, "o"),
        (NodeType.OUTPUT, "o"),
        (NodeType.BIAS, "s"),
    ]:
        # Filter nodelist to only include connected nodes of the current type
        nodelist = [
            nid for nid, node in connected_genome_nodes.items() if node.type == ntype
        ]
        if not nodelist:
            continue
        # Note: nx.draw_networkx_nodes is fine, but need to ensure nx is imported
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_shape=shape,
            node_color=node_colors[ntype],
            edgecolors="black",
            linewidths=1.0,
            ax=ax,
        )

    # Node labels: just the node ID (only for connected nodes)
    labels = {nid: str(nid) for nid in connected_genome_nodes.keys()}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8,
        ax=ax,
    )

    # Activation labels near hidden & output nodes (only for connected nodes)
    for nid, node in connected_genome_nodes.items():
        if node.type in (NodeType.HIDDEN, NodeType.OUTPUT):
            x, y = pos[nid]
            # Assuming activation is an attribute on the node, not node.activation.value
            act_label = getattr(node.activation, 'value', node.activation) 
            ax.text(
                x + 0.08,
                y + 0.03,
                act_label,
                fontsize=7,
                ha="left",
                va="center",
            )

    # ---- Draw edges, encoding sign + magnitude (no change needed here) ----
    enabled_edges = []
    disabled_edges = []
    for u, v, data in G.edges(data=True):
        if data.get("enabled", True):
            enabled_edges.append((u, v, data["weight"]))
        else:
            disabled_edges.append((u, v))

    # Enabled edges: color by sign, width by |weight|
    if enabled_edges:
        weights = np.array([abs(w) for _, _, w in enabled_edges])
        max_w = weights.max() if len(weights) > 0 else 1.0
        # Scale to a reasonable linewidth range
        widths = 0.5 + 2.5 * (weights / max_w)

        colors = []
        for _, _, w in enabled_edges:
            if w > 0:
                colors.append("tab:blue")
            elif w < 0:
                colors.append("tab:red")
            else:
                colors.append("gray")

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for u, v, _ in enabled_edges],
            width=widths,
            edge_color=colors,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=10,
            ax=ax,
        )

    # Disabled edges: faint dashed
    if disabled_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=disabled_edges,
            style="dashed",
            alpha=0.3,
            edge_color="gray",
            arrows=False,
            ax=ax,
        )

    ax.set_axis_off()
    ax.set_title("NEAT Genome (Connected Nodes Only)", fontsize=12)
    plt.tight_layout()
    return ax

class GenomeEvolutionRecorder:
    """
    Helper to periodically save genome graphs and later create a GIF.

    Usage:
        rec = GenomeEvolutionRecorder("viz_runs/run1")

        for gen in range(num_generations):
            ... evolve population ...
            best = pick_best_genome(...)
            rec.save_genome_frame(best, label=f"gen {gen}")

        rec.make_gif("evolution.gif", fps=2)
    """

    def __init__(self, out_dir: str | os.PathLike, prefix: str = "frame"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.frame_idx = 0
        self.frames: list[Path] = []

    def save_genome_frame(self, genome: Genome, label: str | None = None) -> Path:
        """Render and save a single frame for this genome.

        Returns path to the saved PNG.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_genome(genome, ax=ax)

        if label is not None:
            # Append extra info (e.g., generation, fitness)
            ax.set_title(f"{ax.get_title()}  |  {label}", fontsize=10)

        fname = f"{self.prefix}_{self.frame_idx:05d}.png"
        fpath = self.out_dir / fname

        fig.savefig(fpath, dpi=120)
        plt.close(fig)

        self.frames.append(fpath)
        self.frame_idx += 1
        return fpath

    def make_gif(self, gif_path: str | os.PathLike, duration_ms: int = 1000):
        """Combine all saved frames into a GIF."""
        gif_path = Path(gif_path)
        # Sort frames by name just in case
        frame_files = sorted(self.frames, key=lambda p: p.name)

        if not frame_files:
            raise RuntimeError("No frames to make GIF from. Did you call save_genome_frame()?")

        images = [imageio.imread(str(p)) for p in frame_files]
        # duration = 1.0 / fps  # seconds per frame

        # imageio.mimsave(str(gif_path), images, duration=duration)
        # images = [imageio.imread(str(p)) for p in frame_files]
        # durations = [1/fps] * len(images)

        imageio.mimsave(str(gif_path), images, duration=duration_ms)
        print(f"GIF saved to: {gif_path}")