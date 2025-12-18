import os
from pathlib import Path
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import networkx as nx
import numpy as np

from src.neat_core.genome import Genome, NodeType

def _graphviz_pos(G: nx.DiGraph) -> dict[int, np.ndarray] | None:
    """Graphviz dot layout if available; returns None if not."""
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        p = graphviz_layout(G, prog="dot")
        return {n: np.array([float(x), float(y)], dtype=float) for n, (x, y) in p.items()}
    except Exception:
        return None


def _normalize_pos(pos: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Normalize to roughly [-1,1] range for consistent sizing."""
    if not pos:
        return pos
    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    dx = max(x1 - x0, 1e-9)
    dy = max(y1 - y0, 1e-9)

    out = {}
    for n, p in pos.items():
        out[n] = np.array([(p[0] - x0) / dx * 2 - 1, (p[1] - y0) / dy * 2 - 1], dtype=float)
    return out


def _spread_vertical(ids: list[int], yspan: tuple[float, float]) -> dict[int, float]:
    if not ids:
        return {}
    y0, y1 = yspan
    ys = np.linspace(y0, y1, len(ids))
    return {nid: float(y) for nid, y in zip(ids, ys)}


def _resolve_overlaps_hard(
    pos: dict[int, np.ndarray],
    fixed: set[int],
    min_dist: float = 0.24,
    steps: int = 900,
    seed: int = 0,
    xlim: tuple[float, float] = (-0.88, 0.88),
    ylim: tuple[float, float] = (-1.25, 1.25),
) -> dict[int, np.ndarray]:
    """Strong repulsion + per-iteration clamping for non-fixed nodes."""
    rng = np.random.default_rng(seed)
    nodes = list(pos.keys())
    P = {n: pos[n].astype(float).copy() for n in nodes}

    x0, x1 = xlim
    y0, y1 = ylim

    for _ in range(steps):
        moved = 0
        for i in range(len(nodes)):
            ni = nodes[i]
            for j in range(i + 1, len(nodes)):
                nj = nodes[j]
                delta = P[ni] - P[nj]
                dist = float(np.linalg.norm(delta)) + 1e-9
                if dist < min_dist:
                    if dist < 1e-6:
                        delta = rng.uniform(-1, 1, size=2)
                        dist = float(np.linalg.norm(delta)) + 1e-9
                    push = (min_dist - dist) * (delta / dist) * 0.55
                    if ni not in fixed:
                        P[ni] += push
                        moved += 1
                    if nj not in fixed:
                        P[nj] -= push
                        moved += 1

        for n in nodes:
            if n in fixed:
                continue
            P[n][0] = float(np.clip(P[n][0], x0, x1))
            P[n][1] = float(np.clip(P[n][1], y0, y1))

        if moved == 0:
            break

    return P


def _set_axes_from_pos(ax: plt.Axes, pos: dict[int, np.ndarray], pad: float = 0.25) -> None:
    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    if len(xs) == 0:
        return
    ax.set_xlim(float(xs.min() - pad), float(xs.max() + pad))
    ax.set_ylim(float(ys.min() - pad), float(ys.max() + pad))


def _select_functional_subgraph(
    G: nx.DiGraph, input_ids: list[int], output_ids: list[int]
) -> nx.DiGraph:
    """
    Keep only nodes that are:
    - reachable from any input/bias AND
    - can reach any output
    This removes isolated islands and "hanging" components that cannot affect outputs.
    """
    if not input_ids or not output_ids:
        return G

    reachable_from_inputs: set[int] = set()
    for s in input_ids:
        if s in G:
            reachable_from_inputs |= nx.descendants(G, s) | {s}

    can_reach_outputs: set[int] = set()
    Gr = G.reverse(copy=False)
    for t in output_ids:
        if t in Gr:
            can_reach_outputs |= nx.descendants(Gr, t) | {t}

    keep = reachable_from_inputs & can_reach_outputs
    if not keep:
        # Fallback: largest weakly connected component (better than nothing)
        comps = list(nx.weakly_connected_components(G))
        if not comps:
            return G
        keep = set(max(comps, key=len))
    return G.subgraph(keep).copy()


def draw_genome(genome: "Genome", ax: plt.Axes = None) -> plt.Axes:
    """
    Overlap-minimized NEAT genome visualization that is robust to negative weights and removes hanging nodes:
    - Filter to functional subgraph (input/bias -> ... -> output).
    - Prefer Graphviz dot layout; fallback to unweighted Kamada-Kawai (no negative-weight Dijkstra).
    - Strict fixed columns for INPUT/BIAS and OUTPUT.
    - Strong overlap removal for non-fixed nodes + clamping + auto-fit axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    # Build graph (store weight only for drawing, NEVER for layout shortest paths)
    G_full = nx.DiGraph()
    for nid, node in genome.nodes.items():
        G_full.add_node(nid, node_type=node.type, activation=node.activation)
    for c in genome.connections:
        if c.enabled:
            G_full.add_edge(c.in_id, c.out_id, weight=float(c.weight))

    if G_full.number_of_nodes() == 0:
        ax.set_axis_off()
        return ax

    # Partition nodes by type
    connected_genome_nodes = {nid: node for nid, node in genome.nodes.items() if nid in G_full.nodes()}
    groups = {t: [] for t in [NodeType.INPUT, NodeType.BIAS, NodeType.HIDDEN, NodeType.OUTPUT]}
    for nid, node in connected_genome_nodes.items():
        groups[node.type].append(nid)

    left_ids = sorted(groups[NodeType.INPUT] + groups[NodeType.BIAS])
    out_ids = sorted(groups[NodeType.OUTPUT])

    # Filter to functional subgraph to remove hanging nodes/islands
    G = _select_functional_subgraph(G_full, left_ids, out_ids)
    connected_genome_nodes = {nid: node for nid, node in genome.nodes.items() if nid in G.nodes()}

    # Recompute groups after filtering
    groups = {t: [] for t in [NodeType.INPUT, NodeType.BIAS, NodeType.HIDDEN, NodeType.OUTPUT]}
    for nid, node in connected_genome_nodes.items():
        groups[node.type].append(nid)

    left_ids = sorted(groups[NodeType.INPUT] + groups[NodeType.BIAS])
    out_ids = sorted(groups[NodeType.OUTPUT])
    hid_ids = list(groups[NodeType.HIDDEN])

    # 1) Base layout
    pos = _graphviz_pos(G)
    if pos is None:
        # IMPORTANT: unweighted distances to avoid negative-weight Dijkstra failures.
        # Use an undirected copy for a more stable geometric embedding.
        pos = nx.kamada_kawai_layout(G.to_undirected(), weight=None)
        pos = {n: pos[n].astype(float) for n in pos}

    pos = _normalize_pos(pos)

    # 2) Strict anchors
    anchors: dict[int, np.ndarray] = {}
    fixed_nodes: list[int] = []

    # If you have many IO nodes, widen yspan to prevent vertical collisions
    yspan = (-1.45, 1.45)

    left_y = _spread_vertical(left_ids, yspan)
    for nid in left_ids:
        p = np.array([-1.0, left_y[nid]], dtype=float)
        pos[nid] = p.copy()
        anchors[nid] = p.copy()
        fixed_nodes.append(nid)

    out_y = _spread_vertical(out_ids, yspan)
    for nid in out_ids:
        p = np.array([1.0, out_y[nid]], dtype=float)
        pos[nid] = p.copy()
        anchors[nid] = p.copy()
        fixed_nodes.append(nid)

    fixed_set = set(fixed_nodes)

    # 3) Clamp hidden to middle region before de-overlap
    for nid in hid_ids:
        if nid in pos:
            pos[nid][0] = float(np.clip(pos[nid][0], -0.88, 0.88))
            pos[nid][1] = float(np.clip(pos[nid][1], -1.25, 1.25))

    # 4) Hard overlap removal
    pos = _resolve_overlaps_hard(
        pos,
        fixed=fixed_set,
        min_dist=0.24,   # increase if you keep node_size=800 and many nodes
        steps=900,
        seed=0,
        xlim=(-0.88, 0.88),
        ylim=(-1.25, 1.25),
    )

    # Restore anchors (strict)
    for nid, p in anchors.items():
        pos[nid] = p.copy()

    # ---- DRAWING ----
    node_colors = {
        NodeType.INPUT: "#8dd3c7",
        NodeType.HIDDEN: "#ffffb3",
        NodeType.OUTPUT: "#bebada",
        NodeType.BIAS: "#fdb462",
    }

    for ntype, shape in [
        (NodeType.INPUT, "o"),
        (NodeType.HIDDEN, "o"),
        (NodeType.OUTPUT, "o"),
        (NodeType.BIAS, "s"),
    ]:
        nodelist = [nid for nid, node in connected_genome_nodes.items() if node.type == ntype]
        if nodelist:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodelist,
                node_shape=shape,
                node_color=node_colors[ntype],
                edgecolors="black",
                linewidths=1.2,
                ax=ax,
                node_size=800,
            )

    for nid, node in connected_genome_nodes.items():
        x, y = pos[nid]
        if node.type in (NodeType.HIDDEN, NodeType.OUTPUT):
            act_label = getattr(node.activation, "value", str(node.activation))
            display_text = act_label[:4] if len(act_label) > 5 else act_label
            ax.text(x, y, display_text, fontsize=6, fontweight="bold", ha="center", va="center")
        elif node.type == NodeType.BIAS:
            ax.text(x, y, "B", fontsize=8, fontweight="bold", ha="center", va="center")

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        color = "tab:blue" if w > 0 else "tab:red"
        width = 0.8 + 2.5 * min(abs(w), 4.0)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=width,
            edge_color=color,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
            connectionstyle="arc3,rad=0.06",
            ax=ax,
            alpha=0.85,
        )

    _set_axes_from_pos(ax, pos, pad=0.30)
    ax.set_axis_off()
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