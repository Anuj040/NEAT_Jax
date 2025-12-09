import graphviz
import imageio.v3 as iio
from pathlib import Path
import numpy as np
from PIL import Image
from src.neat_core.genome import Genome, NodeType

def render_genome_graph(
    genome: Genome, 
    filename: str = 'genome_graph', 
    directory: str = './log/neat_graphs',
    view: bool = False
) -> str:
    """
    Renders the NEAT genome graph with small circles and external labels,
    emphasizing clarity for small nodes.
    """
    # Use Path objects internally for robust directory handling
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dot = graphviz.Digraph(
        comment='NEAT Genome', 
        graph_attr={
            'rankdir': 'LR', 
            'splines': 'line',
            # Fixed canvas size for reliable GIF creation
            'size': '8,6!', 
            'page': '8,6',
            'margin': '0.5',
            'dpi': '300'
        },
        node_attr={
            # General style for non-bias nodes
            'shape': 'circle', 
            'style': 'filled',
            # Small node size
            'fixedsize': 'true', 
            'width': '0.2',
            'height': '0.2',
            'label': ''        
        }
    )

    # --- 1. Identify and Add Nodes & Set External Labels ---
    input_ids, bias_ids, hidden_ids, output_ids = [], [], [], []
    for nid, node in genome.nodes.items():
        color_map = {
            NodeType.INPUT: 'lightblue',
            NodeType.BIAS: 'gray',
            NodeType.HIDDEN: 'yellow',
            NodeType.OUTPUT: 'lightcoral'
        }
        fillcolor = color_map.get(node.type, 'white')
        
        # 🎯 Improvement 1: Include Node ID in the external label
        if node.type in (NodeType.HIDDEN, NodeType.OUTPUT):
            # Show ID and Activation Type outside the node
            xlabel_text = f"ID:{nid}\n{node.activation.name}"
        else:
            # Show ID and Type name (input/bias) outside the node
            xlabel_text = f"ID:{nid}\n{node.type.name.lower()}"

        node_shape = 'circle'
        if node.type == NodeType.BIAS:
            node_shape = 'box'
            
        # Add the node
        dot.node(
            name=str(nid), 
            fillcolor=fillcolor,
            xlabel=xlabel_text,
            fontsize='8', # Smaller font size for the external label
            shape=node_shape
        )
        
        # Group IDs for layering
        if node.type == NodeType.INPUT:
            input_ids.append(str(nid))
        elif node.type == NodeType.BIAS:
            bias_ids.append(str(nid))
        elif node.type == NodeType.HIDDEN:
            hidden_ids.append(str(nid))
        elif node.type == NodeType.OUTPUT:
            output_ids.append(str(nid))
            
    # --- 2. Enforce Layering (Ranks) ---

    # 1. Inputs (First Layer)
    with dot.subgraph(name='cluster_input') as sub:
        sub.attr(rank='same', label='Input Layer', color='transparent')
        for nid in input_ids:
            sub.node(nid)
    
    # 2. Bias (Grouped visually with Inputs)
    with dot.subgraph(name='cluster_bias') as sub:
        sub.attr(rank='same', label='Bias', color='transparent')
        for nid in bias_ids:
            sub.node(nid)
            
    # 3. Hidden Nodes (Middle Layer)
    if hidden_ids:
        with dot.subgraph(name='cluster_hidden') as sub:
            sub.attr(rank='same', label='Hidden Layer', color='transparent')
            for nid in hidden_ids:
                sub.node(nid)

    # 4. Outputs (Last Layer)
    with dot.subgraph(name='cluster_output') as sub:
        sub.attr(rank='same', label='Output Layer', color='transparent')
        for nid in output_ids:
            sub.node(nid)

    # --- 3. Add Connections (Weight labels removed, thickness/color used) ---
    for conn in genome.connections:
        if not conn.enabled:
            style = 'dashed'
            color = 'lightgray'
            thickness = '0.5'
        else:
            style = 'solid'
            if conn.weight >= 0:
                color = 'green'
            else:
                color = 'red'
            # Line thickness based on weight magnitude
            thickness = str(0.5 + abs(conn.weight) * 1.5) 

        dot.edge(
            str(conn.in_id), 
            str(conn.out_id), 
            color=color,
            style=style,
            penwidth=thickness
        )

    # --- 4. Render and Save ---
    try:
        output_path = dot.render(filename=filename, directory=directory, format='png', view=view, cleanup=True)
        return output_path
    except graphviz.backend.ExecutableNotFound:
        print("\n--- ERROR ---")
        print("Graphviz executable not found. Please install the system package.")
        return ""
    
# def render_genome_graph(
#     genome: Genome, 
#     filename: str = 'genome_graph', 
#     directory: str = './log/neat_graphs',
#     view: bool = False
# ) -> str:
#     """
#     Renders the NEAT genome graph with small circles and external labels.
#     """
#     # Use Path objects internally for robust directory handling
#     output_dir = Path(directory)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     dot = graphviz.Digraph(
#         comment='NEAT Genome', 
#         graph_attr={
#             # General layout
#             'rankdir': 'LR', 
#             'splines': 'line',
#             # FIX: Consistent canvas size (critical for GIF creation)
#             'size': '8,6!', 
#             'page': '8,6',
#             'margin': '0.5',
#             'dpi': '300'
#         },
#         node_attr={
#             'shape': 'circle', 
#             'style': 'filled',
#             # 🎯 FIX 1: Smaller node size
#             'fixedsize': 'true', 
#             'width': '0.35',   # Significantly reduced width
#             'height': '0.35',  # Significantly reduced height
#             # 🎯 FIX 2: Set the internal label to be empty
#             'label': ''        
#         }
#     )

#     # --- 1. Identify and Add Nodes & Set External Labels ---
#     input_ids, bias_ids, hidden_ids, output_ids = [], [], [], []
#     for nid, node in genome.nodes.items():
#         color_map = {
#             NodeType.INPUT: 'lightblue',
#             NodeType.BIAS: 'gray',
#             NodeType.HIDDEN: 'yellow',
#             NodeType.OUTPUT: 'lightcoral'
#         }
#         fillcolor = color_map.get(node.type, 'white')
        
#         # 🎯 FIX 3: Create the external label string
#         if node.type in (NodeType.HIDDEN, NodeType.OUTPUT):
#             # Show ID and Activation Type outside the node
#             xlabel_text = f"{node.activation.name}"
#         else:
#             # Show ID and Type name outside the node
#             xlabel_text = f"{node.type.name.lower()}"

#         # Add the node, using the 'xlabel' attribute for the external label
#         dot.node(
#             name=str(nid), 
#             fillcolor=fillcolor,
#             xlabel=xlabel_text,
#             fontsize='8' # Smaller font size for the external label
#         )
        
#         # Group IDs for layering
#         if node.type == NodeType.INPUT:
#             input_ids.append(str(nid))
#         elif node.type == NodeType.BIAS:
#             bias_ids.append(str(nid))
#         elif node.type == NodeType.HIDDEN:
#             hidden_ids.append(str(nid))
#         elif node.type == NodeType.OUTPUT:
#             output_ids.append(str(nid))
            
#     # --- 2. Enforce Layering (Ranks) ---
#     # ... (Layering logic remains the same as the previous correction) ...

#     # 1. Inputs (First Layer)
#     with dot.subgraph(name='cluster_input') as sub:
#         sub.attr(rank='same', label='Input Layer', color='transparent')
#         for nid in input_ids:
#             sub.node(nid)
    
#     # 2. Bias (Grouped visually with Inputs)
#     with dot.subgraph(name='cluster_bias') as sub:
#         sub.attr(rank='same', label='Bias', color='transparent')
#         for nid in bias_ids:
#             sub.node(nid)
            
#     # 3. Hidden Nodes (Middle Layer)
#     if hidden_ids:
#         with dot.subgraph(name='cluster_hidden') as sub:
#             sub.attr(rank='same', label='Hidden Layer', color='transparent')
#             for nid in hidden_ids:
#                 sub.node(nid)

#     # 4. Outputs (Last Layer)
#     with dot.subgraph(name='cluster_output') as sub:
#         sub.attr(rank='same', label='Output Layer', color='transparent')
#         for nid in output_ids:
#             sub.node(nid)

#     # --- 3. Add Connections (Weight labels removed, thickness/color used) ---
#     for conn in genome.connections:
#         if not conn.enabled:
#             style = 'dashed'
#             color = 'lightgray'
#             thickness = '0.5'
#         else:
#             style = 'solid'
#             if conn.weight >= 0:
#                 color = 'green'
#             else:
#                 color = 'red'
#             # Line thickness based on weight magnitude
#             thickness = str(0.5 + abs(conn.weight) * 1.5) 

#         dot.edge(
#             str(conn.in_id), 
#             str(conn.out_id), 
#             color=color,
#             style=style,
#             penwidth=thickness
#         )

#     # --- 4. Render and Save ---
#     try:
#         output_path = dot.render(filename=filename, directory=directory, format='png', view=view, cleanup=True)
#         return output_path
#     except graphviz.backend.ExecutableNotFound:
#         print("\n--- ERROR ---")
#         print("Graphviz executable not found. Please install the system package.")
#         return ""
      
# def create_evolution_gif(
#     image_paths: list[str], 
#     output_gif_path: str, 
#     duration_ms: int = 500
# ) -> None:
#     """
#     Combines a list of image file paths into a single animated GIF.
    
#     Args:
#         image_paths: List of file paths (e.g., PNGs) to include in the GIF, ordered by time.
#         output_gif_path: The full path and filename for the resulting GIF.
#         duration_ms: How long each frame should display (in milliseconds).
#     """
#     if not image_paths:
#         print("No images found to create GIF.")
#         return
        
#     print(f"Creating GIF from {len(image_paths)} images...")
    
#     # Read all images into a list
#     images = [iio.imread(path) for path in image_paths]
    
#     # Write the list of images as an animated GIF
#     iio.imwrite(
#         output_gif_path, 
#         images, 
#         duration=duration_ms / 1000.0, # imageio expects duration in seconds
#         loop=0 # 0 means loop forever
#     )
#     print(f"Successfully created evolution GIF: {output_gif_path}")

def create_evolution_gif(
    image_paths: list[str], 
    output_gif_path: Path, 
    duration_ms: int = 500
) -> None:
    """
    Combines a list of image file paths into a single animated GIF, 
    enforcing consistent shape via Pillow padding to fix the ValueError.
    """
    if not image_paths:
        print("No images found to create GIF.")
        return
        
    print(f"Creating GIF from {len(image_paths)} images...")
    
    max_width, max_height = 0, 0
    images_pil = []

    # --- Step 1: Find Max Dimensions and Load Images ---
    for path in image_paths:
        try:
            # Open the image using Pillow
            img = Image.open(path)
            images_pil.append(img)
            
            # Track the maximum width and height found so far
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
        except FileNotFoundError:
            print(f"Warning: Image file not found at {path}. Skipping.")
        except Exception as e:
            print(f"Error loading image {path}: {e}. Skipping.")

    if not images_pil:
        print("No valid images were loaded. Aborting GIF creation.")
        return

    # --- Step 2: Pad All Images to Max Dimensions ---
    processed_images = []
    
    # We use white (255, 255, 255, 255) as the background color for padding.
    # Note: PNGs often have 4 channels (RGBA).
    for img in images_pil:
        # Create a new blank canvas with the max dimensions and 4 channels (RGBA)
        new_img = Image.new('RGBA', (max_width, max_height), color='white')
        
        # Calculate offset to center the original image on the new canvas
        x_offset = (max_width - img.width) // 2
        y_offset = (max_height - img.height) // 2
        
        # Paste the original image onto the center of the canvas
        new_img.paste(img, (x_offset, y_offset))
        
        # Convert the padded PIL image back to a NumPy array for imageio
        processed_images.append(np.array(new_img))
        
    iio.imwrite(
        output_gif_path, 
        processed_images, # Use the list of shape-consistent NumPy arrays
        duration=duration_ms,
        loop=0
    )
    print(f"Successfully created evolution GIF: {output_gif_path}")