import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def load_patch_results(path):
    """Load JSON patching results from disk."""
    with open(path, 'r') as f:
        return json.load(f)


def build_circuit_graph(results, threshold=0.0):
    """Construct a directed graph of attention heads and MLP layers filtering by threshold."""
    G = nx.DiGraph()
    z_scores = results.get('z', [])
    mlp_scores = results.get('mlp_out', [])

    # Add attention head nodes
    for layer_idx, layer_heads in enumerate(z_scores):
        # if list is a list of lists, take max for each head
        if isinstance(layer_heads, list) and all(isinstance(h, list) for h in layer_heads):
            # trasnpose to get heads as outer list
            num_tokens = len(layer_heads)
            num_heads = len(layer_heads[0])
            layer_heads = [[layer_heads[j][i] for i in range(len(num_heads))] for j in range(len(num_tokens))]
            layer_heads = [max(head_scores) for head_scores in layer_heads]
            
        for head_idx, score in enumerate(layer_heads):
            # Filter by threshold
            if score >= threshold:
                name = f"Attn_L{layer_idx}H{head_idx}"
                G.add_node(name, score=score, type='attn')

    # Add MLP nodes
    for layer_idx, score_data in enumerate(mlp_scores):
        if isinstance(score_data, list):
            score_val = max(score_data) if score_data else 0.0
        else:
            score_val = score_data
        if score_val >= threshold:
            name = f"MLP_L{layer_idx}"
            G.add_node(name, score=score_val, type='mlp')

    # Add edges: Attn->MLP within same layer, MLP->Attn next layer
    n_layers = max(len(z_scores), len(mlp_scores))
    for L in range(n_layers):
        mlp_node = f"MLP_L{L}"
        if mlp_node in G:
            # Attn heads to MLP
            for head_idx in range(len(z_scores[L]) if L < len(z_scores) else 0):
                attn_node = f"Attn_L{L}H{head_idx}"
                if attn_node in G:
                    G.add_edge(attn_node, mlp_node)
            # MLP to next layer's heads
            nxt = L + 1
            if nxt < len(z_scores):
                for head_idx in range(len(z_scores[nxt])):
                    nxt_attn = f"Attn_L{nxt}H{head_idx}"
                    if nxt_attn in G:
                        G.add_edge(mlp_node, nxt_attn)
    return G


def draw_graph(G, threshold, ax):
    """Draw the circuit graph using a layered layout."""
    # Determine positions: layers on y-axis, heads spread on x-axis
    pos = {}
    layer_map = {}
    for node, data in G.nodes(data=True):
        typ = data['type']
        parts = node.split('_L')
        if typ == 'attn':
            _, rest = parts
            layer_str, head_str = rest.split('H')
            layer = int(layer_str)
            idx = int(head_str)
        else:
            _, layer_str = parts
            layer = int(layer_str)
            idx = -1
        layer_map.setdefault(layer, []).append((node, idx, typ))
    # assign x,y
    for layer, nodes in layer_map.items():
        nodes.sort(key=lambda x: x[1])
        n = len(nodes)
        for i, (node, idx, typ) in enumerate(nodes):
            x = i - n/2
            y = -layer
            pos[node] = (x, y)

    ax.clear()
    attn_nodes = [n for n, d in G.nodes(data=True) if d['type']=='attn']
    mlp_nodes = [n for n, d in G.nodes(data=True) if d['type']=='mlp']
    nx.draw_networkx_nodes(G, pos, nodelist=attn_nodes, node_color='skyblue', label='Attention', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=mlp_nodes, node_color='orange', label='MLP', ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    ax.set_title(f"Circuit Graph (threshold={threshold:.2f})")
    ax.legend(loc='upper right')
    ax.axis('off')


def visualize_interactive(results):
    """Launch an interactive plot with a threshold slider."""
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)
    init_thresh = 0.0
    G_init = build_circuit_graph(results, init_thresh)
    draw_graph(G_init, init_thresh, ax)

    ax_slider = plt.axes([0.1, 0.05, 0.8, 0.05])
    slider = Slider(ax_slider, 'Threshold', -1.0, 1.0, valinit=init_thresh, valstep=0.01)

    def update(val):
        thresh = slider.val
        G = build_circuit_graph(results, thresh)
        draw_graph(G, thresh, ax)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def visualize_static(results, threshold):
    """Draw a static graph at a fixed threshold."""
    G = build_circuit_graph(results, threshold)
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_graph(G, threshold, ax)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize activation patching circuit graph'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Path to activation patching JSON file')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help='Threshold for filtering nodes (if omitted, launch interactive view)')
    args = parser.parse_args()

    results = load_patch_results(args.input)
    if args.threshold is None:
        visualize_interactive(results)
    else:
        visualize_static(results, args.threshold)

if __name__ == '__main__':
    main()
