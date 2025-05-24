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
    """Build a token-level directed graph of attention heads and MLP layers.
    Each node is (layer, head/mlp, token position), labeled with clean/corrupt token strings."""
    G = nx.DiGraph()
    z_scores = results.get('z', [])
    mlp_scores = results.get('mlp_out', [])
    
    tokens = results.get('tokens', {})
    clean_tokens = tokens.get('clean', [])
    corrupt_tokens = tokens.get('corrupt', [])

    # Attention head nodes: (layer, head, pos)
    for layer_idx, layer_heads in enumerate(z_scores):
        for head_idx, head_scores in enumerate(layer_heads):
            for pos, score in enumerate(head_scores):
                if score >= threshold:
                    clean_tok = clean_tokens[pos] if pos < len(clean_tokens) else ""
                    corrupt_tok = corrupt_tokens[pos] if pos < len(corrupt_tokens) else ""
                    name = f"Attn_L{layer_idx}H{head_idx}_P{pos}"
                    label = f"L{layer_idx}H{head_idx} P{pos}\nC:{clean_tok}\nX:{corrupt_tok}\n{score:.2f}"
                    G.add_node(
                        name,
                        score=score,
                        type='attn',
                        layer=layer_idx,
                        head=head_idx,
                        pos=pos,
                        label=label
                    )

    # MLP nodes: (layer, pos)
    for layer_idx, layer_scores in enumerate(mlp_scores):
        for pos, score in enumerate(layer_scores):
            if score >= threshold:
                clean_tok = clean_tokens[pos] if pos < len(clean_tokens) else ""
                corrupt_tok = corrupt_tokens[pos] if pos < len(corrupt_tokens) else ""
                name = f"MLP_L{layer_idx}_P{pos}"
                label = f"MLP L{layer_idx} P{pos}\nC:{clean_tok}\nX:{corrupt_tok}\n{score:.2f}"
                G.add_node(
                    name,
                    score=score,
                    type='mlp',
                    layer=layer_idx,
                    pos=pos,
                    label=label
                )

    # Edges: Attn->MLP (same layer, same pos), MLP->Attn (next layer, same pos)
    n_layers = max(len(z_scores), len(mlp_scores))
    n_pos = len(clean_tokens)
    for L in range(n_layers):
        for pos in range(n_pos):
            mlp_node = f"MLP_L{L}_P{pos}"
            if mlp_node in G:
                # Attn heads to MLP (same layer, same pos)
                for head_idx in range(len(z_scores[L]) if L < len(z_scores) else 0):
                    attn_node = f"Attn_L{L}H{head_idx}_P{pos}"
                    if attn_node in G:
                        G.add_edge(attn_node, mlp_node)
                # MLP to next layer's heads (same pos)
                nxt = L + 1
                if nxt < len(z_scores):
                    for head_idx in range(len(z_scores[nxt])):
                        nxt_attn = f"Attn_L{nxt}H{head_idx}_P{pos}"
                        if nxt_attn in G:
                            G.add_edge(mlp_node, nxt_attn)
    return G


def draw_graph(G, threshold, ax):
    """Draw the token-level circuit graph with token labels."""
    # Arrange nodes by (layer, pos) grid
    pos_dict = {}
    for node, data in G.nodes(data=True):
        layer = data.get('layer', 0)
        pos_idx = data.get('pos', 0)
        # Spread heads horizontally within each (layer, pos)
        if data['type'] == 'attn':
            head = data.get('head', 0)
            x = pos_idx + 0.2 * (head - 2)
        else:
            x = pos_idx
        y = -layer
        pos_dict[node] = (x, y)

    ax.clear()
    attn_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'attn']
    mlp_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'mlp']
    nx.draw_networkx_nodes(G, pos_dict, nodelist=attn_nodes, node_color='skyblue', label='Attention', ax=ax)
    nx.draw_networkx_nodes(G, pos_dict, nodelist=mlp_nodes, node_color='orange', label='MLP', ax=ax)
    nx.draw_networkx_edges(G, pos_dict, arrowstyle='-|>', arrowsize=10, ax=ax)

    # Use precomputed labels
    labels = {n: d['label'] for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos_dict, labels=labels, font_size=6, ax=ax)

    ax.set_title(f"Token-level Circuit Graph (threshold={threshold:.2f})")
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
