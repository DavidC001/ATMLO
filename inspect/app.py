import dash
from dash import html, dcc, Input, Output, State
import dash_cytoscape as cyto
import plotly.express as px
import json
import numpy as np

# Load extra layouts (including dagre)
cyto.load_extra_layouts()

# Load your analysis data
with open('data.json', 'r') as f:
    analysis_data = json.load(f)

app = dash.Dash(__name__)

def build_full_graph(analysis_data):
    """
    Build a full graph across all layers.
    For each layer:
      - Create a tokens/input node (class: tokens).
      - Create attention head nodes (class: attention).
      - Create an MLP node (class: mlp).
      - Connect tokens -> attention heads -> MLP, and link MLP -> next layer's tokens.
    """
    elements = []
    num_layers = len(analysis_data['children'])
    
    for layer_index in range(num_layers):
        layer_info = analysis_data['children'][layer_index]
        
        # Tokens node.
        tokens_id = f"L{layer_index}_tokens"
        tokens_label = "Tokens" if layer_index == 0 else f"Layer {layer_index} Input"
        elements.append({
            'data': {'id': tokens_id, 'label': tokens_label},
            'classes': 'tokens'
        })
        
        # Attention head nodes and edge from tokens.
        for child in layer_info.get('children', []):
            if child['name'].startswith("Attention Head"):
                head_num = child['name'].split()[-1]
                head_id = f"L{layer_index}_head_{head_num}"
                elements.append({
                    'data': {
                        'id': head_id,
                        'label': child['name'],
                        'layer': layer_index,
                        'head': int(head_num)
                    },
                    'classes': 'attention'
                })
                elements.append({'data': {'source': tokens_id, 'target': head_id}})
        
        # MLP node.
        mlp_id = f"L{layer_index}_mlp"
        elements.append({
            'data': {'id': mlp_id, 'label': 'MLP', 'layer': layer_index},
            'classes': 'mlp'
        })
        
        # Edges from attention heads to MLP.
        for child in layer_info.get('children', []):
            if child['name'].startswith("Attention Head"):
                head_num = child['name'].split()[-1]
                head_id = f"L{layer_index}_head_{head_num}"
                elements.append({'data': {'source': head_id, 'target': mlp_id}})
        
        # Connect MLP to next layer's tokens (if any)
        if layer_index < num_layers - 1:
            next_tokens_id = f"L{layer_index+1}_tokens"
            elements.append({'data': {'source': mlp_id, 'target': next_tokens_id}})
            
    return elements

# Build the initial graph elements.
initial_elements = build_full_graph(analysis_data)

# --- Helper function for filtering nodes ---
def filter_elements(elements, remove_ids):
    """
    Remove nodes (and connected edges) whose data.id is in remove_ids.
    Nodes are elements without a "source" field.
    Edges have a "source" field.
    """
    new_elements = []
    for ele in elements:
        data = ele.get('data', {})
        # For nodes: keep if their id is not in remove_ids.
        if 'source' not in data:
            if data.get('id') not in remove_ids:
                new_elements.append(ele)
        else:
            # For edges: only keep if both source and target are not removed.
            if data.get('source') not in remove_ids and data.get('target') not in remove_ids:
                new_elements.append(ele)
    return new_elements

# Define custom Cytoscape stylesheet.
cyto_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'label': 'data(label)',
            'width': '60px',
            'height': '60px',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '10px',
            'border-width': 2,
            'border-color': '#555'
        }
    },
    {
        'selector': '.tokens',
        'style': {
            'shape': 'rectangle',
            'background-color': '#F39C12'
        }
    },
    {
        'selector': '.attention',
        'style': {
            'shape': 'ellipse',
            'background-color': '#3498DB'
        }
    },
    {
        'selector': '.mlp',
        'style': {
            'shape': 'triangle',
            'background-color': '#2ECC71'
        }
    },
    {
        'selector': ':selected',
        'style': {
            'border-color': '#FF4136',
            'border-width': 4,
            'background-color': '#FFDC00'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'line-color': '#A3C4BC',
            'target-arrow-color': '#A3C4BC',
            'width': 2
        }
    }
]

app.layout = html.Div([
    html.H1("LLM Internal Representations Dashboard", style={'textAlign': 'center'}),
    
    # dcc.Store to hold current graph elements.
    dcc.Store(id='graph-store', data=initial_elements),
    
    html.Div([
        # Cytoscape graph: its elements will be updated from the store.
        cyto.Cytoscape(
            id='full-graph',
            layout={'name': 'dagre'},
            stylesheet=cyto_stylesheet,
            style={'width': '100%', 'height': '600px'}
        ),
        # Section for removing unwanted nodes.
        html.Div([
            html.H4("Remove Unwanted Nodes"),
            dcc.Dropdown(
                id='remove-nodes-dropdown',
                multi=True,
                placeholder="Select nodes to remove"
            ),
            html.Button("Remove Selected", id='remove-button', n_clicks=0)
        ], style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px'})
    ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
    
    # Right side: Heatmap and details panel.
    html.Div([
        dcc.Graph(id='heatmap-graph', style={'height': '300px'}),
        html.Div(id='detailed-info', style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px'})
    ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '10px'})
])

# Callback to update the graph elements and removal dropdown options from the store.
@app.callback(
    [Output('full-graph', 'elements'),
     Output('remove-nodes-dropdown', 'options')],
    Input('graph-store', 'data')
)
def update_graph_and_dropdown(store_data):
    # Update Cytoscape graph.
    elements = store_data if store_data is not None else []
    # Create dropdown options only for nodes (elements without "source" in data).
    options = []
    for ele in elements:
        data = ele.get('data', {})
        if 'source' not in data and data.get('id'):
            options.append({'label': data.get('label'), 'value': data.get('id')})
    return elements, options

# Callback to handle node removal.
@app.callback(
    Output('graph-store', 'data'),
    Input('remove-button', 'n_clicks'),
    State('remove-nodes-dropdown', 'value'),
    State('graph-store', 'data')
)
def remove_nodes(n_clicks, selected_nodes, store_data):
    if n_clicks > 0 and selected_nodes:
        updated_elements = filter_elements(store_data, set(selected_nodes))
        return updated_elements
    return store_data

# Callback to update details panel and heatmap on node tap.
@app.callback(
    [Output('heatmap-graph', 'figure'),
     Output('detailed-info', 'children')],
    Input('full-graph', 'tapNodeData')
)
def display_node_details(data):
    if not data:
        fig = px.imshow(np.zeros((10, 10)), color_continuous_scale='viridis')
        fig.update_layout(title="Select a node to view its details")
        return fig, "Click on a node (attention head or MLP) to see more information."
    
    if 'head' in data:
        layer_idx = data.get('layer')
        head_idx = data.get('head')
        layer_info = analysis_data['children'][layer_idx]
        selected_head = None
        for child in layer_info.get('children', []):
            if child['name'] == f"Attention Head {head_idx}":
                selected_head = child
                break
        if selected_head and 'matrix' in selected_head:
            mat = selected_head['matrix']
            fig = px.imshow(mat, color_continuous_scale='viridis')
            fig.update_layout(title=f"Attention Map: {selected_head['name']} (Layer {layer_idx})")
            details = html.Div([
                html.H3(selected_head['name']),
                html.P(selected_head.get('summary', '')),
                html.Pre(selected_head.get('detail', ''))
            ])
            return fig, details
        else:
            fig = px.imshow(np.zeros((10, 10)), color_continuous_scale='viridis')
            fig.update_layout(title=f"No matrix data for {data['label']}")
            return fig, f"No matrix data for {data['label']}"
    
    if data['label'] == "MLP":
        layer_idx = data.get('layer')
        layer_info = analysis_data['children'][layer_idx]
        selected_mlp = None
        for child in layer_info.get('children', []):
            if child['name'] == "MLP":
                selected_mlp = child
                break
        if selected_mlp:
            fig = px.imshow(np.zeros((10, 10)), color_continuous_scale='viridis')
            fig.update_layout(title=f"MLP (Layer {layer_idx})")
            details = html.Div([
                html.H3(selected_mlp['name']),
                html.P(selected_mlp.get('summary', '')),
                html.Pre(selected_mlp.get('detail', 'No additional detail'))
            ])
            return fig, details
        else:
            fig = px.imshow(np.zeros((10, 10)), color_continuous_scale='viridis')
            fig.update_layout(title=f"MLP data not found for Layer {layer_idx}")
            return fig, f"MLP data not found for Layer {layer_idx}"
    
    # Default for tokens or other nodes.
    fig = px.imshow(np.zeros((10, 10)), color_continuous_scale='viridis')
    fig.update_layout(title=f"{data['label']} selected - no additional details")
    return fig, html.Div([
        html.H4(data['label']),
        html.P("No detailed data available for this node.")
    ])

if __name__ == '__main__':
    app.run(debug=True)
