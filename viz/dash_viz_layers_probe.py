"""
A Dash app to visualize the internal states of a transformer model.
This app allows users to select a sample JSON file containing model data,
and then interactively explore the model's architecture and attention heads.
"""

# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, Input, Output, State, ctx, Patch, callback
import dash_cytoscape as cyto
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
import glob
import re
from transformers import AutoTokenizer, AutoConfig
import torch
import traceback

# --- Configuration ---
DATA_DIR = "datasets/probing/Qwen/Qwen2.5-1.5B-Instruct/modus_tollens/"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
Model_Config = AutoConfig.from_pretrained(MODEL_NAME)
NUM_LAYERS = Model_Config.num_hidden_layers
NUM_HEADS = Model_Config.num_attention_heads
DEFAULT_SAMPLE_TYPE = "clean"

# --- Tokenizer Setup ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    interest_tokens = ["Yes", "yes", "No", "no"]
    # Each item can be more than one ID if the tokenizer splits it
    interest_tokens_ids_nested = tokenizer(interest_tokens, add_special_tokens=False)["input_ids"]
    # Flatten the nested lists
    interest_tokens_ids = [id_ for sublist in interest_tokens_ids_nested for id_ in sublist]
    # Map from ID -> token (helpful for the "interest_tokens_logits" decoding)
    interest_id_to_token_map = {}
    for token_str, sublist in zip(interest_tokens, interest_tokens_ids_nested):
        for id_val in sublist:
            interest_id_to_token_map[id_val] = token_str
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load tokenizer '{MODEL_NAME}'. Token decoding will not work. Error: {e}")
    tokenizer = None
    interest_tokens = ["Yes", "yes", "No", "no"]
    interest_tokens_ids = []
    interest_id_to_token_map = {}


# --- Helper Functions ---
def load_data_paths(base_dir):
    """Finds all JSON file paths in clean and corrupt subdirectories."""
    data_paths = {}
    for type_folder in ["clean", "corrupt"]:
        search_path = os.path.join(base_dir, type_folder, "*.json")
        files = glob.glob(search_path)
        try:
            sorted_files = sorted(
                [f for f in files if re.search(r'(\d+)\.json$', f)],
                key=lambda x: int(re.search(r'(\d+)\.json$', x).group(1))
            )
        except Exception as e:
            print(f"Warning: Could not sort files numerically in {type_folder}. Using default sort. Error: {e}")
            sorted_files = sorted(files)
        data_paths[type_folder] = sorted_files
    return data_paths

def load_json_data(file_path):
    """Loads data from a single JSON file."""
    if not file_path or not os.path.exists(file_path):
        return {"error": f"File not found or path invalid: {file_path}"}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        return {"error": f"Could not decode JSON from file: {file_path}"}
    except Exception as e:
        return {"error": f"Error loading file {file_path}: {e}"}

def create_model_graph_elements(num_layers):
    """Creates nodes and edges for the Cytoscape graph."""
    elements = []
    node_spacing_y = 120
    node_spacing_x = 150
    # Input node
    elements.append({
        'data': {'id': 'input', 'label': 'Input'},
        'position': {'x': 0, 'y': 0}
    })
    last_node_id = 'input'
    for i in range(num_layers):
        layer_y = (i + 1) * node_spacing_y
        attn_id = f'layer_{i}_attn'
        elements.append({
            'data': {'id': attn_id, 'label': f'Layer {i}\nAttention', 'type': 'attention', 'layer': i},
            'position': {'x': -node_spacing_x/2, 'y': layer_y}
        })
        elements.append({'data': {'source': last_node_id, 'target': attn_id}})

        mlp_id = f'layer_{i}_mlp'
        elements.append({
            'data': {'id': mlp_id, 'label': f'Layer {i}\nMLP', 'type': 'mlp', 'layer': i},
            'position': {'x': node_spacing_x/2, 'y': layer_y}
        })
        elements.append({'data': {'source': attn_id, 'target': mlp_id}})

        last_node_id = mlp_id

    # Output node
    elements.append({
        'data': {'id': 'output', 'label': 'Output'},
        'position': {'x': 0, 'y': (num_layers + 1) * node_spacing_y}
    })
    elements.append({'data': {'source': last_node_id, 'target': 'output'}})
    return elements

def create_heatmap_figure(matrix_data, layer_index, head_index, tokens=[]):
    """Generates a Plotly heatmap figure for a specific attention head."""
    if matrix_data is None:
        return go.Figure().update_layout(
            title=f"Layer {layer_index}: Matrix not found",
            height=1000, width=1000  # Increased size
        )
    try:
        attn_matrix = np.array(matrix_data)
        if len(attn_matrix.shape) == 3 and 0 <= head_index < attn_matrix.shape[0]:
            head_matrix = attn_matrix[head_index][10:,10:]
            # breakpoint()
            fig = go.Figure(data=go.Heatmap(z=head_matrix, colorscale='Viridis'))
            if len(tokens) > 0:
                # Create numbered token labels: "0: <token>"
                numbered_tokens = [f"{i}: {token}" for i, token in enumerate(tokens)]
                
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=list(range(len(tokens))),
                    ticktext=numbered_tokens,
                    title='Tokens',
                    ticks='outside',
                    tickangle=-45,
                    automargin=True
                )
                fig.update_yaxes(
                    tickmode='array',
                    tickvals=list(range(len(tokens))),
                    ticktext=numbered_tokens,
                    title='Tokens',
                    ticks='outside',
                    tickangle=-45,
                    automargin=True
                )
            
            fig.update_layout(
                title=f"Layer {layer_index}, Head {head_index} Attention",
                yaxis_autorange='reversed',
                height=1000, width=1000  # Increased size
            )
            return fig
        else:
            return go.Figure().update_layout(
                title=f"Layer {layer_index}: Invalid head/shape ({attn_matrix.shape})",
                height=1000, width=1000  # Increased size
            )
    except Exception as e:
        print(f"Error creating heatmap for Layer {layer_index}, Head {head_index}: {e}")
        return go.Figure().update_layout(
            title=f"Layer {layer_index}: Error generating heatmap",
            height=1000, width=1000  # Increased size
        )

# --- Load Data Paths ---
available_files = load_data_paths(DATA_DIR)
sample_options = {}
for type_key, paths in available_files.items():
    sample_options[type_key] = [{'label': os.path.basename(p), 'value': p} for p in paths]

# --- Dash App Initialization ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Transformer Visualization"

# --- Styles ---
cyto_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'label': 'data(label)',
            'background-color': '#88B',
            'shape': 'round-rectangle',
            'width': '80px', 'height': '50px',
            'text-valign': 'center', 'text-halign': 'center',
            'font-size': '10px', 'text-wrap': 'wrap'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'width': 2, 'line-color': '#999',
            'target-arrow-color': '#999'
        }
    },
    {
        'selector': '[type = "attention"]',
        'style': {'background-color': '#A8A'}
    },
    {
        'selector': '[type = "mlp"]',
        'style': {'background-color': '#8A8'}
    },
    {
        'selector': ':selected',
        'style': {'border-width': 3, 'border-color': '#FF0000'}
    },
]

# --- Main App Layout ---
app.layout = html.Div([
    html.H1("Transformer Internal State Visualization"),

    # Row 1: Sample Type + Sample JSON
    html.Div([
        html.Label("Select Sample Type:"),
        dcc.RadioItems(
            id='sample-type-selector',
            options=[{'label': k.capitalize(), 'value': k} for k in available_files.keys()],
            value=DEFAULT_SAMPLE_TYPE,
            inline=True,
            style={'marginLeft': '10px'}
        ),
        html.Label("Select Sample JSON:", style={'marginLeft': '30px'}),
        dcc.Dropdown(
            id='sample-selector',
            options=sample_options.get(DEFAULT_SAMPLE_TYPE, []),
            value=(
                sample_options.get(DEFAULT_SAMPLE_TYPE, [{}])[0].get('value', None)
                if sample_options.get(DEFAULT_SAMPLE_TYPE) else None
            ),
            style={'width': '400px', 'display': 'inline-block', 'marginLeft': '10px'}
        )
    ], style={'marginBottom': '20px'}),

    # Row 2: Graph on the left, Node Data on the right
    html.Div([
        # Left: Model Architecture Graph
        html.Div([
            html.H3("Model Architecture"),
            cyto.Cytoscape(
                id='cytoscape-model-graph',
                layout={'name': 'preset'},
                style={'width': '100%', 'height': '700px', 'border': '1px solid #ccc'},
                elements=create_model_graph_elements(NUM_LAYERS),
                stylesheet=cyto_stylesheet
            )
        ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '15px'}),

        # Right: Node Data + (Attention Controls)
        html.Div([
            html.H3("Selected Node Data"),
            dcc.Loading(
                id="loading-node-data",
                type="circle",
                children=html.Div(
                    id='node-data-output',
                    style={
                        'height': '400px',
                        'overflowY': 'scroll',
                        'border': '1px solid #ccc',
                        'padding': '10px',
                        'marginBottom': '20px'
                    }
                )
            ),

            # -- Always in Layout: Hidden container for attention controls --
            html.Div([
                html.H5("Select Attention Head:"),
                dcc.Dropdown(
                    id='attention-head-selector',
                    options=[{'label': f'Head {h}', 'value': h} for h in range(NUM_HEADS)],
                    value=0,
                    clearable=False,
                    style={'width': '150px', 'marginBottom': '10px'}
                ),
                dcc.Graph(id='attention-head-heatmap', figure=go.Figure())
            ],
            id='attention-control-panel',
            style={'display': 'none'}  # We toggle display on/off dynamically
            )
        ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ]),

    # Data stores
    dcc.Store(id='loaded-data-store'),
    dcc.Store(id='selected-node-store'),
])


# --- Callbacks ---
# 1) Update Sample Dropdown
@callback(
    Output('sample-selector', 'options'),
    Output('sample-selector', 'value'),
    Input('sample-type-selector', 'value'),
    State('sample-selector', 'value')  # Preserve current selection if possible
)
def update_sample_dropdown(selected_type, current_value):
    options = sample_options.get(selected_type, [])
    
    # Try to preserve the current selection by finding a file with the same basename
    value = None
    if current_value and options:
        current_basename = os.path.basename(current_value)
        # Look for a file with the same basename in the new sample type
        for option in options:
            if os.path.basename(option['value']) == current_basename:
                value = option['value']
                break
    
    # If we couldn't preserve the selection, use the first option
    if value is None and options:
        value = options[0]['value']
    
    return options, value

# 2) Load Selected Sample Data
@callback(
    Output('loaded-data-store', 'data'),
    Input('sample-selector', 'value'),
    prevent_initial_call=True
)
def load_selected_sample_data(selected_file_path):
    if selected_file_path:
        data = load_json_data(selected_file_path)
        if "error" in data:
            print(f"Error loading data: {data['error']}")
            return {'error': data['error']}
        return data
    return {}

# 3) Store Selected Node
@callback(
    Output('selected-node-store', 'data'),
    Input('cytoscape-model-graph', 'tapNodeData'),
    prevent_initial_call=True
)
def store_selected_node(node_data):
    return node_data

# 3b) Clear Selected Node when sample type or data changes
# @callback(
#     Output('selected-node-store', 'data', allow_duplicate=True),
#     Input('sample-type-selector', 'value'),
#     Input('loaded-data-store', 'data'),
#     prevent_initial_call=True
# )
# def clear_selected_node_on_data_change(sample_type, loaded_data):
#     return None

# 4) Display Node Data
@callback(
    Output('node-data-output', 'children'),
    Output('attention-control-panel', 'style'),  # We'll show/hide the attention panel
    Input('selected-node-store', 'data'),
    Input('loaded-data-store', 'data'),  # Changed from State to Input so it refreshes when data changes
    prevent_initial_call=True
)
def display_node_data(node_data, loaded_data):
    if not loaded_data:
        return ("Loading data...", {'display': 'none'})
    if "error" in loaded_data:
        return (
            html.Div([
                html.H4("Error loading sample data"),
                html.P(loaded_data['error'])
            ], style={'color': 'red'}),
            {'display': 'none'}
        )
    if not node_data:
        return ("Click a node in the graph to see its data for the selected sample.", {'display': 'none'})

    node_id = node_data['id']
    node_type = node_data.get('type')
    layer_index = node_data.get('layer')
    label = node_data.get('label', node_id)
    children = [html.H4(f"Data for Node: {label}")]

    # Helper: Display data item
    def display_data_item(title, data_key, data_source):
        """Reusable mini-function that tries to decode or show numeric stats."""
        item_children = []
        data_value = data_source.get(data_key)
        if data_value is None:
            return None  # Key not found, skip

        item_children.append(html.H5(title))

        if isinstance(data_value, list):
            data_array = None
            try:
                data_array = np.array(data_value)
            except:
                pass

            # If numeric, show shape/stats
            if data_array is not None and np.issubdtype(data_array.dtype, np.number):
                if data_array.size > 0:
                    stats_str = f"Shape: {data_array.shape} | Mean: {data_array.mean():.4f} | Std: {data_array.std():.4f}"
                else:
                    stats_str = f"Shape: {data_array.shape} (Empty)"
                item_children.append(html.P(stats_str))

            # --- Token decoding logic ---
            decoded_tokens_output = None
            should_decode = (
                tokenizer and data_array is not None
                and ("_ids" in data_key or "interest_tokens_logits" in data_key)
            )

            if should_decode:
                ids_to_decode = []
                is_top_k = "ids" in data_key
                is_interest = "interest_tokens_logits" in data_key
                
                try:
                    if is_top_k:
                        # Flatten and turn into a Python list of int
                        ids_to_decode = data_array.flatten().astype(int).tolist()
                    elif is_interest:
                        # We'll decode the interest_tokens_ids
                        # but the data_value is the logits. So we don't flatten these IDs; we already know them
                        ids_to_decode = interest_tokens_ids

                    if ids_to_decode:
                        tokens = []
                        for id_val in ids_to_decode:
                            try:
                                decoded = tokenizer.decode(int(id_val), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                                tokens.append(decoded if decoded else f"ID:{id_val}(?)")
                            except Exception as decode_err:
                                tokens.append(f"DecodeErr:{id_val}")

                        # Attempt to pair tokens with logits if available
                        logits_values = None
                        logits_array = None
                        if is_top_k:
                            # "top_logits_ids" => "top_logits"
                            logits_key = data_key.replace("_ids", "")
                            logits_values = data_source.get(logits_key)
                        elif is_interest:
                            # interest_tokens_logits => data_value is the logits
                            logits_key = data_key
                            logits_values = data_value

                        if logits_values is not None:
                            try:
                                logits_array = np.array(logits_values).flatten()
                            except Exception as logit_err:
                                pass

                        temp_decoded_list = []
                        if logits_array is not None and len(tokens) == len(logits_array):
                            # Combine tokens + logits
                            if is_top_k:
                                temp_decoded_list = [
                                    f"'{t}' (ID: {i}, Logit: {l:.3f})"
                                    for t, i, l in zip(tokens, ids_to_decode, logits_array)
                                ]
                            elif is_interest:
                                temp_decoded_list = [
                                    f"'{interest_id_to_token_map.get(i, 'UNK')}' (ID: {i}, Logit: {l:.3f})"
                                    for i, l in zip(ids_to_decode, logits_array)
                                ]
                        else:
                            # Fallback: just show the token + ID
                            temp_decoded_list = [
                                f"'{t}' (ID: {i})" for t, i in zip(tokens, ids_to_decode)
                            ]

                        if temp_decoded_list:
                            decoded_tokens_output = '\n'.join(temp_decoded_list)

                except Exception as e:
                    print(f"Error decoding token IDs for '{data_key}': {e}")
                    decoded_tokens_output = "Decoding error."

            # Display either the decoded tokens or a small snippet
            if decoded_tokens_output:
                item_children.append(html.Pre(
                    decoded_tokens_output,
                    style={
                        'whiteSpace': 'pre-wrap',
                        'wordBreak': 'break-all',
                        'maxHeight': '150px',
                        'overflowY': 'auto',
                        'backgroundColor': '#eee',
                        'padding': '3px',
                        'fontFamily': 'monospace'
                    }
                ))
            else:
                # Show partial list or JSON snippet
                if data_array is not None and len(data_array.shape) > 1:
                    snippet = json.dumps(data_value[:5], indent=2)
                else:
                    snippet = str(data_value[:10]) + ('...' if len(data_value) > 10 else '')
                item_children.append(html.Pre(
                    snippet,
                    style={
                        'whiteSpace': 'pre-wrap',
                        'wordBreak': 'break-all',
                        'maxHeight': '150px',
                        'overflowY': 'auto',
                        'backgroundColor': '#eee',
                        'padding': '3px'
                    }
                ))

        else:
            # If not a list, just show the value
            item_children.append(html.Pre(str(data_value), style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'}))

        return html.Div(item_children, style={'marginBottom': '10px'})

    # Populate Node Data
    if node_id == 'input':
        children.append(html.H5("Input Text:"))
        text_content = loaded_data.get('text', 'N/A')
        text_display = text_content if isinstance(text_content, str) else str(text_content)
        children.append(html.Div(
            text_display,
            style={
                'whiteSpace': 'pre-wrap',
                'fontFamily': 'monospace',
                'border': '1px solid lightgrey',
                'padding': '5px',
                'backgroundColor': '#f8f8f8'
            }
        ))
        # Hide attention panel
        return (children, {'display': 'none'})

    elif node_type == 'attention':
        # Show relevant attention data
        attn_key_base = f"attention_layer{layer_index}_"
        to_display = [
            ("Attention Output", f"{attn_key_base}out"),
            ("Attention Output (LN)", f"{attn_key_base}out_ln"),
            ("Top 5 Logits", f"{attn_key_base}top_logits"),
            ("Top 5 Tokens (IDs)", f"{attn_key_base}top_logits_ids"),
            ("Interest Token Logits", f"{attn_key_base}interest_tokens_logits"),
        ]
        for title, key in to_display:
            item = display_data_item(title, key, loaded_data)
            if item:
                children.append(item)

        # Also mention that the actual attention matrix is visible below
        children.append(html.P(
            "Use the dropdown below to select the attention head and view the attention heatmap."
        ))
        # Show attention panel
        return (children, {'display': 'block'})

    elif node_type == 'mlp':
        mlp_key_base = f"MLP_layer{layer_index}_"
        to_display = [
            ("MLP Input (Before)", f"{mlp_key_base}before"),
            ("MLP Output (After)", f"{mlp_key_base}after"),
            ("MLP Output (LN)", f"{mlp_key_base}after_ln"),
            ("Top 5 Logits (Before)", f"{mlp_key_base}top_logits_before"),
            ("Top 5 Tokens (Before)", f"{mlp_key_base}top_logits_before_ids"),
            ("Interest Token Logits (Before)", f"{mlp_key_base}interest_tokens_logits_before"),
            ("Top 5 Logits (After)", f"{mlp_key_base}top_logits_after"),
            ("Top 5 Tokens (After)", f"{mlp_key_base}top_logits_after_ids"),
            ("Interest Token Logits (After)", f"{mlp_key_base}interest_tokens_logits_after"),
        ]
        for title, key in to_display:
            item = display_data_item(title, key, loaded_data)
            if item:
                children.append(item)

        # --- Add Delta Calculation and Display for MLP Logits ---

        # 1. Top Logits Delta (After - Before)
        top_logits_before_key = f"{mlp_key_base}top_logits_before"
        top_logits_after_key = f"{mlp_key_base}top_logits_after"
        top_logits_before_val = loaded_data.get(top_logits_before_key)
        top_logits_after_val = loaded_data.get(top_logits_after_key)

        if top_logits_before_val is not None and top_logits_after_val is not None:
            try:
                before_arr = np.array(top_logits_before_val)
                after_arr = np.array(top_logits_after_val)
                if before_arr.shape == after_arr.shape and \
                   np.issubdtype(before_arr.dtype, np.number) and \
                   np.issubdtype(after_arr.dtype, np.number):
                    
                    delta_arr = after_arr - before_arr
                    delta_children_list = [html.H5("Top Logits Delta (After - Before)")]
                    
                    if delta_arr.size > 0:
                        stats_str = f"Shape: {delta_arr.shape} | Mean: {delta_arr.mean():.4f} | Std: {delta_arr.std():.4f}"
                    else:
                        stats_str = f"Shape: {delta_arr.shape} (Empty)"
                    delta_children_list.append(html.P(stats_str))
                    
                    snippet_arr = delta_arr.flatten()
                    snippet = str(snippet_arr[:10]) + ('...' if len(snippet_arr) > 10 else '')
                    delta_children_list.append(html.Pre(
                        snippet,
                        style={
                            'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all', 'maxHeight': '150px',
                            'overflowY': 'auto', 'backgroundColor': '#eee', 'padding': '3px',
                            'fontFamily': 'monospace'
                        }
                    ))
                    children.append(html.Div(delta_children_list, style={'marginBottom': '10px'}))
                else:
                    children.append(html.P(
                        f"Could not compute Top Logits Delta: Mismatch in shape or type. Before: {before_arr.shape}, After: {after_arr.shape}",
                        style={'color': 'orange'}
                    ))
            except Exception as e:
                children.append(html.P(f"Error calculating Top Logits Delta: {str(e)}", style={'color': 'red'}))

        # 2. Interest Token Logits Delta (After - Before)
        interest_before_key = f"{mlp_key_base}interest_tokens_logits_before"
        interest_after_key = f"{mlp_key_base}interest_tokens_logits_after"
        interest_before_val = loaded_data.get(interest_before_key)
        interest_after_val = loaded_data.get(interest_after_key)

        if interest_before_val is not None and interest_after_val is not None:
            try:
                before_arr = np.array(interest_before_val)
                after_arr = np.array(interest_after_val)

                if before_arr.shape == after_arr.shape and \
                   np.issubdtype(before_arr.dtype, np.number) and \
                   np.issubdtype(after_arr.dtype, np.number):

                    delta_arr = after_arr - before_arr
                    delta_children_list = [html.H5("Interest Token Logits Delta (After - Before)")]
                    
                    flat_delta_arr = delta_arr.flatten()

                    if tokenizer and interest_tokens_ids and len(flat_delta_arr) == len(interest_tokens_ids):
                        delta_display_text_list = []
                        for i, token_id_val in enumerate(interest_tokens_ids):
                            token_display_name = interest_id_to_token_map.get(token_id_val, f"ID:{token_id_val}(?)")
                            delta_val = flat_delta_arr[i]
                            delta_display_text_list.append(f"'{token_display_name}' (ID: {token_id_val}): {delta_val:.3f}")
                        
                        delta_children_list.append(html.Pre(
                            '\n'.join(delta_display_text_list),
                            style={
                                'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all', 'maxHeight': '150px',
                                'overflowY': 'auto', 'backgroundColor': '#eee', 'padding': '3px',
                                'fontFamily': 'monospace'
                            }
                        ))
                    else: # Fallback to numeric display
                        if delta_arr.size > 0:
                            stats_str = f"Shape: {delta_arr.shape} | Mean: {delta_arr.mean():.4f} | Std: {delta_arr.std():.4f}"
                        else:
                            stats_str = f"Shape: {delta_arr.shape} (Empty)"
                        delta_children_list.append(html.P(stats_str))
                        snippet = str(flat_delta_arr[:10]) + ('...' if len(flat_delta_arr) > 10 else '')
                        delta_children_list.append(html.Pre(
                            snippet,
                            style={
                                'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all', 'maxHeight': '150px',
                                'overflowY': 'auto', 'backgroundColor': '#eee', 'padding': '3px',
                                'fontFamily': 'monospace'
                            }
                        ))
                        if not (tokenizer and interest_tokens_ids):
                             delta_children_list.append(html.P("(Tokenizer/interest_tokens not available for detailed breakdown)", style={'fontSize':'small', 'color':'gray'}))
                        elif len(flat_delta_arr) != len(interest_tokens_ids):
                             delta_children_list.append(html.P(f"(Data length mismatch: delta has {len(flat_delta_arr)} values, expected {len(interest_tokens_ids)} for token breakdown)", style={'fontSize':'small', 'color':'orange'}))


                    children.append(html.Div(delta_children_list, style={'marginBottom': '10px'}))
                else:
                    children.append(html.P(
                        f"Could not compute Interest Token Logits Delta: Mismatch in shape or type. Before: {before_arr.shape}, After: {after_arr.shape}",
                        style={'color': 'orange'}
                    ))
            except Exception as e:
                children.append(html.P(f"Error calculating Interest Token Logits Delta: {str(e)}", style={'color': 'red'}))


        # Hide attention panel (not relevant for MLP node)
        return (children, {'display': 'none'})

    elif node_id == 'output':
        children.append(html.P("This node represents the final output stage."))
        return (children, {'display': 'none'})

    else:
        # If unknown node
        children.append(html.P("No specific data available for this node."))
        return (children, {'display': 'none'})  


# 5) Update the Attention Heatmap whenever the user selects a head, data changes, or node changes
@callback(
    Output('attention-head-heatmap', 'figure'),
    Input('attention-head-selector', 'value'),
    Input('selected-node-store', 'data'),  # Changed from State to Input so it updates when node changes
    Input('loaded-data-store', 'data'),  # Added as Input so it updates when data changes
    prevent_initial_call=True
)
def update_attention_heatmap(selected_head, node_data, loaded_data):
    """Generate the heatmap for the chosen attention head."""
    if not node_data or not loaded_data or 'error' in loaded_data:
        return go.Figure().update_layout(title="No Data Available", height=1000, width=1000)  # Increased size

    if node_data.get('type') != 'attention':
        # If the user last clicked an MLP or input node, just show a placeholder
        return go.Figure().update_layout(title="Not an Attention Node", height=1000, width=1000)  # Increased size

    layer_index = node_data.get('layer')
    attn_mat_key = f"attention_layer{layer_index}_attn_mat"
    attn_matrix_data = loaded_data.get(attn_mat_key, None)
    
    # get tokens from text
    text = loaded_data.get("text", "")
    if isinstance(text, str):
        tokens = tokenizer.batch_decode(tokenizer.encode(text))
    else:
        tokens = []
    
    return create_heatmap_figure(attn_matrix_data, layer_index, selected_head, tokens)


# --- Main Entrypoint ---
if __name__ == '__main__':
    if not os.path.isdir(DATA_DIR) or not any(available_files.values()):
        print("-" * 60 + "\nERROR: Data directory not found or empty.")
        print(f"  Checked: '{os.path.abspath(DATA_DIR)}'")
        print("  Ensure DATA_DIR is correct and the data generation script ran.\n" + "-" * 60)
    elif tokenizer is None:
        print("-" * 60 + "\nERROR: Tokenizer failed to load. Token decoding disabled.\n" + "-" * 60)
        print("Starting Dash server with limited functionality...")
        app.run(debug=True)
    else:
        print(f"Found data in: {os.path.abspath(DATA_DIR)}")
        for type_key, paths in available_files.items():
            print(f"  {type_key.capitalize()} samples: {len(paths)}")
        print("-" * 60 + "\nStarting Dash server at http://127.0.0.1:8050/\n" + "-" * 60)
        app.run(debug=True)