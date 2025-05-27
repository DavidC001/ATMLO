# Advanced Topics in Machine Learning and Optimization Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🚀 Key Features

- **Circuit Discovery**: Implements multiple algorithms including ACDC (Automated Circuit DisCovery), Edge Attribution Patching, and Subnetwork Probing
- **LogicBench Integration**: Built-in evaluation suite for logical reasoning tasks
- **Activation Patching**: Comprehensive tools for analyzing model behavior through intervention techniques
- **Probing Analysis**: Utilities for investigating internal model representations
- **Multi-Model Support**: Compatible with GPT-2, Qwen, Llama, and Pythia model families (for the most part)
- **Visualization Tools**: Interactive dashboards and plotting utilities for analysis results
- **Benchmarking Pipeline**: Automated evaluation framework for systematic experiments

## 📁 Repository Structure

```
ATMLO/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Main configuration settings
├── conf.yaml                   # YAML configuration file
├── benchmark.py                # Benchmarking pipeline
├── report.tex                  # Research report (LaTeX)
│
├── auto-circuit/               # AutoCircuit library for circuit discovery
│   ├── auto_circuit/          # Core circuit discovery algorithms
│   ├── experiments/           # Experimental scripts and demos
│   ├── datasets/              # Task-specific datasets
│   └── docs/                  # Documentation
│
├── LogicBench/                # LogicBench dataset for logical reasoning
│   ├── data/                  # Logic reasoning datasets
│   └── README.md              # LogicBench documentation
│
├── circuit_discovery/         # Circuit discovery implementations
├── patching/                  # Activation patching analysis tools
├── probing/                   # Probing utilities and models
├── datasets/                  # Experiment datasets and results
├── results/                   # Experimental results storage
├── utils/                     # Utility functions and data loaders
└── viz/                       # Visualization tools and notebooks
```

## 🛠️ Installation

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/DavidC001/ATMLO.git
cd ATMLO
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Get LogicBench data:**
```bash
git clone https://github.com/Mihir3009/LogicBench.git
```

## 🔬 Usage

After setting up the configuration file, you should first run the benchmark pipeline to evaluate the model's logical reasoning capabilities and create the necessary datasets:
```bash
python benchmark.py
```

### Circuit Discovery
```bash
python circuit_discovery/circuit_discovery.py
```
This will create a directory in `results/{model_name}/` with the discovered circuits attribution scores.
These scores can be later used to see the extracted circuit using the notebook `viz/viz_ACDC.ipynb`.

### Activation Patching
We have implemented two versions of activation patching: one where a single sample is used to patch the model, and another where multiple samples are used to create a more robust patch.
However, due to several limitations, we were only able to use implement the patching at token-level for the first.

For the single-sample patching, you can run:
```bash
python patching/activation_patch_analysis.py
```
This will create a directory in `results/{model_name}/patching/` with the patching results for the specified model.
These can later be visualized using the provided visualization script `viz/token_level_patching.py` and passing the argument `-i {file_path}`. You can also use the `viz/avg_patching.py` script to visualize the result at layer level (we use the max value).

For the multi-sample patching, you can use:
```bash
python patching/average_activation_patch_analysis.py
```
This will create a directory in `results/{model_name}/patching/` with the patching results for the specified model.
These can later be visualized using the provided visualization script `viz/avg_patching.py` and passing the argument `-i {file_path}`.

### Probing
To run probing analysis, you first need to create the probing datasets:
```bash
python probing/probe_data_logic.py
```
This will create a directory in `datasets/probing/` with the probing datasets for the specified model.

Then, you can run the probe training and evaluation:
```bash
python probing/train_probe.py
```
This will train the probes on the probing datasets and evaluate them.

Additionaly, you can visualize the collected data for the probing analysis:
```bash
python viz/dash_viz_layers_probe.py
```
This will create an interactive dashboard to visualize the probing results for the different layers, attention heads, and samples.
