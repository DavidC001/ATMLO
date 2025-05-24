# ATMLO: Circuit Discovery for Logic Tasks

Repository of the project for the master course "Advanced Topics in Machine Learning and Optimization Techiniques" at the University of Trento.

## Overview

This project aimed to discover and analyze the computational circuits used by LLMs to solve logical reasoning tasks. The primary objective was to use ACDC (Automatic Circuit Discovery) to identify the minimal set of model components responsible for logical inference. However, ACDC failed to provide meaningful results on logic tasks for several reasons, leading to an extensive investigation using multiple complementary techniques.

## Project Structure

### Core Components

- **`circuit_discovery/`**: Main ACDC implementation and circuit discovery experiments
- **`probing/`**: Linear classifier probing experiments to understand layer-wise representations  
- **`patching/`**: Manual activation patching analysis and token-level interventions
- **`viz/`**: Comprehensive visualization tools including interactive dashboards
- **`auto-circuit/`**: Modified AutoCircuit library with custom adaptations
- **`LogicBench/`**: Logic reasoning dataset for evaluation

### Key Datasets

- **LogicBench**: Systematic evaluation dataset for logical reasoning with 25 inference rules
- **Modus Tollens**: Primary logic task (if P→Q and ¬Q, then ¬P)
- **Custom preprocessed datasets**: Converted to ACDC-compatible format

## Methodology

### 1. Circuit Discovery with ACDC
**Objective**: Identify minimal circuits for logical reasoning  
**Implementation**: `circuit_discovery/circuit_discover.py`
- Used ACDC algorithm with various threshold configurations
- Applied edge attribution patching and mask gradient methods
- Attempted both token-level and component-level circuit discovery

**Key Challenge**: ACDC failed to converge on meaningful circuits for logic tasks, likely due to the distributed nature of logical reasoning across many model components.

### 2. Linear Probing Analysis
**Objective**: Understand how logical information is encoded across layers  
**Implementation**: `probing/train_probe.py`, `probing/probe_data_logic.py`
- Trained linear classifiers on hidden states from each transformer layer
- Analyzed which layers contain the most task-relevant information
- Investigated attention patterns and MLP activations

### 3. Manual Activation Patching
**Objective**: Systematically test component importance through intervention  
**Implementation**: `patching/activation_patch_analysis.py`
- Performed targeted ablations of attention heads and MLP layers
- Measured impact on logical reasoning performance
- Compared clean vs. corrupted activation patterns

### 4. Prompt Engineering for ACDC
**Objective**: Align prompts with ACDC's expected structure  
**Implementation**: Various preprocessing scripts in `utils/preprocess/`
- Reformatted LogicBench data for ACDC compatibility
- Created clean/corrupt prompt pairs following ACDC conventions
- Experimented with different prompt templates

### 5. Comprehensive Visualization
**Objective**: Understand model behavior through multiple visualization approaches  
**Implementation**: `viz/` directory
- Interactive dashboards for exploring attention patterns (`dash_viz_layers_probe.py`)
- ACDC circuit visualizations (`viz_ACDC.py`)
- Token-level patching analysis (`token_level_patching.py`)

## Key Findings

### Why ACDC Failed
1. **Distributed Processing**: Logical reasoning appears to involve many components working together rather than a minimal circuit
2. **Task Complexity**: Logic tasks may require more global reasoning than ACDC's local patching can capture
3. **Model Architecture**: Modern LLMs may not have the discrete, interpretable circuits that ACDC assumes

### Insights from Probing
- Logical information emerges gradually across layers
- Middle layers show strongest task-relevant representations
- Both attention and MLP components contribute to logical reasoning

### Manual Patching Results
- No single component is solely responsible for logical reasoning
- Performance degrades gradually with component removal
- Attention heads and MLPs have complementary roles

## Technical Details

### Models Tested
- **Qwen/Qwen2.5-0.5B-Instruct**
- **Qwen/Qwen2.5-1.5B-Instruct** 
- **meta-llama/Llama-3.2-1B-Instruct**
- **EleutherAI/pythia variants**

### Datasets
- **LogicBench**: 25 logical inference rules across propositional, first-order, and non-monotonic logic
- **Focus on Modus Tollens**: Classical logical inference pattern
- **Custom preprocessing**: Converted to clean/corrupt pairs for circuit discovery

### Techniques Applied
- **ACDC**: Automatic Circuit Discovery with various hyperparameters
- **Edge Attribution Patching**: Gradient-based circuit discovery (works only on gpt2 models)
- **Linear Probing**: Layer-wise representation analysis
- **Activation Pathchig**: Systematic activation patching to see the effects on the output
- **Visualization**: Multiple approaches for understanding model behavior

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Circuit Discovery
```bash
python circuit_discovery/circuit_discover.py
```

### Probing Experiments
```bash
python probing/train_probe.py
```

### Manual Patching
```bash
python patching/activation_patch_analysis.py
python patching/average_activation_patch_analysis.py
```

### Visualization
The script for visualization are located in the `viz/` directory.

## Configuration

Key parameters are controlled via `conf.yaml`:
- Model selection
- Dataset configuration  
- ACDC hyperparameters
- Visualization settings

## Results

Detailed results are stored in the `results/` directory, organized by:
- Model name
- Task type
- Analysis method
- Visualization outputs


## Acknowledgments

This project builds upon:
- [AutoCircuit](https://github.com/UFO-101/auto-circuit) by UFO-101
- [LogicBench](https://github.com/rahulnair23/LogicBench) dataset
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) library