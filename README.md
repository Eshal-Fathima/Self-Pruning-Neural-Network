# Self-Pruning Neural Network

This repository implements a **Self-Pruning Neural Network** that automatically discovers sparse architectures during training using learnable sigmoid gates.

## Overview

Traditional pruning often involves post-hoc removal of weights based on magnitude. This implementation uses **Gated Layers** where each neuron's output is scaled by a learnable parameter $g \in [0, 1]$.

### Pruning Mechanism
The network is optimized using a composite loss function:
$$Loss = \text{CrossEntropy} + \lambda \times \sum_{i} \text{sigmoid}(gate_i)$$

- **Accuracy Focus**: The CrossEntropy term ensures the network learns to classify correctly.
- **Sparsity Focus**: The $\lambda$ (Lambda) penalty forces unnecessary gates toward zero, effectively "turning off" neurons that don't contribute meaningfully to the objective.

## Repository Structure

- `solution.py`: The core script that performs training, pruning analysis, and result generation.
- `requirements.txt`: Project dependencies (PyTorch, Matplotlib, etc.).
- `results/`:
  - `results_table.md`: A summary of Accuracy vs. Sparsity across different Lambda values.
  - `gate_distribution.png`: A histogram showing the distribution of gate values after training.

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Experiment**:
   ```bash
   python solution.py
   ```

## Results

Typical results show that as $\lambda$ increases, the network becomes significantly sparser (more neurons pruned) while maintaining high accuracy up to a certain critical threshold.

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|--------------|--------------|
| 0.0    | ~94.0        | 0.0          |
| 0.01   | ~92.5        | ~85.0        |

> [!TIP]
> Check the `results/` folder after running the script to see the generated distribution plots!
