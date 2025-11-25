# SIGMA - Score-Informed Geometric Momentum Adaptation

I’m developing a PyTorch-based optimization framework that explores and benchmarks new stochastic optimization algorithms for deep learning. 

## Installation

```bash
pip install torch torchvision matplotlib numpy
```

## Quick Start

```python
from sigma_v2.optimizers_v2 import SIGMA_C_v2
import torch.optim as optim

# Initialize optimizer
optimizer = SIGMA_C_v2(model.parameters(), lr=0.01, beta=0.9)

# Training loop
for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()

        # SIGMA requires loss value
        optimizer.step(loss_item=loss.item())
```

## Available Versions

### sigma_v1/
Basic SIGMA implementation with two optimizer variants:
- **SIGMA-D**: Point D-based geometric score
- **SIGMA-C**: Point C-based geometric score

```bash
cd sigma_v1
python main.py
```

### sigma_v2/
Enhanced version with advanced features (warmup, weight decay, gradient clipping, AMSGrad):

```bash
cd sigma_v2
python main_v2.py
```

**sigma_v2_1/**: Extended with momentum type variants (sigma, classic, nesterov)

```bash
cd sigma_v2/sigma_v2_1
python main_v2_1.py
```

### sigma_v3/
Latest experimental versions including **SIGMA++** (smooth Adam-to-SIGMA transition):

```bash
cd sigma_v3
python test_plusplus.py
```

### sigma_complex/
Complex-valued neural network implementation:

```bash
cd sigma_complex
python main_complex.py
```

## Project Structure

```
SIGMA/
├── sigma_v1/           # Basic implementation
│   ├── main.py
│   ├── models.py
│   ├── optimizers.py
│   ├── utils.py
│   └── plotting.py
├── sigma_v2/           # Enhanced version
│   ├── sigma_v2_1/     # Momentum variants
│   ├── main_v2.py
│   ├── optimizers_v2.py
│   └── ...
├── sigma_v3/           # Experimental (SIGMA++)
└── sigma_complex/      # Complex-valued networks
```

## Key Features

- **Score-based optimization**: Adaptive learning based on loss landscape geometry
- **Hybrid training**: Sequential optimizer switching (e.g., Adam → SIGMA)
- **Momentum variants**: Support for sigma, classic, and Nesterov momentum
- **Advanced features**: Warmup, weight decay, gradient clipping, AMSGrad
- **Complex networks**: Support for complex-valued neural networks

## Experiments

All experiments run MNIST classification tasks comparing SIGMA optimizers against Adam and SGD+Momentum baselines.

**Output**: Each experiment generates PDF plots and console summaries with accuracy, loss, and timing metrics.

## Optimizer Parameters

Common hyperparameters:
- `lr`: Learning rate (default: 0.01)
- `beta`: Momentum coefficient (default: 0.9)
- `alpha_min`, `alpha_max`: Score bounds (default: [0.1, 2.0])
- `weight_decay`: L2 regularization (default: 0.01)
- `warmup_steps`: Warmup iterations (v2 only)
- `momentum_type`: 'sigma', 'classic', or 'nesterov' (v2.1 only)

## Important Notes

**All SIGMA optimizers require passing the loss value to `step()`:**

```python
optimizer.step(loss_item=loss.item())
```

This is critical for computing the geometric score that drives the optimization.

## License

Apache License 2.0 - Copyright 2025 José Wilson C. Souza
