# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SIGMA (Score-Informed Geometric Momentum Adaptation) is a PyTorch-based optimization framework that implements novel stochastic optimization algorithms based on geometric convexity scores computed from the loss function.

The project compares SIGMA optimizers against standard baselines (Adam, SGD+Momentum) on MNIST classification tasks using both neural networks and logistic regression models.

## Repository Structure

The codebase contains four main implementations:

- **sigma_v1/**: Basic SIGMA implementation with two optimizer variants (SIGMA-D and SIGMA-C)
- **sigma_v2/**: Enhanced version with advanced features (warmup, weight decay, gradient clipping, AMSGrad, second-order approximation)
  - **sigma_v2/sigma_v2_1/**: Extended v2 with momentum type variants (sigma, classic, nesterov)
- **sigma_v3/**: Latest experimental versions including SIGMA++ (hybrid optimizer with smooth Adam-to-SIGMA transition)
- **sigma_complex/**: Complex-valued neural network implementation of SIGMA optimizers

Each directory is self-contained with its own models, optimizers, utilities, and plotting modules.

## Core Architecture

### SIGMA Optimizers

The framework implements multiple score-based optimizers:

1. **SIGMA-D** (Theorem 1 - Point D): Computes a geometric score based on parameter interpolation at point D
   - Score formula uses `D1 = (θ_prev * θ_t) / (θ_prev + θ_t + ε)` and `D2 = (θ_t * L_prev + θ_prev * L_t) / (θ_prev + θ_t + ε)`
   - Requires passing `loss_item` to `step()` method

2. **SIGMA-C** (Theorem 2 - Point C): Computes a geometric score based on loss interpolation at point C
   - Score formula uses `C1 = (|L_prev| * θ_t + |L_t| * θ_prev) / (|L_prev| + |L_t| + ε)` and `C2 = (L_t * L_prev) / (|L_prev| + |L_t| + ε)`
   - Also requires passing `loss_item` to `step()` method

3. **SIGMA++** (v3): Hybrid optimizer that smoothly transitions from Adam to SIGMA-C using a cosine schedule
   - `λ(t)` interpolates between Adam (early epochs) and SIGMA (later epochs)
   - Combines adaptive learning rates with geometric scores
   - Automatically handles epoch tracking via `set_epoch()` method

**V2 Momentum Variants** (sigma_v2_1):
- `momentum_type='sigma'`: Uses score-based momentum (β on the score α)
- `momentum_type='classic'`: Classical momentum on parameters
- `momentum_type='nesterov'`: Nesterov accelerated gradient

**CRITICAL**: All SIGMA optimizers require the loss value to be passed to the step function:
```python
optimizer.step(loss_item=loss.item())
```

### Hybrid Sequential Training

The main experiment design uses a **hybrid sequential approach**:
1. **Phase 1**: Train with Adam for initial epochs to get to a good region
2. **Phase 2**: Switch to SIGMA (or SGD+M for control) for fine-tuning

This hybrid approach is the core experimental design comparing SIGMA's fine-tuning capabilities against SGD+Momentum.

**SIGMA++ Alternative**: Instead of discrete phase switching, SIGMA++ provides smooth interpolation between Adam and SIGMA-C using a learned transition schedule, eliminating the need for manual phase configuration.

### Model Architectures

- **MNISTNet**: 3-layer feedforward network (784 � 128 � 64 � 10) with ReLU activations
- **LogisticRegression**: Single linear layer (784 � 10)
- **ComplexMNISTNet**: Complex-valued network using CReLU activation (ReLU applied separately to real and imaginary parts)
- **ComplexLogisticRegression**: Complex-valued single layer

## Common Commands

### Running Experiments

```bash
# Run basic SIGMA experiments (v1)
cd sigma_v1
python main.py

# Run enhanced SIGMA experiments (v2 with advanced features)
cd sigma_v2
python main_v2.py

# Run v2.1 experiments (momentum type comparison)
cd sigma_v2/sigma_v2_1
python main_v2_1.py

# Run SIGMA++ experiments (hybrid optimizer with smooth transition)
cd sigma_v3
python test_plusplus.py

# Run complex-valued SIGMA experiments
cd sigma_complex
python main_complex.py
```

### Expected Outputs

All experiments:
- Print detailed training progress to console with loss and accuracy per epoch
- Generate PDF plots comparing optimizer performance
- Display final summary tables with accuracy, loss, and timing metrics

V1 generates:
- `sigma_hybrid_comparison_nn.pdf` (neural network results)
- `sigma_full_comparison_logistic.pdf` (logistic regression results)

V2 generates:
- `sigma_v2_hybrid_comparison_nn.pdf`
- `sigma_v2_full_comparison_logistic.pdf`

V2.1 generates:
- `sigma_summary_comparison_v3.pdf` (summary comparison of momentum types)
- `sigma_hybrid_comparison_nn_v3.pdf` (neural network detailed plots)
- `sigma_full_comparison_logistic_v3.pdf` (logistic regression plots)

V3 generates results based on the specific test scripts (see experiment output files: `exp_v3.txt`, `exp_v3-plus.txt`)

Complex generates:
- `complex_sigma_hybrid_comparison_nn.pdf`
- `complex_sigma_full_comparison_logistic.pdf`

### Dependencies

The project requires:
- PyTorch (with torchvision for MNIST datasets)
- matplotlib (for plotting)
- numpy

Install with:
```bash
pip install torch torchvision matplotlib numpy
```

## Key Implementation Details

### Data Loading
- MNIST dataset is automatically downloaded to `./data/` directory
- Default batch size: 128 for training, 1000 for testing
- Normalization: mean=0.1307, std=0.3081

### Training Configuration

**Neural Network Experiments:**
- Total epochs: 20 (10 Adam + 10 second optimizer)
- Adam learning rate: 0.001
- SGD/SIGMA learning rate: 0.01
- SIGMA beta (momentum): 0.9
- SIGMA alpha bounds: [0.1, 2.0]

**Logistic Regression Experiments:**
- Total epochs: 30 (15 + 15 split)
- Same learning rates and hyperparameters as neural network

**V2 Additions:**
- Weight decay: 0.01 (applied to all optimizers for fair comparison)
- Cyclic experiment: (Adam → SIGMA-C) × 2 with 5 epochs per phase

**V2.1 Additions:**
- Weight decay: 0.0 (disabled to isolate momentum effects)
- Tests three momentum types: 'sigma' (score-based), 'classic' (SGD-style), 'nesterov' (accelerated)

**V3 (SIGMA++) Configuration:**
- Smooth transition using cosine schedule: `λ(t) = 0.5 * (1 + cos(π * (t - switch_epoch) / transition_epochs))`
- Configurable switch_epoch (when transition begins) and transition_epochs (duration)
- Uses global gradient approximation for score computation

### Loss Function Behavior

SIGMA optimizers use the loss value to compute adaptive scores, making them distinct from gradient-based optimizers. The score α modulates the update: `θ_{t+1} = θ_t - lr * g_t * α_t`

The geometric score α is computed from the current and previous loss values (L_t, L_prev) and parameter norms, capturing the local geometry of the loss surface. This makes SIGMA particularly effective for fine-tuning in regions where the loss landscape exhibits strong geometric structure.

### Complex-Valued Networks

The complex implementation:
- Converts real MNIST data to complex dtype (`torch.cfloat`)
- Uses magnitude of output for loss computation and predictions
- Applies CReLU activation: `CReLU(z) = ReLU(Re(z)) + i�ReLU(Im(z))`
- SIGMA-D produces complex scores; SIGMA-C produces real scores

## Module Organization

Each version follows the same structure:

- **main.py / main_v2.py / main_complex.py / test_plusplus.py**: Entry point that orchestrates all experiments
- **models.py**: Neural network architectures (MNISTNet, LogisticRegression, and complex variants)
- **optimizers.py / optimizers_v2.py / optimizers_complex.py**: SIGMA optimizer implementations
- **utils.py / utils_v2.py / utils_complex.py**: Data loading and training loop functions
- **plotting.py / plotting_v2.py / plotting_complex.py**: Visualization generation

The modular design allows easy experimentation with different optimizer variants and model architectures.

### Key Cross-Version Differences

**sigma_v1**: Baseline implementation
- Basic SIGMA-D and SIGMA-C
- No advanced features

**sigma_v2**: Production-ready version
- Adds warmup, weight decay, gradient clipping, AMSGrad, second-order approximation
- `sigma_v2_1` sub-directory explores momentum variants

**sigma_v3**: Experimental cutting-edge
- SIGMA++ with smooth Adam→SIGMA transition
- Advanced hybrid strategies
- Currently in active development

**sigma_complex**: Research implementation
- Complex-valued neural networks
- CReLU activation for complex tensors
- Demonstrates SIGMA's generality beyond real-valued networks

## Important Implementation Patterns

### Training Loop with SIGMA Optimizers

All SIGMA optimizers require passing the loss value to `step()`. The typical training loop pattern:

```python
from optimizers import SIGMA_C, SIGMA_D  # or from optimizers_v2 import SIGMA_C_v2, etc.

# Training loop
for epoch in range(n_epochs):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()

        # CRITICAL: Pass loss.item() to SIGMA optimizers
        if isinstance(optimizer, (SIGMA_C, SIGMA_D)):  # Add other SIGMA variants as needed
            optimizer.step(loss_item=loss.item())
        else:
            optimizer.step()
```

### SIGMA++ Specific Pattern

SIGMA++ requires epoch tracking for its transition schedule:

```python
from sigma_v3.sigma_plusplus import SIGMAPlusPlus

optimizer = SIGMAPlusPlus(
    model.parameters(),
    lr=1e-3,
    switch_epoch=5,      # When to start transitioning
    transition_epochs=5   # How long the transition takes
)

for epoch in range(n_epochs):
    # Set current epoch for transition schedule
    optimizer.set_epoch(epoch)

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step(loss_item=loss.item())
```

### Hybrid Sequential Training Pattern

The `utils.py` / `utils_v2.py` modules provide `run_experiment()` which accepts an `optimizer_config` list for sequential phase switching:

```python
from utils_v2 import run_experiment

# Define optimizers for each phase
optimizer_phase1 = optim.Adam(model.parameters(), lr=1e-3)
optimizer_phase2 = SIGMA_C_v2(model.parameters(), lr=1e-2)

# Run experiment with phase switching
history, elapsed_time = run_experiment(
    experiment_name="Adam -> SIGMA-C",
    model=model,
    optimizer_config=[
        (optimizer_phase1, 10),  # 10 epochs with Adam
        (optimizer_phase2, 10)   # 10 epochs with SIGMA-C
    ],
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    loss_fn=loss_fn,
    n_epochs=20
)
```

## License

Apache License 2.0 - Copyright 2025 Jos� Wilson C. Souza
