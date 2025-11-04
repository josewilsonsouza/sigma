"""
Benchmark Comparativo de Híbridos Sequenciais no MNIST (VERSÃO COMPLEXA)
======================================================================

Experimentos (Redes Neurais Complexas):
1. Adam (Baseline): 20 épocas contínuas
2. Híbrido (Adam → SGD+M): 10 épocas Adam + 10 épocas SGD+M
3. Híbrido (Adam → Complex_SIGMA-D): 10 épocas Adam + 10 épocas SIGMA (Score D)
4. Híbrido (Adam → Complex_SIGMA-C): 10 épocas Adam + 10 épocas SIGMA (Score C)

Experimentos (Regressão Logística Complexa):
5. Otimizadores Puros (Adam, SGD, SIGMA-D, SIGMA-C)
6. Híbridos (Adam → SGD, Adam → SIGMA-D, Adam → SIGMA-C)

Baseado no script _SIGMA.py e na Seção 5.1 das
anotações de J.W.C. Souza (2025). 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from typing import Optional, Callable
from torch.optim.optimizer import Optimizer

# ============================================================================
# OTIMIZADOR 1: Complex_SIGMA-D (Score Teorema 1 - Ponto D)
# ============================================================================

class Complex_SIGMA_D(Optimizer):
    """
    Implementação do SIGMA-D (Teorema 1) para PARÂMETROS COMPLEXOS.
    A loss (L_t, L_prev) é REAL.
    Os parâmetros (theta_t, theta_prev) e gradientes (g_t) são COMPLEXOS.
    O score (sigma) resultante é COMPLEXO.
    """
    def __init__(self, params, lr=1e-2, beta=None, alpha_min=0.1, alpha_max=2.0, eps=1e-8):
        # Definições idênticas à V1
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate inválido: {lr}")
        if beta is not None and not 0.0 <= beta < 1.0:
            raise ValueError(f"Beta inválido: {beta}. Deve estar em [0, 1)")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon inválido: {eps}")
        if not 0.0 < alpha_min < alpha_max:
            raise ValueError(f"Limites alpha inválidos: [{alpha_min}, {alpha_max}]")
            
        defaults = dict(
            lr=lr,
            beta=beta,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            eps=eps
        )
        super(Complex_SIGMA_D, self).__init__(params, defaults)
        
        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    @torch.no_grad()
    def step(self, loss_item=None):
        if loss_item is None:
            raise ValueError("SIGMA.step() requer 'loss_item'")
            
        loss_t = loss_item # REAL
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev'] # REAL
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                g_t = p.grad.data # COMPLEXO
                theta_t = p.data # COMPLEXO
                state = self.state[p]

                if len(state) == 0:
                    state['param_prev'] = theta_t.clone()
                    if beta is not None:
                        # .clone() preserva o dtype complexo
                        state['score_momentum'] = torch.ones_like(p.data) 
                
                theta_prev = state['param_prev'] # COMPLEXO
                
                if self.state['global_step'] > 1:
                    # O denominador é complexo
                    
                    # --- CORREÇÃO AQUI ---
                    # A linha original do sigma.py funciona
                    denom_xy = theta_prev + theta_t + eps
                    # --- FIM DA CORREÇÃO ---

                    # D1 = (complex * complex) / complex -> COMPLEXO
                    D1 = (theta_prev * theta_t) / denom_xy
                    # D2 = (complex * real + complex * real) / complex -> COMPLEXO
                    D2 = (theta_t * loss_prev + theta_prev * loss_t) / denom_xy
                    
                    # Aproximação de Taylor para f(2D₁)
                    # diff = 2*D1 - theta_t -> COMPLEXO
                    diff = (2 * D1 - theta_t)
                    # g_t * diff -> COMPLEXO (produto element-wise)
                    # f_proxy = real + real(complex * complex) -> REAL
                    f_proxy = loss_t + (g_t * diff).real
                    f_proxy = torch.clamp(f_proxy, min=eps) # Clamping REAL
                    
                    # sigma_raw = D2 (COMPLEXO) / f_proxy (REAL) -> COMPLEXO
                    sigma_raw = D2 / (f_proxy + eps)
                    
                    # Tratamento de valores inválidos (aplicado a real e imag)
                    sigma_raw[torch.isnan(sigma_raw)] = 1.0 + 0.0j
                    sigma_raw[torch.isinf(sigma_raw)] = 1.0 + 0.0j
                else:
                    sigma_raw = torch.ones_like(p.data) # COMPLEXO (1.0 + 0.0j)
                
                if beta is not None:
                    score_mom = state['score_momentum'] # COMPLEXO
                    score_mom.mul_(beta).add_(sigma_raw, alpha=1 - beta)
                    sigma_to_use = score_mom
                else:
                    sigma_to_use = sigma_raw
                
                # Clipping da MAGNITUDE do score complexo
                sigma_abs = sigma_to_use.abs()
                sigma_abs.clamp_(alpha_min, alpha_max)
                # Reaplicar a fase original
                sigma_to_use = torch.polar(sigma_abs, sigma_to_use.angle())

                # Atualização: p.data = p.data - lr * (g_t * sigma_to_use)
                # Esta é uma multiplicação COMPLEXA
                p.data.addcmul_(g_t, sigma_to_use, value=-lr)
                
                state['param_prev'] = theta_t.clone()
        
        self.state['global_loss_prev'] = loss_t
        return loss_t
    
    def __repr__(self):
        variant = "Complex_SIGMA-M (Score D)" if self.defaults['beta'] is not None else "Complex_SIGMA-T (Score D)"

        return f"{variant}(lr={self.defaults['lr']}, beta={self.defaults['beta']})"
# ============================================================================
# OTIMIZADOR 2: Complex_SIGMA-C (Score Teorema 2 - Ponto C)
# ============================================================================

class Complex_SIGMA_C(Optimizer):
    """
    Implementação do SIGMA-C (Teorema 2) para PARÂMETROS COMPLEXOS.
    A loss (L_t, L_prev) é REAL.
    Os parâmetros (theta_t, theta_prev) e gradientes (g_t) são COMPLEXOS.
    O score (sigma) resultante é REAL.
    """
    def __init__(self, params, lr=1e-2, beta=None, alpha_min=0.1, alpha_max=2.0, eps=1e-8):
        # Definições idênticas à V1
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate inválido: {lr}")
        if beta is not None and not 0.0 <= beta < 1.0:
            raise ValueError(f"Beta inválido: {beta}. Deve estar em [0, 1)")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon inválido: {eps}")
        if not 0.0 < alpha_min < alpha_max:
            raise ValueError(f"Limites alpha inválidos: [{alpha_min}, {alpha_max}]")
            
        defaults = dict(
            lr=lr,
            beta=beta,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            eps=eps
        )
        super(Complex_SIGMA_C, self).__init__(params, defaults)
        
        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    @torch.no_grad()
    def step(self, loss_item=None):
        if loss_item is None:
            raise ValueError("SIGMA.step() requer 'loss_item'")
            
        loss_t = loss_item # REAL
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev'] # REAL
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                g_t = p.grad.data # COMPLEXO
                theta_t = p.data # COMPLEXO
                state = self.state[p]

                if len(state) == 0:
                    state['param_prev'] = theta_t.clone()
                    if beta is not None:
                        # Score momentum é REAL para SIGMA-C
                        state['score_momentum'] = torch.ones_like(p.data.real) 
                
                theta_prev = state['param_prev'] # COMPLEXO
                
                if self.state['global_step'] > 1:
                    # Denominador é REAL
                    denom_L = abs(loss_prev) + abs(loss_t) + eps
                    
                    # C1 = (real * complex + real * complex) / real -> COMPLEXO
                    C1 = (abs(loss_prev) * theta_t + abs(loss_t) * theta_prev) / denom_L
                    
                    # C2 = (real * real) / real -> REAL
                    C2 = (loss_t * loss_prev) / denom_L
                    
                    # Aproximação de Taylor para f(C₁)
                    # diff = C1 - theta_t -> COMPLEXO
                    diff = C1 - theta_t
                    # f_proxy = real + real(complex * complex) -> REAL
                    f_proxy = loss_t + (g_t * diff).real
                    f_proxy = torch.clamp(f_proxy, min=eps) # Clamping REAL
                    
                    # sigma_raw = 2 * C2 (REAL) / f_proxy (REAL) -> REAL
                    sigma_raw = (2 * C2) / (f_proxy + eps)
                    
                    sigma_raw[torch.isnan(sigma_raw)] = 1.0
                    sigma_raw[torch.isinf(sigma_raw)] = 1.0
                else:
                    sigma_raw = torch.ones_like(p.data.real) # REAL
                
                if beta is not None:
                    score_mom = state['score_momentum'] # REAL
                    score_mom.mul_(beta).add_(sigma_raw, alpha=1 - beta)
                    sigma_to_use = score_mom
                else:
                    sigma_to_use = sigma_raw
                
                sigma_to_use.clamp_(alpha_min, alpha_max) # REAL

                # Atualização: p.data = p.data - lr * (g_t * sigma_to_use)
                # Esta é uma multiplicação COMPLEXO * REAL
                p.data.addcmul_(g_t, sigma_to_use, value=-lr)
                
                state['param_prev'] = theta_t.clone()
        
        self.state['global_loss_prev'] = loss_t
        return loss_t
    
    def __repr__(self):
        variant = "Complex_SIGMA-M (Score C)" if self.defaults['beta'] is not None else "Complex_SIGMA-T (Score C)"
        return f"{variant}(lr={self.defaults['lr']}, beta={self.defaults['beta']})"


# ============================================================================
# 1. DEFINIÇÕES DO MODELO E DADOS (VERSÃO COMPLEXA)
# ============================================================================

def CReLU(z):
    """ Ativação CReLU: ReLU(real) + i * ReLU(imag) """
    return F.relu(z.real) + 1j * F.relu(z.imag)

class ComplexMNISTNet(nn.Module):
    """Rede Neural Feedforward COMPLEXA (784 → 128 → 64 → 10)."""
    
    def __init__(self):
        super(ComplexMNISTNet, self).__init__()
        # Camadas lineares agora usam dtype=torch.cfloat
        self.fc1 = nn.Linear(784, 128, dtype=torch.cfloat)
        self.fc2 = nn.Linear(128, 64, dtype=torch.cfloat)
        self.fc3 = nn.Linear(64, 10, dtype=torch.cfloat)
    
    def forward(self, x):
        # x já é complexo
        x = x.view(-1, 784)
        x = CReLU(self.fc1(x))
        x = CReLU(self.fc2(x))
        return self.fc3(x) # Saída complexa


class ComplexLogisticRegression(nn.Module):
    """Regressão Logística COMPLEXA (784 → 10)."""
    
    def __init__(self):
        super(ComplexLogisticRegression, self).__init__()
        self.linear = nn.Linear(784, 10, dtype=torch.cfloat)
    
    def forward(self, x):
        # x já é complexo
        x = x.view(-1, 784)
        return self.linear(x) # Saída complexa


def get_data_loaders(batch_size=128):
    """Carrega os datasets MNIST (dados permanecem reais aqui)."""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader


# ============================================================================
# 2. FUNÇÕES DE TREINAMENTO E AVALIAÇÃO (ADAPTADAS PARA COMPLEXO)
# ============================================================================

def train_epoch(model, optimizer, train_loader, device, loss_fn):
    """Treina o modelo por uma época."""
    
    model.train()
    total_loss = 0
    
    for data, target in train_loader:
        # data (REAL) -> data (COMPLEXO)
        data = data.to(device).to(torch.cfloat)
        target = target.to(device) # REAL
        
        optimizer.zero_grad()
        output = model(data) # COMPLEXO
        
        # CrossEntropyLoss não aceita input complexo.
        # Usamos a magnitude da saída para calcular a loss.
        loss = loss_fn(output.abs(), target) # REAL
        
        loss.backward()
        loss_item = loss.item() # REAL

        # Passa loss_item (real) para os otimizadores
        if isinstance(optimizer, (Complex_SIGMA_D, Complex_SIGMA_C)):
            optimizer.step(loss_item=loss_item) 
        else:
            optimizer.step()
        
        total_loss += loss_item
        
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device, loss_fn):
    """Avalia o modelo no conjunto de teste."""
    
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # data (REAL) -> data (COMPLEXO)
            data = data.to(device).to(torch.cfloat)
            target = target.to(device) # REAL
            
            output = model(data) # COMPLEXO
            
            # Usamos a magnitude para loss e predição
            output_real_abs = output.abs()
            loss = loss_fn(output_real_abs, target) # REAL
            test_loss += loss.item() * data.size(0) 
            
            pred = output_real_abs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


# ============================================================================
# 3. FUNÇÃO PARA EXECUTAR EXPERIMENTOS (Sem alteração)
# ============================================================================

def run_experiment(experiment_name, model, optimizer_config, train_loader, test_loader, 
                   device, loss_fn, n_epochs):
    """
    Executa um experimento de treinamento.
    """
    
    print("\n" + "="*80)
    print(f"TREINANDO: {experiment_name}")
    print("="*80)
    
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    start_time = time.time()
    epoch_counter = 0
    
    for phase_idx, (optimizer, phase_epochs) in enumerate(optimizer_config):
        phase_name = f"Fase {phase_idx + 1}" if len(optimizer_config) > 1 else "Única"
        print(f"\n--- {phase_name}: {optimizer.__class__.__name__} ---")
        
        for epoch in range(phase_epochs):
            train_loss = train_epoch(model, optimizer, train_loader, device, loss_fn)
            test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)
            
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            epoch_counter += 1
            print(f"Época [{epoch_counter:2d}/{n_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.2f}%")
    
    elapsed_time = time.time() - start_time
    print(f"\n--- {experiment_name} concluído em {elapsed_time:.2f}s ---")
    
    return history, elapsed_time


# ============================================================================
# 4. SCRIPT PRINCIPAL DE COMPARAÇÃO (ADAPTADO PARA COMPLEXO)
# ============================================================================

def main():
    """Executa todos os experimentos complexos e gera análises."""
    
    # Configuração
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {DEVICE}")
    
    N_EPOCHS_NN_TOTAL = 20
    N_EPOCHS_NN_PHASE1 = 10
    N_EPOCHS_NN_PHASE2 = N_EPOCHS_NN_TOTAL - N_EPOCHS_NN_PHASE1
    
    N_EPOCHS_LR_TOTAL = 30
    N_EPOCHS_LR_PHASE1 = 15
    N_EPOCHS_LR_PHASE2 = N_EPOCHS_LR_TOTAL - N_EPOCHS_LR_PHASE1
    
    LR_ADAM = 0.001
    LR_SGD = 0.01
    LR_SIGMA = 0.01
    
    # Dados
    train_loader, test_loader = get_data_loaders()
    loss_fn = nn.CrossEntropyLoss()
    
    # =======================================================================
    # PARTE 1: EXPERIMENTOS COM REDES NEURAIS COMPLEXAS
    # =======================================================================
    
    print("\n" + "="*80)
    print("PARTE 1: EXPERIMENTOS COM REDES NEURAIS COMPLEXAS")
    print("="*80)
    
    base_model = ComplexMNISTNet().to(DEVICE)
    
    results_nn = {}
    times_nn = {}
    
    # --- EXPERIMENTO 1: Adam (Baseline) ---
    model_adam = copy.deepcopy(base_model)
    # optim.Adam suporta parâmetros complexos nativamente
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=LR_ADAM)
    
    history_adam, time_adam = run_experiment(
        experiment_name="Complex Adam (Baseline)",
        model=model_adam,
        optimizer_config=[(optimizer_adam, N_EPOCHS_NN_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Complex Adam (Baseline)'] = history_adam
    times_nn['Complex Adam (Baseline)'] = time_adam
    
    # --- EXPERIMENTO 2: Híbrido (Adam → SGD+M) ---
    model_hybrid_sgd = copy.deepcopy(base_model)
    optimizer_adam_phase1_ctrl = optim.Adam(model_hybrid_sgd.parameters(), lr=LR_ADAM)
    # optim.SGD suporta parâmetros complexos nativamente
    optimizer_sgd_phase2 = optim.SGD(model_hybrid_sgd.parameters(), lr=LR_SGD, momentum=0.9)
    
    history_hybrid_sgd, time_hybrid_sgd = run_experiment(
        experiment_name="Híbrido (Adam -> Complex SGD+M)",
        model=model_hybrid_sgd,
        optimizer_config=[
            (optimizer_adam_phase1_ctrl, N_EPOCHS_NN_PHASE1),
            (optimizer_sgd_phase2, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Adam -> Complex SGD+M'] = history_hybrid_sgd
    times_nn['Adam -> Complex SGD+M'] = time_hybrid_sgd

    # --- EXPERIMENTO 3: Híbrido (Adam → Complex_SIGMA-D) ---
    model_hybrid_sigma_d = copy.deepcopy(base_model)
    optimizer_adam_phase1_d = optim.Adam(model_hybrid_sigma_d.parameters(), lr=LR_ADAM)
    optimizer_sigma_d = Complex_SIGMA_D(
        model_hybrid_sigma_d.parameters(),
        lr=LR_SIGMA,
        beta=0.9,
        alpha_min=0.1,
        alpha_max=2.0
    )
    
    history_hybrid_sigma_d, time_hybrid_sigma_d = run_experiment(
        experiment_name="Híbrido (Adam -> Complex SIGMA-D)",
        model=model_hybrid_sigma_d,
        optimizer_config=[
            (optimizer_adam_phase1_d, N_EPOCHS_NN_PHASE1),
            (optimizer_sigma_d, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Adam -> Complex SIGMA-D'] = history_hybrid_sigma_d
    times_nn['Adam -> Complex SIGMA-D'] = time_hybrid_sigma_d

    # --- EXPERIMENTO 4: Híbrido (Adam → Complex_SIGMA-C) ---
    model_hybrid_sigma_c = copy.deepcopy(base_model)
    optimizer_adam_phase1_c = optim.Adam(model_hybrid_sigma_c.parameters(), lr=LR_ADAM)
    optimizer_sigma_c = Complex_SIGMA_C(
        model_hybrid_sigma_c.parameters(),
        lr=LR_SIGMA,
        beta=0.9,
        alpha_min=0.1,
        alpha_max=2.0
    )
    
    history_hybrid_sigma_c, time_hybrid_sigma_c = run_experiment(
        experiment_name="Híbrido (Adam -> Complex SIGMA-C)",
        model=model_hybrid_sigma_c,
        optimizer_config=[
            (optimizer_adam_phase1_c, N_EPOCHS_NN_PHASE1),
            (optimizer_sigma_c, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Adam -> Complex SIGMA-C'] = history_hybrid_sigma_c
    times_nn['Adam -> Complex SIGMA-C'] = time_hybrid_sigma_c
    
    
    # =======================================================================
    # PARTE 2: EXPERIMENTOS COM REGRESSÃO LOGÍSTICA COMPLEXA
    # =======================================================================
    
    print("\n" + "="*80)
    print("PARTE 2: EXPERIMENTOS COM REGRESSÃO LOGÍSTICA COMPLEXA")
    print("="*80)
    
    base_logistic = ComplexLogisticRegression().to(DEVICE)
    results_lr = {}
    times_lr = {}
    
    # --- Experimento LR-Puro 1: Adam ---
    model_lr_adam = copy.deepcopy(base_logistic)
    optimizer_lr_adam = optim.Adam(model_lr_adam.parameters(), lr=LR_ADAM)
    history_lr_adam, time_lr_adam = run_experiment(
        experiment_name="[LR] Complex Adam (Puro)",
        model=model_lr_adam,
        optimizer_config=[(optimizer_lr_adam, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Complex Adam (Puro)'] = history_lr_adam
    times_lr['Complex Adam (Puro)'] = time_lr_adam
    
    # --- Experimento LR-Puro 2: SGD+Momentum ---
    model_lr_sgd = copy.deepcopy(base_logistic)
    optimizer_lr_sgd = optim.SGD(model_lr_sgd.parameters(), lr=LR_SGD, momentum=0.9)
    history_lr_sgd, time_lr_sgd = run_experiment(
        experiment_name="[LR] Complex SGD+M (Puro)",
        model=model_lr_sgd,
        optimizer_config=[(optimizer_lr_sgd, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Complex SGD+M (Puro)'] = history_lr_sgd
    times_lr['Complex SGD+M (Puro)'] = time_lr_sgd
    
    # --- Experimento LR-Puro 3: Complex_SIGMA-D ---
    model_lr_sigma_d = copy.deepcopy(base_logistic)
    optimizer_lr_sigma_d = Complex_SIGMA_D(
        model_lr_sigma_d.parameters(), lr=LR_SIGMA, beta=0.9
    )
    history_lr_sigma_d, time_lr_sigma_d = run_experiment(
        experiment_name="[LR] Complex SIGMA-D (Puro)",
        model=model_lr_sigma_d,
        optimizer_config=[(optimizer_lr_sigma_d, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Complex SIGMA-D (Puro)'] = history_lr_sigma_d
    times_lr['Complex SIGMA-D (Puro)'] = time_lr_sigma_d

    # --- Experimento LR-Puro 4: Complex_SIGMA-C ---
    model_lr_sigma_c = copy.deepcopy(base_logistic)
    optimizer_lr_sigma_c = Complex_SIGMA_C(
        model_lr_sigma_c.parameters(), lr=LR_SIGMA, beta=0.9
    )
    history_lr_sigma_c, time_lr_sigma_c = run_experiment(
        experiment_name="[LR] Complex SIGMA-C (Puro)",
        model=model_lr_sigma_c,
        optimizer_config=[(optimizer_lr_sigma_c, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Complex SIGMA-C (Puro)'] = history_lr_sigma_c
    times_lr['Complex SIGMA-C (Puro)'] = time_lr_sigma_c
    
    # --- Experimento LR-Híbrido 1: Adam -> SGD+M ---
    model_lr_h_sgd = copy.deepcopy(base_logistic)
    optimizer_lr_h_adam = optim.Adam(model_lr_h_sgd.parameters(), lr=LR_ADAM)
    optimizer_lr_h_sgd = optim.SGD(model_lr_h_sgd.parameters(), lr=LR_SGD, momentum=0.9)
    
    history_lr_h_sgd, time_lr_h_sgd = run_experiment(
        experiment_name="[LR] Adam -> Complex SGD+M",
        model=model_lr_h_sgd,
        optimizer_config=[
            (optimizer_lr_h_adam, N_EPOCHS_LR_PHASE1),
            (optimizer_lr_h_sgd, N_EPOCHS_LR_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Adam -> Complex SGD+M'] = history_lr_h_sgd
    times_lr['Adam -> Complex SGD+M'] = time_lr_h_sgd

    # --- Experimento LR-Híbrido 2: Adam -> Complex_SIGMA-D ---
    model_lr_h_sigmad = copy.deepcopy(base_logistic)
    optimizer_lr_h_adam_d = optim.Adam(model_lr_h_sigmad.parameters(), lr=LR_ADAM)
    optimizer_lr_h_sigmad = Complex_SIGMA_D(
        model_lr_h_sigmad.parameters(), lr=LR_SIGMA, beta=0.9
    )
    
    history_lr_h_sigmad, time_lr_h_sigmad = run_experiment(
        experiment_name="[LR] Adam -> Complex SIGMA-D",
        model=model_lr_h_sigmad,
        optimizer_config=[
            (optimizer_lr_h_adam_d, N_EPOCHS_LR_PHASE1),
            (optimizer_lr_h_sigmad, N_EPOCHS_LR_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Adam -> Complex SIGMA-D'] = history_lr_h_sigmad
    times_lr['Adam -> Complex SIGMA-D'] = time_lr_h_sigmad

    # --- Experimento LR-Híbrido 3: Adam -> Complex_SIGMA-C ---
    model_lr_h_sigmac = copy.deepcopy(base_logistic)
    optimizer_lr_h_adam_c = optim.Adam(model_lr_h_sigmac.parameters(), lr=LR_ADAM)
    optimizer_lr_h_sigmac = Complex_SIGMA_C(
        model_lr_h_sigmac.parameters(), lr=LR_SIGMA, beta=0.9
    )
    
    history_lr_h_sigmac, time_lr_h_sigmac = run_experiment(
        experiment_name="[LR] Adam -> Complex SIGMA-C",
        model=model_lr_h_sigmac,
        optimizer_config=[
            (optimizer_lr_h_adam_c, N_EPOCHS_LR_PHASE1),
            (optimizer_lr_h_sigmac, N_EPOCHS_LR_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Adam -> Complex SIGMA-C'] = history_lr_h_sigmac
    times_lr['Adam -> Complex SIGMA-C'] = time_lr_h_sigmac
    
    # ========================================================================
    # GERAÇÃO DE PLOTS E RESUMOS
    # ========================================================================
    
    # --- Gráficos da Rede Neural Complexa ---
    print("\nGerando gráficos de comparação (Redes Neurais Complexas)...")
    fig_nn, axes_nn = plt.subplots(2, 2, figsize=(18, 12))
    colors_nn = {
        'Complex Adam (Baseline)': '#1f77b4',
        'Adam -> Complex SGD+M': '#d62728',
        'Adam -> Complex SIGMA-D': '#2ca02c',
        'Adam -> Complex SIGMA-C': '#9467bd',
    }
    markers_nn = {
        'Complex Adam (Baseline)': 'o',
        'Adam -> Complex SGD+M': 'v',
        'Adam -> Complex SIGMA-D': '^',
        'Adam -> Complex SIGMA-C': 'P',
    }
    
    ax1 = axes_nn[0, 0]
    for name, history in results_nn.items():
        ax1.plot(history['test_acc'], label=name, color=colors_nn[name], marker=markers_nn[name], markersize=4, linewidth=2)
    ax1.axvline(x=N_EPOCHS_NN_PHASE1 - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Troca')
    ax1.set_title('Acurácia (Rede Neural Complexa)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Acurácia no Teste (%)', fontsize=12)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes_nn[0, 1]
    for name, history in results_nn.items():
        ax2.plot(history['test_loss'], label=name, color=colors_nn[name], marker=markers_nn[name], markersize=4, linewidth=2)
    ax2.axvline(x=N_EPOCHS_NN_PHASE1 - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Troca')
    ax2.set_title('Loss de Teste (Rede Neural Complexa)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Loss no Teste (escala log)', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes_nn[1, 0]
    for name, history in results_nn.items():
        ax3.plot(history['train_loss'], label=name, color=colors_nn[name], marker=markers_nn[name], markersize=3, linewidth=2, alpha=0.8)
    ax3.axvline(x=N_EPOCHS_NN_PHASE1 - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_title('Loss de Treino (Rede Neural Complexa)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Época', fontsize=12)
    ax3.set_ylabel('Loss de Treino', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes_nn[1, 1]
    names_nn = list(times_nn.keys())
    time_values_nn = list(times_nn.values())
    bars = ax4.barh(names_nn, time_values_nn, color=[colors_nn[n] for n in names_nn])
    for i, (bar, val) in enumerate(zip(bars, time_values_nn)):
        ax4.text(val + 1, i, f'{val:.1f}s', va='center', fontsize=10)
    ax4.set_title('Eficiência Computacional (Rede Neural Complexa)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Tempo de Treinamento (s)', fontsize=12)
    ax4.grid(axis='x', alpha=0.3)
    
    fig_nn.tight_layout()
    fig_nn.savefig('complex_sigma_hybrid_comparison_nn.pdf', dpi=150, bbox_inches='tight')
    print("Gráfico salvo: 'complex_sigma_hybrid_comparison_nn.pdf'")
    
    # --- Gráficos da Regressão Logística Complexa ---
    print("\nGerando gráficos de comparação (Regressão Logística Complexa)...")
    fig_lr, axes_lr = plt.subplots(2, 2, figsize=(18, 12))
    colors_lr = {
        'Complex Adam (Puro)': '#1f77b4',
        'Complex SGD+M (Puro)': '#ff7f0e',
        'Complex SIGMA-D (Puro)': '#2ca02c',
        'Complex SIGMA-C (Puro)': '#9467bd',
        'Adam -> Complex SGD+M': '#d62728',
        'Adam -> Complex SIGMA-D': '#8c564b',
        'Adam -> Complex SIGMA-C': '#e377c2',
    }
    markers_lr = { k: 'o' if 'Puro' in k else '^' for k in colors_lr.keys() }
    linestyles_lr = { k: ':' if 'Puro' in k else '-' for k in colors_lr.keys() }
    
    ax1_lr = axes_lr[0, 0]
    for name, history in results_lr.items():
        ax1_lr.plot(history['test_acc'], label=name, color=colors_lr[name], linestyle=linestyles_lr[name], marker=markers_lr[name], markersize=4, linewidth=2)
    ax1_lr.axvline(x=N_EPOCHS_LR_PHASE1 - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Troca')
    ax1_lr.set_title('Acurácia (Regressão Logística Complexa)', fontsize=14, fontweight='bold')
    ax1_lr.set_xlabel('Época', fontsize=12)
    ax1_lr.set_ylabel('Acurácia no Teste (%)', fontsize=12)
    ax1_lr.legend(fontsize=9, loc='lower right')
    ax1_lr.grid(True, alpha=0.3)
    
    ax2_lr = axes_lr[0, 1]
    for name, history in results_lr.items():
        ax2_lr.plot(history['test_loss'], label=name, color=colors_lr[name], linestyle=linestyles_lr[name], marker=markers_lr[name], markersize=4, linewidth=2)
    ax2_lr.axvline(x=N_EPOCHS_LR_PHASE1 - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Troca')
    ax2_lr.set_title('Loss de Teste (Regressão Logística Complexa)', fontsize=14, fontweight='bold')
    ax2_lr.set_xlabel('Época', fontsize=12)
    ax2_lr.set_ylabel('Loss no Teste (escala log)', fontsize=12)
    ax2_lr.set_yscale('log')
    ax2_lr.legend(fontsize=9, loc='upper right')
    ax2_lr.grid(True, alpha=0.3)
    
    ax3_lr = axes_lr[1, 0]
    for name, history in results_lr.items():
        ax3_lr.plot(history['train_loss'], label=name, color=colors_lr[name], linestyle=linestyles_lr[name], marker=markers_lr[name], markersize=4, linewidth=2)
    ax3_lr.axvline(x=N_EPOCHS_LR_PHASE1 - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Troca')
    ax3_lr.set_title('Loss de Treino (Regressão Logística Complexa)', fontsize=14, fontweight='bold')
    ax3_lr.set_xlabel('Época', fontsize=12)
    ax3_lr.set_ylabel('Loss de Treino (escala log)', fontsize=12)
    ax3_lr.set_yscale('log')
    ax3_lr.legend(fontsize=9, loc='upper right')
    ax3_lr.grid(True, alpha=0.3)
    
    ax4_lr = axes_lr[1, 1]
    names_lr = list(times_lr.keys())
    time_values_lr = list(times_lr.values())
    bars = ax4_lr.barh(names_lr, time_values_lr, color=[colors_lr[n] for n in names_lr])
    for i, (bar, val) in enumerate(zip(bars, time_values_lr)):
        ax4_lr.text(val + 1, i, f'{val:.1f}s', va='center', fontsize=10)
    ax4_lr.set_title('Eficiência Computacional (Regressão Logística Complexa)', fontsize=14, fontweight='bold')
    ax4_lr.set_xlabel('Tempo de Treinamento (s)', fontsize=12)
    ax4_lr.grid(axis='x', alpha=0.3)
    
    fig_lr.tight_layout()
    fig_lr.savefig('complex_sigma_full_comparison_logistic.pdf', dpi=150, bbox_inches='tight')
    print("Gráfico salvo: 'complex_sigma_full_comparison_logistic.pdf'")
    
    # ========================================================================
    # RESUMOS ESTATÍSTICOS (TERMINAL)
    # ========================================================================
    
    print("\n" + "="*80)
    print("RESUMO FINAL - REDES NEURAIS (COMPLEXAS)")
    print("="*80)
    print(f"\n{'Experimento':<30} | {'Acc Final':<10} | {'Loss Final':<11} | {'Tempo (s)':<10}")
    print("-" * 70)
    for name in results_nn.keys():
        acc_final = results_nn[name]['test_acc'][-1]
        loss_final = results_nn[name]['test_loss'][-1]
        time_final = times_nn[name]
        print(f"{name:<30} | {acc_final:>9.2f}% | {loss_final:>10.4f} | {time_final:>9.2f}s")
    
    print("\n" + "="*80)
    print("RESUMO FINAL - REGRESSÃO LOGÍSTICA (COMPLEXA)")
    print("="*80)
    print(f"\n{'Otimizador':<30} | {'Acc Final':<10} | {'Loss Final':<11} | {'Tempo (s)':<10}")
    print("-" * 70)
    for name in results_lr.keys():
        acc_final = results_lr[name]['test_acc'][-1]
        loss_final = results_lr[name]['train_loss'][-1]
        time_final = times_lr[name]
        print(f"{name:<30} | {acc_final:>9.2f}% | {loss_final:>10.6f} | {time_final:>9.2f}s")
    
    print("\n" + "="*80)
    print("Benchmark complexo concluído com sucesso!")
    print("="*80)

if __name__ == "__main__":
    main()