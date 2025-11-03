"""
Benchmark SIGMA vs Baselines no MNIST
======================================

Experimentos (Redes Neurais):
1. Adam (Baseline): 20 épocas contínuas
2. SGD+Momentum (Baseline forte): 20 épocas contínuas  
3. SIGMA-M (Puro): 20 épocas contínuas
4. Híbrido (Adam → SIGMA): 10 épocas Adam + 10 épocas SIGMA-M
5. Híbrido (Adam → SGD+M): 10 épocas Adam + 10 épocas SGD+M (controle)

Experimentos (Regressão Logística):
6. Adam vs SIGMA-M vs SGD+M em problema convexo

Objetivo: Verificar se o ganho do Híbrido (Adam → SIGMA) vem do SIGMA em si,
ou apenas do efeito de trocar para um otimizador mais simples na Fase 2.
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

# Importar o otimizador SIGMA
try:
    from sigma import SIGMA
except ImportError:
    print("="*80)
    print("ERRO: O arquivo 'sigma.py' não foi encontrado.")
    print("Certifique-se de que ambos os arquivos estão no mesmo diretório.")
    print("="*80)
    exit()

# ============================================================================
# 1. DEFINIÇÕES DO MODELO E DADOS
# ============================================================================

class MNISTNet(nn.Module):
    """Rede Neural Feedforward simples para MNIST (784 → 128 → 64 → 10)."""
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LogisticRegression(nn.Module):
    """Regressão Logística para MNIST (784 → 10)."""
    
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(784, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        return self.linear(x)


def get_data_loaders(batch_size=128):
    """Carrega os datasets MNIST para treino e teste."""
    
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
# 2. FUNÇÕES DE TREINAMENTO E AVALIAÇÃO
# ============================================================================

def train_epoch(model, optimizer, train_loader, device, loss_fn):
    """Treina o modelo por uma época."""
    
    model.train()
    total_loss = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        # Closure para calcular loss e gradientes
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        loss_item = loss.item()

        # SIGMA requer loss_item; outros otimizadores não
        if isinstance(optimizer, SIGMA):
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
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            test_loss += loss.item() * data.size(0) 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


# ============================================================================
# 3. FUNÇÃO PARA EXECUTAR EXPERIMENTOS
# ============================================================================

def run_experiment(experiment_name, model, optimizer_config, train_loader, test_loader, 
                   device, loss_fn, n_epochs):
    """
    Executa um experimento de treinamento.
    
    Args:
        experiment_name: Nome do experimento
        model: Modelo PyTorch
        optimizer_config: Lista de tuplas (optimizer, n_epochs) para cada fase
        train_loader, test_loader: DataLoaders
        device: Device (CPU/GPU)
        loss_fn: Função de perda
        n_epochs: Total de épocas
    
    Returns:
        history: Dicionário com métricas de treino/teste
        elapsed_time: Tempo total de treinamento
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
# 4. SCRIPT PRINCIPAL DE COMPARAÇÃO
# ============================================================================

def main():
    """Executa todos os experimentos e gera análises comparativas."""
    
    # Configuração
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {DEVICE}")
    
    N_EPOCHS_TOTAL = 20
    N_EPOCHS_PHASE1 = 10  # Para experimentos híbridos
    N_EPOCHS_PHASE2 = N_EPOCHS_TOTAL - N_EPOCHS_PHASE1
    
    # Dados
    train_loader, test_loader = get_data_loaders()
    loss_fn = nn.CrossEntropyLoss()
    
    # =======================================================================
    # PARTE 1: EXPERIMENTOS COM REDES NEURAIS
    # =======================================================================
    
    print("\n" + "="*80)
    print("PARTE 1: EXPERIMENTOS COM REDES NEURAIS")
    print("="*80)
    
    # Modelo base (será copiado para cada experimento)
    base_model = MNISTNet().to(DEVICE)
    
    # Armazenar resultados
    results_nn = {}
    times_nn = {}
    
    # ========================================================================
    # EXPERIMENTO 1: Adam (Baseline)
    # ========================================================================
    
    model_adam = copy.deepcopy(base_model)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
    
    history_adam, time_adam = run_experiment(
        experiment_name="Adam (Baseline)",
        model=model_adam,
        optimizer_config=[(optimizer_adam, N_EPOCHS_TOTAL)],
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        loss_fn=loss_fn,
        n_epochs=N_EPOCHS_TOTAL
    )
    
    results_nn['Adam (Baseline)'] = history_adam
    times_nn['Adam (Baseline)'] = time_adam
    
    # ========================================================================
    # EXPERIMENTO 2: SGD+Momentum (Baseline Forte)
    # ========================================================================
    
    model_sgd = copy.deepcopy(base_model)
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
    
    history_sgd, time_sgd = run_experiment(
        experiment_name="SGD+Momentum (Baseline)",
        model=model_sgd,
        optimizer_config=[(optimizer_sgd, N_EPOCHS_TOTAL)],
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        loss_fn=loss_fn,
        n_epochs=N_EPOCHS_TOTAL
    )
    
    results_nn['SGD+Momentum'] = history_sgd
    times_nn['SGD+Momentum'] = time_sgd
    
    # ========================================================================
    # EXPERIMENTO 3: SIGMA-M Puro (20 épocas)
    # ========================================================================
    
    model_sigma = copy.deepcopy(base_model)
    optimizer_sigma_pure = SIGMA(
        model_sigma.parameters(),
        lr=0.01,
        beta=0.9,
        alpha_min=0.1,
        alpha_max=2.0
    )
    
    history_sigma, time_sigma = run_experiment(
        experiment_name="SIGMA-M (Puro)",
        model=model_sigma,
        optimizer_config=[(optimizer_sigma_pure, N_EPOCHS_TOTAL)],
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        loss_fn=loss_fn,
        n_epochs=N_EPOCHS_TOTAL
    )
    
    results_nn['SIGMA-M (Puro)'] = history_sigma
    times_nn['SIGMA-M (Puro)'] = time_sigma
    
    # ========================================================================
    # EXPERIMENTO 4: Híbrido (Adam → SIGMA-M)
    # ========================================================================
    
    model_hybrid_sigma = copy.deepcopy(base_model)
    
    # Fase 1: Adam (convergência rápida)
    optimizer_adam_phase1 = optim.Adam(model_hybrid_sigma.parameters(), lr=0.001)
    
    # Fase 2: SIGMA-M (fine-tuning)
    optimizer_sigma = SIGMA(
        model_hybrid_sigma.parameters(),
        lr=0.01,
        beta=0.9,
        alpha_min=0.1,
        alpha_max=2.0
    )
    
    history_hybrid_sigma, time_hybrid_sigma = run_experiment(
        experiment_name="Híbrido (Adam → SIGMA-M)",
        model=model_hybrid_sigma,
        optimizer_config=[
            (optimizer_adam_phase1, N_EPOCHS_PHASE1),
            (optimizer_sigma, N_EPOCHS_PHASE2)
        ],
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        loss_fn=loss_fn,
        n_epochs=N_EPOCHS_TOTAL
    )
    
    results_nn['Híbrido (Adam → SIGMA-M)'] = history_hybrid_sigma
    times_nn['Híbrido (Adam → SIGMA-M)'] = time_hybrid_sigma
    
    # ========================================================================
    # EXPERIMENTO 5: Híbrido (Adam → SGD+M) - GRUPO DE CONTROLE
    # ========================================================================
    
    model_hybrid_sgd = copy.deepcopy(base_model)
    
    # Fase 1: Adam
    optimizer_adam_phase1_ctrl = optim.Adam(model_hybrid_sgd.parameters(), lr=0.001)
    
    # Fase 2: SGD+Momentum (controle para verificar efeito da troca)
    optimizer_sgd_phase2 = optim.SGD(model_hybrid_sgd.parameters(), lr=0.01, momentum=0.9)
    
    history_hybrid_sgd, time_hybrid_sgd = run_experiment(
        experiment_name="Híbrido (Adam → SGD+M) [Controle]",
        model=model_hybrid_sgd,
        optimizer_config=[
            (optimizer_adam_phase1_ctrl, N_EPOCHS_PHASE1),
            (optimizer_sgd_phase2, N_EPOCHS_PHASE2)
        ],
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        loss_fn=loss_fn,
        n_epochs=N_EPOCHS_TOTAL
    )
    
    results_nn['Híbrido (Adam → SGD+M)'] = history_hybrid_sgd
    times_nn['Híbrido (Adam → SGD+M)'] = time_hybrid_sgd
    
    # =======================================================================
    # PARTE 2: EXPERIMENTOS COM REGRESSÃO LOGÍSTICA
    # =======================================================================
    
    print("\n" + "="*80)
    print("PARTE 2: EXPERIMENTOS COM REGRESSÃO LOGÍSTICA (PROBLEMA CONVEXO)")
    print("="*80)
    
    base_logistic = LogisticRegression().to(DEVICE)
    results_lr = {}
    times_lr = {}
    
    N_EPOCHS_LR = 30  # Mais épocas para convergência em problema convexo
    
    # Experimento LR1: Adam
    model_lr_adam = copy.deepcopy(base_logistic)
    optimizer_lr_adam = optim.Adam(model_lr_adam.parameters(), lr=0.001)
    
    history_lr_adam, time_lr_adam = run_experiment(
        experiment_name="[LR] Adam",
        model=model_lr_adam,
        optimizer_config=[(optimizer_lr_adam, N_EPOCHS_LR)],
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        loss_fn=loss_fn,
        n_epochs=N_EPOCHS_LR
    )
    
    results_lr['Adam'] = history_lr_adam
    times_lr['Adam'] = time_lr_adam
    
    # Experimento LR2: SGD+Momentum
    model_lr_sgd = copy.deepcopy(base_logistic)
    optimizer_lr_sgd = optim.SGD(model_lr_sgd.parameters(), lr=0.01, momentum=0.9)
    
    history_lr_sgd, time_lr_sgd = run_experiment(
        experiment_name="[LR] SGD+Momentum",
        model=model_lr_sgd,
        optimizer_config=[(optimizer_lr_sgd, N_EPOCHS_LR)],
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        loss_fn=loss_fn,
        n_epochs=N_EPOCHS_LR
    )
    
    results_lr['SGD+Momentum'] = history_lr_sgd
    times_lr['SGD+Momentum'] = time_lr_sgd
    
    # Experimento LR3: SIGMA-M
    model_lr_sigma = copy.deepcopy(base_logistic)
    optimizer_lr_sigma = SIGMA(
        model_lr_sigma.parameters(),
        lr=0.01,
        beta=0.9,
        alpha_min=0.1,
        alpha_max=2.0
    )
    
    history_lr_sigma, time_lr_sigma = run_experiment(
        experiment_name="[LR] SIGMA-M",
        model=model_lr_sigma,
        optimizer_config=[(optimizer_lr_sigma, N_EPOCHS_LR)],
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        loss_fn=loss_fn,
        n_epochs=N_EPOCHS_LR
    )
    
    results_lr['SIGMA-M'] = history_lr_sigma
    times_lr['SIGMA-M'] = time_lr_sigma
    
    # ========================================================================
    # VISUALIZAÇÃO DOS RESULTADOS - REDES NEURAIS
    # ========================================================================
    
    print("\nGerando gráficos de comparação (Redes Neurais)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Cores e estilos
    colors_nn = {
        'Adam (Baseline)': '#1f77b4',
        'SGD+Momentum': '#ff7f0e',
        'SIGMA-M (Puro)': '#9467bd',
        'Híbrido (Adam → SIGMA-M)': '#2ca02c',
        'Híbrido (Adam → SGD+M)': '#d62728'
    }
    
    markers_nn = {
        'Adam (Baseline)': 'o',
        'SGD+Momentum': 's',
        'SIGMA-M (Puro)': 'D',
        'Híbrido (Adam → SIGMA-M)': '^',
        'Híbrido (Adam → SGD+M)': 'v'
    }
    
    # Plot 1: Acurácia no Teste
    ax1 = axes[0, 0]
    for name, history in results_nn.items():
        ax1.plot(history['test_acc'], 
                label=name, 
                color=colors_nn[name],
                marker=markers_nn[name], 
                markersize=4,
                linewidth=2)
    
    ax1.axvline(x=N_EPOCHS_PHASE1 - 0.5, color='gray', linestyle='--', 
                linewidth=1.5, alpha=0.7, label='Troca de Otimizador')
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Acurácia no Teste (%)', fontsize=12)
    ax1.set_title('Comparação de Acurácia: SIGMA vs Baselines (Redes Neurais)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss no Teste (escala log)
    ax2 = axes[0, 1]
    for name, history in results_nn.items():
        ax2.plot(history['test_loss'], 
                label=name,
                color=colors_nn[name],
                marker=markers_nn[name], 
                markersize=4,
                linewidth=2)
    
    ax2.axvline(x=N_EPOCHS_PHASE1 - 0.5, color='gray', linestyle='--', 
                linewidth=1.5, alpha=0.7, label='Troca de Otimizador')
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Loss no Teste (escala log)', fontsize=12)
    ax2.set_title('Comparação de Loss (Redes Neurais)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss de Treino
    ax3 = axes[1, 0]
    for name, history in results_nn.items():
        ax3.plot(history['train_loss'], 
                label=name,
                color=colors_nn[name],
                marker=markers_nn[name], 
                markersize=3,
                linewidth=2,
                alpha=0.8)
    
    ax3.axvline(x=N_EPOCHS_PHASE1 - 0.5, color='gray', linestyle='--', 
                linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Época', fontsize=12)
    ax3.set_ylabel('Loss de Treino', fontsize=12)
    ax3.set_title('Convergência no Conjunto de Treino (Redes Neurais)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Comparação de Tempo de Treinamento (Barras)
    ax4 = axes[1, 1]
    names_nn = list(times_nn.keys())
    time_values_nn = list(times_nn.values())
    bars = ax4.barh(names_nn, time_values_nn, color=[colors_nn[n] for n in names_nn])
    
    # Adicionar valores nas barras
    for i, (bar, val) in enumerate(zip(bars, time_values_nn)):
        ax4.text(val + 1, i, f'{val:.1f}s', va='center', fontsize=10)
    
    ax4.set_xlabel('Tempo de Treinamento (s)', fontsize=12)
    ax4.set_title('Eficiência Computacional (Redes Neurais)', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sigma_comparison_nn.png', dpi=150, bbox_inches='tight')
    print("Gráfico salvo: 'sigma_comparison_nn.png'")
    
    # ========================================================================
    # VISUALIZAÇÃO - REGRESSÃO LOGÍSTICA
    # ========================================================================
    
    print("\nGerando gráficos de comparação (Regressão Logística)...")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    
    colors_lr = {
        'Adam': '#1f77b4',
        'SGD+Momentum': '#ff7f0e',
        'SIGMA-M': '#2ca02c'
    }
    
    markers_lr = {
        'Adam': 'o',
        'SGD+Momentum': 's',
        'SIGMA-M': '^'
    }
    
    # Plot 1: Acurácia
    ax1_lr = axes2[0]
    for name, history in results_lr.items():
        ax1_lr.plot(history['test_acc'], 
                   label=name, 
                   color=colors_lr[name],
                   marker=markers_lr[name], 
                   markersize=5,
                   linewidth=2.5)
    
    ax1_lr.set_xlabel('Época', fontsize=12)
    ax1_lr.set_ylabel('Acurácia no Teste (%)', fontsize=12)
    ax1_lr.set_title('Regressão Logística: Acurácia (Problema Convexo)', fontsize=14, fontweight='bold')
    ax1_lr.legend(fontsize=11, loc='lower right')
    ax1_lr.grid(True, alpha=0.3)
    
    # Plot 2: Loss (escala log)
    ax2_lr = axes2[1]
    for name, history in results_lr.items():
        ax2_lr.plot(history['train_loss'], 
                   label=name,
                   color=colors_lr[name],
                   marker=markers_lr[name], 
                   markersize=5,
                   linewidth=2.5)
    
    ax2_lr.set_xlabel('Época', fontsize=12)
    ax2_lr.set_ylabel('Loss de Treino (escala log)', fontsize=12)
    ax2_lr.set_title('Regressão Logística: Convergência', fontsize=14, fontweight='bold')
    ax2_lr.set_yscale('log')
    ax2_lr.legend(fontsize=11, loc='upper right')
    ax2_lr.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sigma_comparison_logistic.png', dpi=150, bbox_inches='tight')
    print("Gráfico salvo: 'sigma_comparison_logistic.png'")
    
    # ========================================================================
    # RESUMO ESTATÍSTICO - REDES NEURAIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("RESUMO FINAL - REDES NEURAIS")
    print("="*80)
    
    # Tabela de resultados finais
    print(f"\n{'Experimento':<35} | {'Acc Final':<10} | {'Loss Final':<11} | {'Tempo (s)':<10}")
    print("-" * 80)
    
    for name in results_nn.keys():
        acc_final = results_nn[name]['test_acc'][-1]
        loss_final = results_nn[name]['test_loss'][-1]
        time_final = times_nn[name]
        print(f"{name:<35} | {acc_final:>9.2f}% | {loss_final:>10.4f} | {time_final:>9.2f}s")
    
    # Análise de melhorias
    print("\n" + "-"*80)
    print("ANÁLISE DE MELHORIAS (vs Adam Baseline)")
    print("-"*80)
    
    baseline_acc = results_nn['Adam (Baseline)']['test_acc'][-1]
    baseline_time = times_nn['Adam (Baseline)']
    
    for name in results_nn.keys():
        if name == 'Adam (Baseline)':
            continue
        
        acc_diff = results_nn[name]['test_acc'][-1] - baseline_acc
        time_diff = times_nn[name] - baseline_time
        
        print(f"\n{name}:")
        print(f"  Δ Acurácia: {acc_diff:+.2f}% {'✓' if acc_diff > 0 else '✗'}")
        print(f"  Δ Tempo: {time_diff:+.2f}s {'✓' if time_diff < 0 else '✗'}")
    
    # Comparação crítica: SIGMA vs SGD+M (ambos híbridos)
    print("\n" + "="*80)
    print("COMPARAÇÃO CRÍTICA: Híbrido SIGMA vs Híbrido SGD+M")
    print("="*80)
    print("Pergunta: O ganho vem do SIGMA ou apenas da troca de otimizador?")
    print("-"*80)
    
    acc_sigma = results_nn['Híbrido (Adam → SIGMA-M)']['test_acc'][-1]
    acc_sgd = results_nn['Híbrido (Adam → SGD+M)']['test_acc'][-1]
    diff_critical = acc_sigma - acc_sgd
    
    print(f"\nAcurácia Híbrido (Adam → SIGMA-M): {acc_sigma:.2f}%")
    print(f"Acurácia Híbrido (Adam → SGD+M):   {acc_sgd:.2f}%")
    print(f"Diferença: {diff_critical:+.2f}%")
    
    if abs(diff_critical) < 0.1:
        print("\n⚠️  RESULTADO: Diferença desprezível!")
        print("   → O efeito pode ser apenas da troca de otimizador, não do SIGMA.")
    elif diff_critical > 0:
        print(f"\n✓ RESULTADO: SIGMA é {diff_critical:.2f}% superior!")
        print("   → O ganho é específico do método SIGMA.")
    else:
        print(f"\n✗ RESULTADO: SGD+M é {abs(diff_critical):.2f}% superior.")
        print("   → SIGMA não oferece vantagem neste cenário.")
    
    # Comparação SIGMA Puro vs Adam
    print("\n" + "-"*80)
    print("SIGMA-M Puro vs Adam:")
    print("-"*80)
    
    acc_sigma_pure = results_nn['SIGMA-M (Puro)']['test_acc'][-1]
    diff_sigma_adam = acc_sigma_pure - baseline_acc
    
    print(f"Acurácia SIGMA-M (Puro): {acc_sigma_pure:.2f}%")
    print(f"Acurácia Adam (Baseline): {baseline_acc:.2f}%")
    print(f"Diferença: {diff_sigma_adam:+.2f}%")
    
    if diff_sigma_adam > 0.1:
        print(f"\n✓ SIGMA-M é {diff_sigma_adam:.2f}% superior ao Adam!")
    elif diff_sigma_adam < -0.1:
        print(f"\n✗ Adam é {abs(diff_sigma_adam):.2f}% superior ao SIGMA-M.")
    else:
        print("\n≈ Desempenho comparável entre SIGMA-M e Adam.")
    
    # ========================================================================
    # RESUMO ESTATÍSTICO - REGRESSÃO LOGÍSTICA
    # ========================================================================
    
    print("\n" + "="*80)
    print("RESUMO FINAL - REGRESSÃO LOGÍSTICA (PROBLEMA CONVEXO)")
    print("="*80)
    
    print(f"\n{'Otimizador':<20} | {'Acc Final':<10} | {'Loss Final':<11} | {'Tempo (s)':<10}")
    print("-" * 65)
    
    for name in results_lr.keys():
        acc_final = results_lr[name]['test_acc'][-1]
        loss_final = results_lr[name]['train_loss'][-1]
        time_final = times_lr[name]
        print(f"{name:<20} | {acc_final:>9.2f}% | {loss_final:>10.6f} | {time_final:>9.2f}s")
    
    print("\n" + "-"*80)
    print("ANÁLISE (Regressão Logística):")
    print("-"*80)
    
    acc_lr_adam = results_lr['Adam']['test_acc'][-1]
    acc_lr_sigma = results_lr['SIGMA-M']['test_acc'][-1]
    acc_lr_sgd = results_lr['SGD+Momentum']['test_acc'][-1]
    
    print(f"\nMelhor Acurácia: {max(acc_lr_adam, acc_lr_sigma, acc_lr_sgd):.2f}%")
    
    best_lr = max(results_lr.items(), key=lambda x: x[1]['test_acc'][-1])
    print(f"Melhor Otimizador: {best_lr[0]}")
    
    print("\nInterpretação:")
    print("Em problemas convexos (regressão logística), todos os métodos")
    print("convergem para o ótimo global. Diferenças surgem em:")
    print("  - Velocidade de convergência inicial")
    print("  - Estabilidade durante o treinamento")
    print("  - Eficiência computacional")
    
    print("\n" + "="*80)
    print("Benchmark concluído com sucesso!")
    print("="*80)


if __name__ == "__main__":
    main()