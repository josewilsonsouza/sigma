import matplotlib.pyplot as plt
import numpy as np

# ========================================================================
# VISUALIZAÇÃO DOS RESULTADOS - REDES NEURAIS
# ========================================================================

def generate_nn_plots(results_nn, times_nn, n_epochs_phase1):
    """
    Gera e salva o gráfico 2x2 para os resultados da Rede Neural.
    Salva como 'sigma_hybrid_comparison_nn_v2.pdf'.
    """
    
    print("\nGerando gráficos de comparação (Redes Neurais - v2)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # *** ATUALIZADO: Adicionado novo experimento Cíclico ***
    colors_nn = {
        'Adam (Baseline)': '#1f77b4',
        'Adam -> SGD+M': '#d62728',
        'Adam -> SIGMA-D_v2': '#2ca02c',
        'Adam -> SIGMA-C_v2': '#9467bd',
        'Cíclico (A->C)x2': '#ff7f0e' # Nova cor para o cíclico
    }
    
    markers_nn = {
        'Adam (Baseline)': 'o',
        'Adam -> SGD+M': 'v',
        'Adam -> SIGMA-D_v2': '^',
        'Adam -> SIGMA-C_v2': 'P',
        'Cíclico (A->C)x2': 'X' # Novo marcador
    }
    
    # --- Plot 1: Acurácia no Teste ---
    ax1 = axes[0, 0]
    for name, history in results_nn.items():
        ax1.plot(history['test_acc'], 
                label=name, 
                color=colors_nn.get(name, 'black'), # .get() para segurança
                marker=markers_nn.get(name, '.'), 
                markersize=4,
                linewidth=2)
    
    # *** ATUALIZADO: Lógica para desenhar linhas de troca ***
    ax1.axvline(x=n_epochs_phase1 - 0.5, color='gray', linestyle=':', 
                linewidth=1.5, alpha=0.7, label='Troca (Híbridos 10/10)')
    # Linhas para o cíclico
    for i in [5, 10, 15]:
         ax1.axvline(x=i - 0.5, color='#ff7f0e', linestyle='--', 
                     linewidth=1.0, alpha=0.5)
    
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Acurácia no Teste (%)', fontsize=12)
    ax1.set_title('Comparação de Acurácia: Híbridos v2 vs Baseline (Redes Neurais)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Loss no Teste (escala log) ---
    ax2 = axes[0, 1]
    for name, history in results_nn.items():
        ax2.plot(history['test_loss'], 
                label=name,
                color=colors_nn.get(name, 'black'),
                marker=markers_nn.get(name, '.'), 
                markersize=4,
                linewidth=2)
    
    ax2.axvline(x=n_epochs_phase1 - 0.5, color='gray', linestyle=':', 
                linewidth=1.5, alpha=0.7, label='Troca (Híbridos 10/10)')
    for i in [5, 10, 15]:
         ax2.axvline(x=i - 0.5, color='#ff7f0e', linestyle='--', 
                     linewidth=1.0, alpha=0.5)
                     
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Loss no Teste (escala log)', fontsize=12)
    ax2.set_title('Comparação de Loss v2 (Redes Neurais)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Loss de Treino ---
    ax3 = axes[1, 0]
    for name, history in results_nn.items():
        ax3.plot(history['train_loss'], 
                label=name,
                color=colors_nn.get(name, 'black'),
                marker=markers_nn.get(name, '.'), 
                markersize=3,
                linewidth=2,
                alpha=0.8)
    
    ax3.axvline(x=n_epochs_phase1 - 0.5, color='gray', linestyle=':', 
                linewidth=1.5, alpha=0.7)
    for i in [5, 10, 15]:
         ax3.axvline(x=i - 0.5, color='#ff7f0e', linestyle='--', 
                     linewidth=1.0, alpha=0.5)
                     
    ax3.set_xlabel('Época', fontsize=12)
    ax3.set_ylabel('Loss de Treino', fontsize=12)
    ax3.set_title('Convergência no Treino v2 (Redes Neurais)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Comparação de Tempo de Treinamento (Barras) ---
    ax4 = axes[1, 1]
    names_nn = list(times_nn.keys())
    time_values_nn = list(times_nn.values())
    bar_colors = [colors_nn.get(n, 'gray') for n in names_nn]
    bars = ax4.barh(names_nn, time_values_nn, color=bar_colors)
    
    for i, (bar, val) in enumerate(zip(bars, time_values_nn)):
        ax4.text(val + 1, i, f'{val:.1f}s', va='center', fontsize=10)
    
    ax4.set_xlabel('Tempo de Treinamento (s)', fontsize=12)
    ax4.set_title('Eficiência Computacional v2 (Redes Neurais)', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sigma_hybrid_comparison_nn_v2.pdf', dpi=150, bbox_inches='tight')
    print("Gráfico salvo: 'sigma_hybrid_comparison_nn_v2.pdf'")

# ========================================================================
# VISUALIZAÇÃO - REGRESSÃO LOGÍSTICA
# ========================================================================

def generate_lr_plots(results_lr, times_lr, n_epochs_phase1):
    """
    Gera e salva o gráfico 2x2 para os resultados da Regressão Logística.
    Salva como 'sigma_full_comparison_logistic_v2.pdf'.
    (Este foi mantido como estava, sem o experimento cíclico, por clareza)
    """
    
    print("\nGerando gráficos de comparação (Regressão Logística - v2)...")
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12)) 
    
    colors_lr = {
        'Adam (Puro)': '#1f77b4',
        'SGD+M (Puro)': '#ff7f0e',
        'SIGMA-D_v2 (Puro)': '#2ca02c',
        'SIGMA-C_v2 (Puro)': '#9467bd',
        'Adam -> SGD+M': '#d62728',
        'Adam -> SIGMA-D_v2': '#8c564b',
        'Adam -> SIGMA-C_v2': '#e377c2',
    }
    
    markers_lr = { k: 'o' if 'Puro' in k else '^' for k in colors_lr.keys() }
    linestyles_lr = { k: ':' if 'Puro' in k else '-' for k in colors_lr.keys() }
    
    # --- Plot 1: Acurácia (axes2[0, 0]) ---
    ax1_lr = axes2[0, 0]
    for name, history in results_lr.items():
        ax1_lr.plot(history['test_acc'], 
                   label=name, 
                   color=colors_lr.get(name, 'black'),
                   linestyle=linestyles_lr.get(name, '-'),
                   marker=markers_lr.get(name, '.'), 
                   markersize=4,
                   linewidth=2)
    
    ax1_lr.axvline(x=n_epochs_phase1 - 0.5, color='gray', linestyle='--', 
                linewidth=1.5, alpha=0.7, label='Troca (Híbridos)')
    ax1_lr.set_xlabel('Época', fontsize=12)
    ax1_lr.set_ylabel('Acurácia no Teste (%)', fontsize=12)
    ax1_lr.set_title('Regressão Logística v2: Acurácia (Problema Convexo)', fontsize=14, fontweight='bold')
    ax1_lr.legend(fontsize=9, loc='lower right')
    ax1_lr.grid(True, alpha=0.3)
    
    # --- Plot 2: Loss de Teste (axes2[0, 1]) ---
    ax2_lr = axes2[0, 1]
    for name, history in results_lr.items():
        ax2_lr.plot(history['test_loss'], 
                   label=name,
                   color=colors_lr.get(name, 'black'),
                   linestyle=linestyles_lr.get(name, '-'),
                   marker=markers_lr.get(name, '.'), 
                   markersize=4,
                   linewidth=2)
    
    ax2_lr.axvline(x=n_epochs_phase1 - 0.5, color='gray', linestyle='--', 
                linewidth=1.5, alpha=0.7, label='Troca (Híbridos)')
    ax2_lr.set_xlabel('Época', fontsize=12)
    ax2_lr.set_ylabel('Loss de Teste (escala log)', fontsize=12)
    ax2_lr.set_title('Regressão Logística v2: Loss de Teste', fontsize=14, fontweight='bold')
    ax2_lr.set_yscale('log')
    ax2_lr.legend(fontsize=9, loc='upper right')
    ax2_lr.grid(True, alpha=0.3)

    # --- Plot 3: Loss de Treino (axes2[1, 0]) ---
    ax3_lr = axes2[1, 0]
    for name, history in results_lr.items():
        ax3_lr.plot(history['train_loss'], 
                   label=name,
                   color=colors_lr.get(name, 'black'),
                   linestyle=linestyles_lr.get(name, '-'),
                   marker=markers_lr.get(name, '.'), 
                   markersize=4,
                   linewidth=2)
    
    ax3_lr.axvline(x=n_epochs_phase1 - 0.5, color='gray', linestyle='--', 
                linewidth=1.5, alpha=0.7, label='Troca (Híbridos)')
    ax3_lr.set_xlabel('Época', fontsize=12)
    ax3_lr.set_ylabel('Loss de Treino (escala log)', fontsize=12)
    ax3_lr.set_title('Regressão Logística v2: Convergência', fontsize=14, fontweight='bold')
    ax3_lr.set_yscale('log')
    ax3_lr.legend(fontsize=9, loc='upper right')
    ax3_lr.grid(True, alpha=0.3)

    # --- Plot 4: Tempo (axes2[1, 1]) ---
    ax4_lr = axes2[1, 1]
    names_lr = list(times_lr.keys())
    time_values_lr = list(times_lr.values())
    bar_colors_lr = [colors_lr.get(n, 'gray') for n in names_lr]
    bars = ax4_lr.barh(names_lr, time_values_lr, color=bar_colors_lr)
    
    for i, (bar, val) in enumerate(zip(bars, time_values_lr)):
        ax4_lr.text(val + 1, i, f'{val:.1f}s', va='center', fontsize=10)
    
    ax4_lr.set_xlabel('Tempo de Treinamento (s)', fontsize=12)
    ax4_lr.set_title('Eficiência Computacional v2 (Regressão Logística)', fontsize=14, fontweight='bold')
    ax4_lr.grid(axis='x', alpha=0.3)

    
    plt.tight_layout()
    plt.savefig('sigma_full_comparison_logistic_v2.pdf', dpi=150, bbox_inches='tight')
    print("Gráfico salvo: 'sigma_full_comparison_logistic_v2.pdf'")