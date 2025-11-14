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
    
    print("\nGerando gráficos de comparação (RedES Neurais - v3)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # *** ATUALIZADO: Cores e Marcadores para 10 experimentos ***
    colors_nn = {
        'Adam (Puro)': '#1f77b4',
        'SGD+M (Puro)': '#ff7f0e',
        'SIGMA-D_v2 (Puro)': '#2ca02c',
        'SIGMA-C_v2 (Puro)': '#9467bd',
        'Adam -> SGD+M': '#d62728',
        'Adam -> SIGMA-D_v2': '#8c564b',
        'Adam -> SIGMA-C_v2': '#e377c2',
        'Cíclico (A->S)x2': '#7f7f7f',
        'Cíclico (A->D)x2': '#bcbd22',
        'Cíclico (A->C)x2': '#17becf'
    }
    
    markers_nn = { k: '.' for k in colors_nn.keys() }
    
    # --- Plot 1: Acurácia no Teste ---
    ax1 = axes[0, 0]
    for name, history in results_nn.items():
        ax1.plot(history['test_acc'], 
                label=name.replace('[NN] ', ''), 
                color=colors_nn.get(name.replace('[NN] ', ''), 'black'),
                marker=markers_nn.get(name.replace('[NN] ', ''), '.'), 
                markersize=4,
                linewidth=2)
    
    ax1.axvline(x=n_epochs_phase1 - 0.5, color='gray', linestyle=':', 
                linewidth=1.5, alpha=0.7, label='Troca (Híbridos 10/10)')
    for i in [5, 10, 15]:
         ax1.axvline(x=i - 0.5, color='blue', linestyle='--', 
                     linewidth=1.0, alpha=0.5, label='Troca (Cíclicos 5/5)')
    
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Acurácia no Teste (%)', fontsize=12)
    ax1.set_title('Comparação de Acurácia: Híbridos v3 vs Baseline (Redes Neurais)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Loss no Teste (escala log) ---
    ax2 = axes[0, 1]
    for name, history in results_nn.items():
        ax2.plot(history['test_loss'], 
                label=name.replace('[NN] ', ''),
                color=colors_nn.get(name.replace('[NN] ', ''), 'black'),
                marker=markers_nn.get(name.replace('[NN] ', ''), '.'), 
                markersize=4,
                linewidth=2)
    
    ax2.axvline(x=n_epochs_phase1 - 0.5, color='gray', linestyle=':', 
                linewidth=1.5, alpha=0.7)
    for i in [5, 10, 15]:
         ax2.axvline(x=i - 0.5, color='blue', linestyle='--', 
                     linewidth=1.0, alpha=0.5)
                     
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Loss no Teste (escala log)', fontsize=12)
    ax2.set_title('Comparação de Loss v3 (Redes Neurais)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Loss de Treino ---
    ax3 = axes[1, 0]
    for name, history in results_nn.items():
        ax3.plot(history['train_loss'], 
                label=name.replace('[NN] ', ''),
                color=colors_nn.get(name.replace('[NN] ', ''), 'black'),
                marker=markers_nn.get(name.replace('[NN] ', ''), '.'), 
                markersize=3,
                linewidth=2,
                alpha=0.8)
    
    ax3.axvline(x=n_epochs_phase1 - 0.5, color='gray', linestyle=':', 
                linewidth=1.5, alpha=0.7)
    for i in [5, 10, 15]:
         ax3.axvline(x=i - 0.5, color='blue', linestyle='--', 
                     linewidth=1.0, alpha=0.5)
                     
    ax3.set_xlabel('Época', fontsize=12)
    ax3.set_ylabel('Loss de Treino', fontsize=12)
    ax3.set_title('Convergência no Treino v3 (Redes Neurais)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Comparação de Tempo de Treinamento (Barras) ---
    ax4 = axes[1, 1]
    # Filtra para não poluir o gráfico de tempo
    filtered_times = {k: v for k, v in times_nn.items() if 'Puro' in k or 'Adam ->' in k}
    names_nn = [n.replace('[NN] ', '') for n in filtered_times.keys()]
    time_values_nn = list(filtered_times.values())
    bar_colors = [colors_nn.get(n, 'gray') for n in names_nn]
    
    bars = ax4.barh(names_nn, time_values_nn, color=bar_colors)
    
    for i, (bar, val) in enumerate(zip(bars, time_values_nn)):
        ax4.text(val + 1, i, f'{val:.1f}s', va='center', fontsize=10)
    
    ax4.set_xlabel('Tempo de Treinamento (s)', fontsize=12)
    ax4.set_title('Eficiência Computacional v3 (Redes Neurais)', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sigma_hybrid_comparison_nn_v3.pdf', dpi=150, bbox_inches='tight')
    print("Gráfico salvo: 'sigma_hybrid_comparison_nn_v3.pdf'")

# ========================================================================
# VISUALIZAÇÃO - REGRESSÃO LOGÍSTICA
# ========================================================================

def generate_lr_plots(results_lr, times_lr, n_epochs_phase1):
    """
    Gera e salva o gráfico 2x2 para os resultados da Regressão Logística.
    Salva como 'sigma_full_comparison_logistic_v3.pdf'.
    """
    
    print("\nGerando gráficos de comparação (Regressão Logística - v3)...")
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12)) 
    
    colors_lr = {
        'Adam (Puro)': '#1f77b4',
        'SGD+M (Puro)': '#ff7f0e',
        'SIGMA-D_v2 (Puro)': '#2ca02c',
        'SIGMA-C_v2 (Puro)': '#9467bd',
        'Adam -> SGD+M': '#d62728',
        'Adam -> SIGMA-D_v2': '#8c564b',
        'Adam -> SIGMA-C_v2': '#e377c2',
        'Cíclico (A->S)x2': '#7f7f7f',
        'Cíclico (A->D)x2': '#bcbd22',
        'Cíclico (A->C)x2': '#17becf'
    }
    
    markers_lr = { k: '.' for k in colors_lr.keys() }
    linestyles_lr = { k: '-' for k in colors_lr.keys() }
    
    # --- Plot 1: Acurácia (axes2[0, 0]) ---
    ax1_lr = axes2[0, 0]
    for name, history in results_lr.items():
        ax1_lr.plot(history['test_acc'], 
                   label=name.replace('[LR] ', ''), 
                   color=colors_lr.get(name.replace('[LR] ', ''), 'black'),
                   linestyle=linestyles_lr.get(name.replace('[LR] ', ''), '-'),
                   marker=markers_lr.get(name.replace('[LR] ', ''), '.'), 
                   markersize=4,
                   linewidth=2)
    
    ax1_lr.axvline(x=n_epochs_phase1 - 0.5, color='gray', linestyle=':', 
                linewidth=1.5, alpha=0.7, label='Troca (Híbridos 15/15)')
    for i in [7, 15, 22]: # 7, 7+8, 7+8+7
         ax1_lr.axvline(x=i - 0.5, color='blue', linestyle='--', 
                     linewidth=1.0, alpha=0.5, label='Troca (Cíclicos 7/8)')

    ax1_lr.set_xlabel('Época', fontsize=12)
    ax1_lr.set_ylabel('Acurácia no Teste (%)', fontsize=12)
    ax1_lr.set_title('Regressão Logística v3: Acurácia (Problema Convexo)', fontsize=14, fontweight='bold')
    ax1_lr.legend(fontsize=9, loc='lower right')
    ax1_lr.grid(True, alpha=0.3)
    
    # --- Plot 2: Loss de Teste (axes2[0, 1]) ---
    ax2_lr = axes2[0, 1]
    for name, history in results_lr.items():
        ax2_lr.plot(history['test_loss'], 
                   label=name.replace('[LR] ', ''),
                   color=colors_lr.get(name.replace('[LR] ', ''), 'black'),
                   linestyle=linestyles_lr.get(name.replace('[LR] ', ''), '-'),
                   marker=markers_lr.get(name.replace('[LR] ', ''), '.'), 
                   markersize=4,
                   linewidth=2)
    
    ax2_lr.axvline(x=n_epochs_phase1 - 0.5, color='gray', linestyle=':', 
                linewidth=1.5, alpha=0.7)
    for i in [7, 15, 22]:
         ax2_lr.axvline(x=i - 0.5, color='blue', linestyle='--', 
                     linewidth=1.0, alpha=0.5)

    ax2_lr.set_xlabel('Época', fontsize=12)
    ax2_lr.set_ylabel('Loss de Teste (escala log)', fontsize=12)
    ax2_lr.set_title('Regressão Logística v3: Loss de Teste', fontsize=14, fontweight='bold')
    ax2_lr.set_yscale('log')
    ax2_lr.legend(fontsize=9, loc='upper right')
    ax2_lr.grid(True, alpha=0.3)

    # --- Plot 3: Loss de Treino (axes2[1, 0]) ---
    ax3_lr = axes2[1, 0]
    for name, history in results_lr.items():
        ax3_lr.plot(history['train_loss'], 
                   label=name.replace('[LR] ', ''),
                   color=colors_lr.get(name.replace('[LR] ', ''), 'black'),
                   linestyle=linestyles_lr.get(name.replace('[LR] ', ''), '-'),
                   marker=markers_lr.get(name.replace('[LR] ', ''), '.'), 
                   markersize=4,
                   linewidth=2)
    
    ax3_lr.axvline(x=n_epochs_phase1 - 0.5, color='gray', linestyle=':', 
                linewidth=1.5, alpha=0.7)
    for i in [7, 15, 22]:
         ax3_lr.axvline(x=i - 0.5, color='blue', linestyle='--', 
                     linewidth=1.0, alpha=0.5)
                     
    ax3_lr.set_xlabel('Época', fontsize=12)
    ax3_lr.set_ylabel('Loss de Treino (escala log)', fontsize=12)
    ax3_lr.set_title('Regressão Logística v3: Convergência', fontsize=14, fontweight='bold')
    ax3_lr.set_yscale('log')
    ax3_lr.legend(fontsize=9, loc='upper right')
    ax3_lr.grid(True, alpha=0.3)

    # --- Plot 4: Tempo (axes2[1, 1]) ---
    ax4_lr = axes2[1, 1]
    filtered_times_lr = {k: v for k, v in times_lr.items() if 'Puro' in k or 'Adam ->' in k}
    names_lr = [n.replace('[LR] ', '') for n in filtered_times_lr.keys()]
    time_values_lr = list(filtered_times_lr.values())
    bar_colors_lr = [colors_lr.get(n, 'gray') for n in names_lr]
    
    bars = ax4_lr.barh(names_lr, time_values_lr, color=bar_colors_lr)
    
    for i, (bar, val) in enumerate(zip(bars, time_values_lr)):
        ax4_lr.text(val + 1, i, f'{val:.1f}s', va='center', fontsize=10)
    
    ax4_lr.set_xlabel('Tempo de Treinamento (s)', fontsize=12)
    ax4_lr.set_title('Eficiência Computacional v3 (Regressão Logística)', fontsize=14, fontweight='bold')
    ax4_lr.grid(axis='x', alpha=0.3)

    
    plt.tight_layout()
    plt.savefig('sigma_full_comparison_logistic_v3.pdf', dpi=150, bbox_inches='tight')
    print("Gráfico salvo: 'sigma_full_comparison_logistic_v3.pdf'")


# ========================================================================
# [NOVO] VISUALIZAÇÃO DE RESUMO - PLOT 3
# ========================================================================

def generate_combined_plot(results_nn, times_nn, results_lr, times_lr, 
                           nn_phase1, lr_phase1):
    """
    Gera um gráfico 2x3 comparando os 7 otimizadores-chave em
    ambos os cenários (NN e LR).
    Salva como 'sigma_summary_comparison_v3.pdf'.
    """
    
    print("\nGerando gráfico de resumo (Plot 3 - v3)...")
    
    # 1. Definir os 7 métodos para plotar e seus novos rótulos
    
    # Chaves originais nos dicionários de resultados
    keys_to_plot_nn = {
        'Adam (Puro)': 'Adam (Puro)',
        'Cíclico (A->S)x2': 'Adam->SGD (Cíclico)',
        'Cíclico (A->C)x2': 'Adam->SIGMA-C (Cíclico)',
        'Cíclico (A->D)x2': 'Adam->SIGMA-D (Cíclico)',
        'Adam -> SIGMA-C_v2': 'Adam->SIGMA-C (Híbrido)',
        'Adam -> SIGMA-D_v2': 'Adam->SIGMA-D (Híbrido)',
        'SIGMA-C_v2 (Puro)': 'SIGMA-C (Puro)'
    }
    
    # O set de chaves é o mesmo para LR
    keys_to_plot_lr = {
        'Adam (Puro)': 'Adam (Puro)',
        'Cíclico (A->S)x2': 'Adam->SGD (Cíclico)',
        'Cíclico (A->C)x2': 'Adam->SIGMA-C (Cíclico)',
        'Cíclico (A->D)x2': 'Adam->SIGMA-D (Cíclico)',
        'Adam -> SIGMA-C_v2': 'Adam->SIGMA-C (Híbrido)',
        'Adam -> SIGMA-D_v2': 'Adam->SIGMA-D (Híbrido)',
        'SIGMA-C_v2 (Puro)': 'SIGMA-C (Puro)'
    }
    
    # Paleta de cores para os 7 métodos
    colors = {
        'Adam (Puro)': '#1f77b4',             # Azul
        'Adam->SGD (Cíclico)': '#ff7f0e',     # Laranja
        'Adam->SIGMA-C (Cíclico)': '#9467bd', # Roxo
        'Adam->SIGMA-D (Cíclico)': '#2ca02c', # Verde
        'Adam->SIGMA-C (Híbrido)': '#e377c2', # Rosa
        'Adam->SIGMA-D (Híbrido)': '#8c564b', # Marrom
        'SIGMA-C (Puro)': '#d62728'          # Vermelho
    }
    
    markers = { k: 'o' for k in colors.keys() }
    
    # 2. Criar a figura 2x3
    fig, axes = plt.subplots(2, 3, figsize=(26, 15))
    fig.suptitle('Resumo Comparativo: Otimizadores Híbridos vs Cíclicos vs Puros (v3)', 
                 fontsize=20, fontweight='bold')
    
    # 3. --- LINHA 1: REDES NEURAIS (NÃO-CONVEXO) ---
    
    ax_nn_acc = axes[0, 0]
    ax_nn_loss = axes[0, 1]
    ax_nn_time = axes[0, 2]
    
    ax_nn_acc.set_title('Rede Neural: Acurácia no Teste', fontsize=16)
    ax_nn_loss.set_title('Rede Neural: Loss no Teste (log)', fontsize=16)
    ax_nn_time.set_title('Rede Neural: Tempo de Treinamento', fontsize=16)
    
    plot_times_nn = {}
    
    # Plot de Acurácia e Loss (NN)
    for orig_key, new_label in keys_to_plot_nn.items():
        if orig_key not in results_nn: continue
        history = results_nn[orig_key]
        c = colors.get(new_label, 'black')
        m = markers.get(new_label, '.')
        
        ax_nn_acc.plot(history['test_acc'], label=new_label, color=c, marker=m, markersize=4, alpha=0.8)
        ax_nn_loss.plot(history['test_loss'], label=new_label, color=c, marker=m, markersize=4, alpha=0.8)
        
        plot_times_nn[new_label] = times_nn[orig_key]

    ax_nn_acc.axvline(x=nn_phase1 - 0.5, color='gray', linestyle=':', label='Troca Híbrida (10)')
    ax_nn_acc.grid(True, alpha=0.3)
    ax_nn_acc.set_xlabel('Época', fontsize=12)
    ax_nn_acc.set_ylabel('Acurácia (%)', fontsize=12)

    ax_nn_loss.axvline(x=nn_phase1 - 0.5, color='gray', linestyle=':', label='Troca Híbrida (10)')
    ax_nn_loss.set_yscale('log')
    ax_nn_loss.grid(True, alpha=0.3)
    ax_nn_loss.set_xlabel('Época', fontsize=12)
    ax_nn_loss.set_ylabel('Loss (log)', fontsize=12)

    # Plot de Tempo (NN)
    labels = list(plot_times_nn.keys())
    values = list(plot_times_nn.values())
    bar_colors_nn = [colors.get(l, 'gray') for l in labels]
    
    bars_nn = ax_nn_time.barh(labels, values, color=bar_colors_nn)
    ax_nn_time.set_xlabel('Tempo (s)', fontsize=12)
    ax_nn_time.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars_nn, values):
        ax_nn_time.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}s', va='center', fontsize=10)

    # 4. --- LINHA 2: REGRESSÃO LOGÍSTICA (CONVEXO) ---
    
    ax_lr_acc = axes[1, 0]
    ax_lr_loss = axes[1, 1]
    ax_lr_time = axes[1, 2]
    
    ax_lr_acc.set_title('Reg. Logística: Acurácia no Teste', fontsize=16)
    ax_lr_loss.set_title('Reg. Logística: Loss no Teste (log)', fontsize=16)
    ax_lr_time.set_title('Reg. Logística: Tempo de Treinamento', fontsize=16)
    
    plot_times_lr = {}
    
    # Plot de Acurácia e Loss (LR)
    for orig_key, new_label in keys_to_plot_lr.items():
        if orig_key not in results_lr: continue
        history = results_lr[orig_key]
        c = colors.get(new_label, 'black')
        m = markers.get(new_label, '.')
        
        ax_lr_acc.plot(history['test_acc'], label=new_label, color=c, marker=m, markersize=4, alpha=0.8)
        ax_lr_loss.plot(history['test_loss'], label=new_label, color=c, marker=m, markersize=4, alpha=0.8)
        
        plot_times_lr[new_label] = times_lr[orig_key]

    ax_lr_acc.axvline(x=lr_phase1 - 0.5, color='gray', linestyle=':', label='Troca Híbrida (15)')
    ax_lr_acc.grid(True, alpha=0.3)
    ax_lr_acc.set_xlabel('Época', fontsize=12)
    ax_lr_acc.set_ylabel('Acurácia (%)', fontsize=12)

    ax_lr_loss.axvline(x=lr_phase1 - 0.5, color='gray', linestyle=':', label='Troca Híbrida (15)')
    ax_lr_loss.set_yscale('log')
    ax_lr_loss.grid(True, alpha=0.3)
    ax_lr_loss.set_xlabel('Época', fontsize=12)
    ax_lr_loss.set_ylabel('Loss (log)', fontsize=12)

    # Plot de Tempo (LR)
    labels_lr = list(plot_times_lr.keys())
    values_lr = list(plot_times_lr.values())
    bar_colors_lr = [colors.get(l, 'gray') for l in labels_lr]
    
    bars_lr = ax_lr_time.barh(labels_lr, values_lr, color=bar_colors_lr)
    ax_lr_time.set_xlabel('Tempo (s)', fontsize=12)
    ax_lr_time.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars_lr, values_lr):
        ax_lr_time.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}s', va='center', fontsize=10)

    # 5. Legenda e Salvamento
    handles, labels = ax_nn_acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=14, bbox_to_anchor=(0.5, -0.01))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Ajustar para supertítulo e legenda
    plt.savefig('sigma_summary_comparison_v3.pdf', dpi=150, bbox_inches='tight')
    print("Gráfico de resumo salvo: 'sigma_summary_comparison_v3.pdf'")