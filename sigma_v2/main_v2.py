"""
Benchmark Comparativo de Híbridos Sequenciais v2
======================================================
Script Principal para Execução (Versão 2)

Este script importa:
- Otimizadores de 'optimizers_v2.py'
- Modelos de 'models.py'
- Funções de utilidade de 'utils_v2.py'
- Funções de plotagem de 'plotting_v2.py'

Todos os otimizadores agora usam weight_decay para uma comparação
mais justa (AdamW, SGDW, SIGMA-D_v2, SIGMA-C_v2).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Importações dos arquivos locais v2
from models import MNISTNet, LogisticRegression
from optimizers_v2 import SIGMA_D_v2, SIGMA_C_v2
from utils_v2 import get_data_loaders, run_experiment
from plotting_v2 import generate_nn_plots, generate_lr_plots

def main():
    """Executa todos os experimentos v2 e gera análises comparativas."""
    
    # Configuração
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {DEVICE}")
    
    # Configurações de Épocas da Rede Neural
    N_EPOCHS_NN_TOTAL = 20
    N_EPOCHS_NN_PHASE1 = 10
    N_EPOCHS_NN_PHASE2 = N_EPOCHS_NN_TOTAL - N_EPOCHS_NN_PHASE1
    
    # Configurações de Épocas da Regressão Logística
    N_EPOCHS_LR_TOTAL = 30
    N_EPOCHS_LR_PHASE1 = 15
    N_EPOCHS_LR_PHASE2 = N_EPOCHS_LR_TOTAL - N_EPOCHS_LR_PHASE1
    
    # Parâmetros de LR
    LR_ADAM = 0.001
    LR_SGD = 0.01
    LR_SIGMA = 0.01
    
    # *** Weight decay padrão para todos os otimizadores ***
    WD = 0.01 
    
    # Dados
    train_loader, test_loader = get_data_loaders()
    loss_fn = nn.CrossEntropyLoss()
    
    # =======================================================================
    # PARTE 1: EXPERIMENTOS COM REDES NEURAIS
    # =======================================================================
    
    print("\n" + "="*80)
    print("PARTE 1: EXPERIMENTOS v2 COM REDES NEURAIS (HÍBRIDOS SEQUENCIAIS)")
    print("="*80)
    
    base_model = MNISTNet().to(DEVICE)
    
    results_nn = {}
    times_nn = {}
    
    # --- EXPERIMENTO 1: Adam (Baseline) ---
    model_adam = copy.deepcopy(base_model)
    # *** ATUALIZADO para AdamW ***
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=LR_ADAM, weight_decay=WD)
    
    history_adam, time_adam = run_experiment(
        experiment_name="Adam (Baseline)",
        model=model_adam,
        optimizer_config=[(optimizer_adam, N_EPOCHS_NN_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Adam (Baseline)'] = history_adam
    times_nn['Adam (Baseline)'] = time_adam
    
    # --- EXPERIMENTO 2: Híbrido (Adam → SGD+M) - GRUPO DE CONTROLE ---
    model_hybrid_sgd = copy.deepcopy(base_model)
    # *** ATUALIZADO para AdamW / SGDW ***
    optimizer_adam_phase1_ctrl = optim.Adam(model_hybrid_sgd.parameters(), lr=LR_ADAM, weight_decay=WD)
    optimizer_sgd_phase2 = optim.SGD(model_hybrid_sgd.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD)
    
    history_hybrid_sgd, time_hybrid_sgd = run_experiment(
        experiment_name="Adam -> SGD+M",
        model=model_hybrid_sgd,
        optimizer_config=[
            (optimizer_adam_phase1_ctrl, N_EPOCHS_NN_PHASE1),
            (optimizer_sgd_phase2, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Adam -> SGD+M'] = history_hybrid_sgd
    times_nn['Adam -> SGD+M'] = time_hybrid_sgd

    # --- EXPERIMENTO 3: Híbrido (Adam → SIGMA-D_v2) ---
    model_hybrid_sigma_d = copy.deepcopy(base_model)
    # *** ATUALIZADO para AdamW / SIGMA_D_v2 ***
    optimizer_adam_phase1_d = optim.Adam(model_hybrid_sigma_d.parameters(), lr=LR_ADAM, weight_decay=WD)
    optimizer_sigma_d = SIGMA_D_v2(
        model_hybrid_sigma_d.parameters(),
        lr=LR_SIGMA,
        beta=0.9,
        alpha_min=0.1,
        alpha_max=2.0,
        weight_decay=WD # Adicionado
    )
    
    history_hybrid_sigma_d, time_hybrid_sigma_d = run_experiment(
        experiment_name="Adam -> SIGMA-D_v2",
        model=model_hybrid_sigma_d,
        optimizer_config=[
            (optimizer_adam_phase1_d, N_EPOCHS_NN_PHASE1),
            (optimizer_sigma_d, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Adam -> SIGMA-D_v2'] = history_hybrid_sigma_d
    times_nn['Adam -> SIGMA-D_v2'] = time_hybrid_sigma_d

    # --- EXPERIMENTO 4: Híbrido (Adam → SIGMA-C_v2) ---
    model_hybrid_sigma_c = copy.deepcopy(base_model)
    # *** ATUALIZADO para AdamW / SIGMA_C_v2 ***
    optimizer_adam_phase1_c = optim.Adam(model_hybrid_sigma_c.parameters(), lr=LR_ADAM, weight_decay=WD)
    optimizer_sigma_c = SIGMA_C_v2(
        model_hybrid_sigma_c.parameters(),
        lr=LR_SIGMA,
        beta=0.9,
        alpha_min=0.1,
        alpha_max=2.0,
        weight_decay=WD # Adicionado
    )
    
    history_hybrid_sigma_c, time_hybrid_sigma_c = run_experiment(
        experiment_name="Adam -> SIGMA-C_v2",
        model=model_hybrid_sigma_c,
        optimizer_config=[
            (optimizer_adam_phase1_c, N_EPOCHS_NN_PHASE1),
            (optimizer_sigma_c, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Adam -> SIGMA-C_v2'] = history_hybrid_sigma_c
    times_nn['Adam -> SIGMA-C_v2'] = time_hybrid_sigma_c
    
    
    # =======================================================================
    # PARTE 2: EXPERIMENTOS COM REGRESSÃO LOGÍSTICA
    # =======================================================================
    
    print("\n" + "="*80)
    print("PARTE 2: EXPERIMENTOS v2 COM REGRESSÃO LOGÍSTICA (PROBLEMA CONVEXO)")
    print("="*80)
    
    base_logistic = LogisticRegression().to(DEVICE)
    results_lr = {}
    times_lr = {}
    
    # --- Experimento LR-Puro 1: Adam ---
    model_lr_adam = copy.deepcopy(base_logistic)
    optimizer_lr_adam = optim.Adam(model_lr_adam.parameters(), lr=LR_ADAM, weight_decay=WD)
    history_lr_adam, time_lr_adam = run_experiment(
        experiment_name="[LR] Adam (Puro)",
        model=model_lr_adam,
        optimizer_config=[(optimizer_lr_adam, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Adam (Puro)'] = history_lr_adam
    times_lr['Adam (Puro)'] = time_lr_adam
    
    # --- Experimento LR-Puro 2: SGD+Momentum ---
    model_lr_sgd = copy.deepcopy(base_logistic)
    optimizer_lr_sgd = optim.SGD(model_lr_sgd.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD)
    history_lr_sgd, time_lr_sgd = run_experiment(
        experiment_name="[LR] SGD+M (Puro)",
        model=model_lr_sgd,
        optimizer_config=[(optimizer_lr_sgd, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['SGD+M (Puro)'] = history_lr_sgd
    times_lr['SGD+M (Puro)'] = time_lr_sgd
    
    # --- Experimento LR-Puro 3: SIGMA-D_v2 ---
    model_lr_sigma_d = copy.deepcopy(base_logistic)
    optimizer_lr_sigma_d = SIGMA_D_v2(
        model_lr_sigma_d.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD
    )
    history_lr_sigma_d, time_lr_sigma_d = run_experiment(
        experiment_name="[LR] SIGMA-D_v2 (Puro)",
        model=model_lr_sigma_d,
        optimizer_config=[(optimizer_lr_sigma_d, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['SIGMA-D_v2 (Puro)'] = history_lr_sigma_d
    times_lr['SIGMA-D_v2 (Puro)'] = time_lr_sigma_d

    # --- Experimento LR-Puro 4: SIGMA-C_v2 ---
    model_lr_sigma_c = copy.deepcopy(base_logistic)
    optimizer_lr_sigma_c = SIGMA_C_v2(
        model_lr_sigma_c.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD
    )
    history_lr_sigma_c, time_lr_sigma_c = run_experiment(
        experiment_name="[LR] SIGMA-C_v2 (Puro)",
        model=model_lr_sigma_c,
        optimizer_config=[(optimizer_lr_sigma_c, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['SIGMA-C_v2 (Puro)'] = history_lr_sigma_c
    times_lr['SIGMA-C_v2 (Puro)'] = time_lr_sigma_c
    
    # --- Experimento LR-Híbrido 1: Adam -> SGD+M ---
    model_lr_h_sgd = copy.deepcopy(base_logistic)
    optimizer_lr_h_adam = optim.Adam(model_lr_h_sgd.parameters(), lr=LR_ADAM, weight_decay=WD)
    optimizer_lr_h_sgd = optim.SGD(model_lr_h_sgd.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD)
    
    history_lr_h_sgd, time_lr_h_sgd = run_experiment(
        experiment_name="[LR] Adam -> SGD+M",
        model=model_lr_h_sgd,
        optimizer_config=[
            (optimizer_lr_h_adam, N_EPOCHS_LR_PHASE1),
            (optimizer_lr_h_sgd, N_EPOCHS_LR_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Adam -> SGD+M'] = history_lr_h_sgd
    times_lr['Adam -> SGD+M'] = time_lr_h_sgd

    # --- Experimento LR-Híbrido 2: Adam -> SIGMA-D_v2 ---
    model_lr_h_sigmad = copy.deepcopy(base_logistic)
    optimizer_lr_h_adam_d = optim.Adam(model_lr_h_sigmad.parameters(), lr=LR_ADAM, weight_decay=WD)
    optimizer_lr_h_sigmad = SIGMA_D_v2(
        model_lr_h_sigmad.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD
    )
    
    history_lr_h_sigmad, time_lr_h_sigmad = run_experiment(
        experiment_name="[LR] Adam -> SIGMA-D_v2",
        model=model_lr_h_sigmad,
        optimizer_config=[
            (optimizer_lr_h_adam_d, N_EPOCHS_LR_PHASE1),
            (optimizer_lr_h_sigmad, N_EPOCHS_LR_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Adam -> SIGMA-D_v2'] = history_lr_h_sigmad
    times_lr['Adam -> SIGMA-D_v2'] = time_lr_h_sigmad

    # --- Experimento LR-Híbrido 3: Adam -> SIGMA-C_v2 ---
    model_lr_h_sigmac = copy.deepcopy(base_logistic)
    optimizer_lr_h_adam_c = optim.Adam(model_lr_h_sigmac.parameters(), lr=LR_ADAM, weight_decay=WD)
    optimizer_lr_h_sigmac = SIGMA_C_v2(
        model_lr_h_sigmac.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD
    )
    
    history_lr_h_sigmac, time_lr_h_sigmac = run_experiment(
        experiment_name="[LR] Adam -> SIGMA-C_v2",
        model=model_lr_h_sigmac,
        optimizer_config=[
            (optimizer_lr_h_adam_c, N_EPOCHS_LR_PHASE1),
            (optimizer_lr_h_sigmac, N_EPOCHS_LR_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Adam -> SIGMA-C_v2'] = history_lr_h_sigmac
    times_lr['Adam -> SIGMA-C_v2'] = time_lr_h_sigmac
    
    # ========================================================================
    # GERAÇÃO DE PLOTS E RESUMOS
    # ========================================================================
    
    # Gerar gráficos .pdf
    generate_nn_plots(results_nn, times_nn, N_EPOCHS_NN_PHASE1)
    generate_lr_plots(results_lr, times_lr, N_EPOCHS_LR_PHASE1)
    
    # Imprimir resumos estatísticos no console
    
    print("\n" + "="*80)
    print("RESUMO FINAL - REDES NEURAIS (v2)")
    print("="*80)
    
    print(f"\n{'Experimento':<25} | {'Acc Final':<10} | {'Loss Final':<11} | {'Tempo (s)':<10}")
    print("-" * 65)
    
    for name in results_nn.keys():
        acc_final = results_nn[name]['test_acc'][-1]
        loss_final = results_nn[name]['test_loss'][-1]
        time_final = times_nn[name]
        print(f"{name:<25} | {acc_final:>9.2f}% | {loss_final:>10.4f} | {time_final:>9.2f}s")
    
    print("\n" + "-"*80)
    print("ANÁLISE DE MELHORIAS (v2) (vs Híbrido SGD)")
    print("-"*80)
    
    baseline_acc_nn = results_nn['Adam -> SGD+M']['test_acc'][-1]
    acc_d_nn = results_nn['Adam -> SIGMA-D_v2']['test_acc'][-1]
    acc_c_nn = results_nn['Adam -> SIGMA-C_v2']['test_acc'][-1]
    
    diff_d_nn = acc_d_nn - baseline_acc_nn
    diff_c_nn = acc_c_nn - baseline_acc_nn
    
    print(f"Híbrido (Adam -> SIGMA-D_v2) vs (Adam -> SGD+M): {diff_d_nn:+.2f}% {'✓' if diff_d_nn > 0 else '✗'}")
    print(f"Híbrido (Adam -> SIGMA-C_v2) vs (Adam -> SGD+M): {diff_c_nn:+.2f}% {'✓' if diff_c_nn > 0 else '✗'}")

    
    print("\n" + "="*80)
    print("RESUMO FINAL - REGRESSÃO LOGÍSTICA (v2) (PROBLEMA CONVEXO)")
    print("="*80)
    
    print(f"\n{'Otimizador':<20} | {'Acc Final':<10} | {'Loss Final':<11} | {'Tempo (s)':<10}")
    print("-" * 65)
    
    for name in results_lr.keys():
        acc_final = results_lr[name]['test_acc'][-1]
        loss_final = results_lr[name]['train_loss'][-1]
        time_final = times_lr[name]
        print(f"{name:<20} | {acc_final:>9.2f}% | {loss_final:>10.6f} | {time_final:>9.2f}s")

    print("\n" + "-"*80)
    print("ANÁLISE (v2) (Regressão Logística)")
    print("-"*80)
    
    baseline_acc_lr = results_lr['Adam -> SGD+M']['test_acc'][-1]
    acc_d_lr = results_lr['Adam -> SIGMA-D_v2']['test_acc'][-1]
    acc_c_lr = results_lr['Adam -> SIGMA-C_v2']['test_acc'][-1]
    
    diff_d_lr = acc_d_lr - baseline_acc_lr
    diff_c_lr = acc_c_lr - baseline_acc_lr
    
    print("Comparação Híbridos (vs Híbrido SGD):")
    print(f"Híbrido (Adam -> SIGMA-D_v2) vs (Adam -> SGD+M): {diff_d_lr:+.2f}% {'✓' if diff_d_lr > 0 else '✗'}")
    print(f"Híbrido (Adam -> SIGMA-C_v2) vs (Adam -> SGD+M): {diff_c_lr:+.2f}% {'✓' if diff_c_lr > 0 else '✗'}")

    print("\n" + "="*80)
    print("Benchmark v2 concluído com sucesso!")
    print("="*80)


if __name__ == "__main__":
    main()