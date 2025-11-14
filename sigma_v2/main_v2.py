"""
Benchmark Comparativo v3 - Sincronizado
======================================================
Script Principal para Execução

Atualizações:
1. Experimentos de NN e LR estão idênticos (Puros, Híbridos, Cíclicos).
2. Gera o gráfico combinado específico (Resumo) no final.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Importações locais
from models import MNISTNet, LogisticRegression
from optimizers_v2 import SIGMA_D_v2, SIGMA_C_v2
from utils_v2 import get_data_loaders, run_experiment
# Importa a nova função de plotagem combinada
from plotting_v2 import generate_nn_plots, generate_lr_plots, generate_combined_plot

def main():
    # Configuração
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {DEVICE}")
    
    # Configurações de Épocas (NN: 20 total, LR: 30 total)
    N_EPOCHS_NN_TOTAL = 20
    N_EPOCHS_NN_PHASE1 = 10
    N_EPOCHS_NN_PHASE2 = 10
    
    N_EPOCHS_LR_TOTAL = 30
    N_EPOCHS_LR_PHASE1 = 15
    N_EPOCHS_LR_PHASE2 = 15
    
    # Hiperparâmetros
    LR_ADAM = 0.001
    LR_SGD = 0.01
    LR_SIGMA = 0.01
    WD = 0
    
    # Dados
    train_loader, test_loader = get_data_loaders()
    loss_fn = nn.CrossEntropyLoss()
    
    # =======================================================================
    # PARTE 1: EXPERIMENTOS COM REDES NEURAIS (NÃO-CONVEXO)
    # =======================================================================
    print("\n" + "="*80)
    print("PARTE 1: REDES NEURAIS (Conjunto Completo)")
    print("="*80)
    
    base_model = MNISTNet().to(DEVICE)
    results_nn = {}
    times_nn = {}
    
    # 1. Adam (Puro)
    m = copy.deepcopy(base_model)
    opt = optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD)
    results_nn['Adam (Puro)'], times_nn['Adam (Puro)'] = run_experiment(
        "[NN] Adam (Puro)", m, [(opt, N_EPOCHS_NN_TOTAL)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_NN_TOTAL
    )

    # 2. SGD+M (Puro)
    m = copy.deepcopy(base_model)
    opt = optim.SGD(m.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD)
    results_nn['SGD+M (Puro)'], times_nn['SGD+M (Puro)'] = run_experiment(
        "[NN] SGD+M (Puro)", m, [(opt, N_EPOCHS_NN_TOTAL)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_NN_TOTAL
    )

    # 3. SIGMA-D (Puro)
    m = copy.deepcopy(base_model)
    opt = SIGMA_D_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD)
    results_nn['SIGMA-D_v2 (Puro)'], times_nn['SIGMA-D_v2 (Puro)'] = run_experiment(
        "[NN] SIGMA-D_v2 (Puro)", m, [(opt, N_EPOCHS_NN_TOTAL)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_NN_TOTAL
    )

    # 4. SIGMA-C (Puro)
    m = copy.deepcopy(base_model)
    opt = SIGMA_C_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD)
    results_nn['SIGMA-C_v2 (Puro)'], times_nn['SIGMA-C_v2 (Puro)'] = run_experiment(
        "[NN] SIGMA-C_v2 (Puro)", m, [(opt, N_EPOCHS_NN_TOTAL)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_NN_TOTAL
    )
    
    # 5. Adam -> SGD+M (Híbrido)
    m = copy.deepcopy(base_model)
    opt1 = optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD)
    opt2 = optim.SGD(m.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD)
    results_nn['Adam -> SGD+M'], times_nn['Adam -> SGD+M'] = run_experiment(
        "[NN] Adam -> SGD+M", m, [(opt1, N_EPOCHS_NN_PHASE1), (opt2, N_EPOCHS_NN_PHASE2)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_NN_TOTAL
    )

    # 6. Adam -> SIGMA-D (Híbrido)
    m = copy.deepcopy(base_model)
    opt1 = optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD)
    opt2 = SIGMA_D_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD)
    results_nn['Adam -> SIGMA-D_v2'], times_nn['Adam -> SIGMA-D_v2'] = run_experiment(
        "[NN] Adam -> SIGMA-D_v2", m, [(opt1, N_EPOCHS_NN_PHASE1), (opt2, N_EPOCHS_NN_PHASE2)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_NN_TOTAL
    )

    # 7. Adam -> SIGMA-C (Híbrido)
    m = copy.deepcopy(base_model)
    opt1 = optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD)
    opt2 = SIGMA_C_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD)
    results_nn['Adam -> SIGMA-C_v2'], times_nn['Adam -> SIGMA-C_v2'] = run_experiment(
        "[NN] Adam -> SIGMA-C_v2", m, [(opt1, N_EPOCHS_NN_PHASE1), (opt2, N_EPOCHS_NN_PHASE2)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_NN_TOTAL
    )
    
    # 8. Cíclico (A->S)x2 (5 épocas cada)
    m = copy.deepcopy(base_model)
    opt_list = [
        (optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD), 5),
        (optim.SGD(m.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD), 5),
        (optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD), 5),
        (optim.SGD(m.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD), 5)
    ]
    results_nn['Cíclico (A->S)x2'], times_nn['Cíclico (A->S)x2'] = run_experiment(
        "[NN] Cíclico (A->S)x2", m, opt_list, train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_NN_TOTAL
    )

    # 9. Cíclico (A->D)x2 (5 épocas cada)
    m = copy.deepcopy(base_model)
    opt_list = [
        (optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD), 5),
        (SIGMA_D_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD), 5),
        (optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD), 5),
        (SIGMA_D_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD), 5)
    ]
    results_nn['Cíclico (A->D)x2'], times_nn['Cíclico (A->D)x2'] = run_experiment(
        "[NN] Cíclico (A->D)x2", m, opt_list, train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_NN_TOTAL
    )

    # 10. Cíclico (A->C)x2 (5 épocas cada)
    m = copy.deepcopy(base_model)
    opt_list = [
        (optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD), 5),
        (SIGMA_C_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD), 5),
        (optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD), 5),
        (SIGMA_C_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD), 5)
    ]
    results_nn['Cíclico (A->C)x2'], times_nn['Cíclico (A->C)x2'] = run_experiment(
        "[NN] Cíclico (A->C)x2", m, opt_list, train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_NN_TOTAL
    )
    
    # =======================================================================
    # PARTE 2: EXPERIMENTOS COM REGRESSÃO LOGÍSTICA (CONVEXO)
    # =======================================================================
    print("\n" + "="*80)
    print("PARTE 2: REGRESSÃO LOGÍSTICA (Conjunto Completo)")
    print("="*80)
    
    base_logistic = LogisticRegression().to(DEVICE)
    results_lr = {}
    times_lr = {}
    
    # 1. Adam (Puro)
    m = copy.deepcopy(base_logistic)
    opt = optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD)
    results_lr['Adam (Puro)'], times_lr['Adam (Puro)'] = run_experiment(
        "[LR] Adam (Puro)", m, [(opt, N_EPOCHS_LR_TOTAL)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_LR_TOTAL
    )
    
    # 2. SGD+M (Puro)
    m = copy.deepcopy(base_logistic)
    opt = optim.SGD(m.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD)
    results_lr['SGD+M (Puro)'], times_lr['SGD+M (Puro)'] = run_experiment(
        "[LR] SGD+M (Puro)", m, [(opt, N_EPOCHS_LR_TOTAL)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_LR_TOTAL
    )
    
    # 3. SIGMA-D (Puro)
    m = copy.deepcopy(base_logistic)
    opt = SIGMA_D_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD)
    results_lr['SIGMA-D_v2 (Puro)'], times_lr['SIGMA-D_v2 (Puro)'] = run_experiment(
        "[LR] SIGMA-D_v2 (Puro)", m, [(opt, N_EPOCHS_LR_TOTAL)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_LR_TOTAL
    )

    # 4. SIGMA-C (Puro)
    m = copy.deepcopy(base_logistic)
    opt = SIGMA_C_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD)
    results_lr['SIGMA-C_v2 (Puro)'], times_lr['SIGMA-C_v2 (Puro)'] = run_experiment(
        "[LR] SIGMA-C_v2 (Puro)", m, [(opt, N_EPOCHS_LR_TOTAL)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_LR_TOTAL
    )
    
    # 5. Adam -> SGD+M (Híbrido)
    m = copy.deepcopy(base_logistic)
    opt1 = optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD)
    opt2 = optim.SGD(m.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD)
    results_lr['Adam -> SGD+M'], times_lr['Adam -> SGD+M'] = run_experiment(
        "[LR] Adam -> SGD+M", m, [(opt1, N_EPOCHS_LR_PHASE1), (opt2, N_EPOCHS_LR_PHASE2)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_LR_TOTAL
    )

    # 6. Adam -> SIGMA-D (Híbrido)
    m = copy.deepcopy(base_logistic)
    opt1 = optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD)
    opt2 = SIGMA_D_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD)
    results_lr['Adam -> SIGMA-D_v2'], times_lr['Adam -> SIGMA-D_v2'] = run_experiment(
        "[LR] Adam -> SIGMA-D_v2", m, [(opt1, N_EPOCHS_LR_PHASE1), (opt2, N_EPOCHS_LR_PHASE2)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_LR_TOTAL
    )

    # 7. Adam -> SIGMA-C (Híbrido)
    m = copy.deepcopy(base_logistic)
    opt1 = optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD)
    opt2 = SIGMA_C_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD)
    results_lr['Adam -> SIGMA-C_v2'], times_lr['Adam -> SIGMA-C_v2'] = run_experiment(
        "[LR] Adam -> SIGMA-C_v2", m, [(opt1, N_EPOCHS_LR_PHASE1), (opt2, N_EPOCHS_LR_PHASE2)], 
        train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_LR_TOTAL
    )

    # 8. Cíclico (A->S)x2 (7+8+7+8 = 30 épocas)
    m = copy.deepcopy(base_logistic)
    opt_list = [
        (optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD), 7),
        (optim.SGD(m.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD), 8),
        (optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD), 7),
        (optim.SGD(m.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD), 8)
    ]
    results_lr['Cíclico (A->S)x2'], times_lr['Cíclico (A->S)x2'] = run_experiment(
        "[LR] Cíclico (A->S)x2", m, opt_list, train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_LR_TOTAL
    )

    # 9. Cíclico (A->D)x2 (7+8+7+8)
    m = copy.deepcopy(base_logistic)
    opt_list = [
        (optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD), 7),
        (SIGMA_D_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD), 8),
        (optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD), 7),
        (SIGMA_D_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD), 8)
    ]
    results_lr['Cíclico (A->D)x2'], times_lr['Cíclico (A->D)x2'] = run_experiment(
        "[LR] Cíclico (A->D)x2", m, opt_list, train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_LR_TOTAL
    )

    # 10. Cíclico (A->C)x2 (7+8+7+8)
    m = copy.deepcopy(base_logistic)
    opt_list = [
        (optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD), 7),
        (SIGMA_C_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD), 8),
        (optim.Adam(m.parameters(), lr=LR_ADAM, weight_decay=WD), 7),
        (SIGMA_C_v2(m.parameters(), lr=LR_SIGMA, beta=0.9, weight_decay=WD), 8)
    ]
    results_lr['Cíclico (A->C)x2'], times_lr['Cíclico (A->C)x2'] = run_experiment(
        "[LR] Cíclico (A->C)x2", m, opt_list, train_loader, test_loader, DEVICE, loss_fn, N_EPOCHS_LR_TOTAL
    )
    
    # ========================================================================
    # GERAÇÃO DE PLOTS
    # ========================================================================
    
    generate_nn_plots(results_nn, times_nn, N_EPOCHS_NN_PHASE1)
    generate_lr_plots(results_lr, times_lr, N_EPOCHS_LR_PHASE1)
    
    # [NOVO] Gera o gráfico combinado específico (Plot 3)
    generate_combined_plot(results_nn, times_nn, results_lr, times_lr, N_EPOCHS_NN_PHASE1, N_EPOCHS_LR_PHASE1)
    
    print("Benchmark Completo Finalizado!")

if __name__ == "__main__":
    main()