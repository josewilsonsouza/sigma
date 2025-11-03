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

# Importar o seu novo otimizador AdamS
try:
    from adams import AdamS
except ImportError:
    print("="*80)
    print("ERRO: O arquivo 'adams.py' não foi encontrado.")
    print("Certifique-se de salvar ambos os arquivos no mesmo diretório.")
    print("="*80)
    exit()

# --- 1. Definições do Modelo e Dados ---

class MNISTNet(nn.Module):
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

def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# --- 2. Funções de Treinamento e Avaliação ---

def train_epoch(model, optimizer, train_loader, device, loss_fn):
    model.train()
    total_loss = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            return loss

        loss_tensor = closure()
        loss_item = loss_tensor.item()

        # Atualizado para lidar com AdamS
        if isinstance(optimizer, AdamS):
            optimizer.step(loss_item=loss_item) 
        else:
            optimizer.step()
        
        total_loss += loss_item
        
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device, loss_fn):
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

# --- 3. Script Principal de Comparação ---

def run_full_comparison(n_epochs_total=20):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {DEVICE}")
    
    train_loader, test_loader = get_data_loaders()
    loss_fn = nn.CrossEntropyLoss()
    
    base_model = MNISTNet().to(DEVICE)
    
    results = {}
    times = {}
    
    # --- Configurações dos Experimentos ---
    # (Baseado no benchmark sigma_mnist.py)
    
    configs = [
        {
            "name": "Adam (Baseline)",
            "optimizer": optim.Adam,
            "opt_kwargs": {"lr": 0.001}
        },
        {
            "name": "SGD+M (Baseline)",
            "optimizer": optim.SGD,
            "opt_kwargs": {"lr": 0.01, "momentum": 0.9}
        },
        {
            "name": "Híbrido (Adam -> SGD+M)",
            "optimizer": "hybrid",
            "phase1_opt": optim.Adam,
            "phase1_kwargs": {"lr": 0.001},
            "phase2_opt": optim.SGD,
            "phase2_kwargs": {"lr": 0.01, "momentum": 0.9}
        },
        {
            "name": "AdamS (Integrado)",
            "optimizer": AdamS,
            "opt_kwargs": {
                "lr": 0.001, # LR do Adam
                "betas": (0.9, 0.999), # Betas do Adam
                "beta_s": 0.999, # Beta do SIGMA
                "alpha_max": 2.0,
                "warmup_steps": 10 * len(train_loader) # Warmup de 10 épocas
            }
        }
    ]

    for config in configs:
        name = config["name"]
        print("\n" + "="*80)
        print(f"TREINANDO COM: {name}")
        print("="*80)
        
        model = copy.deepcopy(base_model)
        history = {'test_acc': []}
        start_time = time.time()
        
        if config["optimizer"] == "hybrid":
            # --- Lógica Híbrida ---
            n_epochs_phase1 = n_epochs_total // 2
            n_epochs_phase2 = n_epochs_total - n_epochs_phase1
            
            # Fase 1
            print(f"--- FASE 1: Adam ({n_epochs_phase1} épocas) ---")
            opt_phase1 = config["phase1_opt"](model.parameters(), **config["phase1_kwargs"])
            for epoch in range(n_epochs_phase1):
                train_epoch(model, opt_phase1, train_loader, DEVICE, loss_fn)
                _, test_acc = evaluate(model, test_loader, DEVICE, loss_fn)
                history['test_acc'].append(test_acc)
                print(f"Epoch [{epoch+1:2d}/{n_epochs_total}] | Acc: {test_acc:.2f}%")
                
            # Fase 2
            print(f"--- FASE 2: Trocando para SGD+M ({n_epochs_phase2} épocas) ---")
            opt_phase2 = config["phase2_opt"](model.parameters(), **config["phase2_kwargs"])
            for epoch in range(n_epochs_phase2):
                train_epoch(model, opt_phase2, train_loader, DEVICE, loss_fn)
                _, test_acc = evaluate(model, test_loader, DEVICE, loss_fn)
                history['test_acc'].append(test_acc)
                print(f"Epoch [{epoch+1+n_epochs_phase1:2d}/{n_epochs_total}] | Acc: {test_acc:.2f}%")
        
        else:
            # --- Lógica de Otimizador Único (Adam, SGD+M, AdamS) ---
            optimizer = config["optimizer"](model.parameters(), **config["opt_kwargs"])
            for epoch in range(n_epochs_total):
                train_epoch(model, optimizer, train_loader, DEVICE, loss_fn)
                _, test_acc = evaluate(model, test_loader, DEVICE, loss_fn)
                history['test_acc'].append(test_acc)
                print(f"Epoch [{epoch+1:2d}/{n_epochs_total}] | Acc: {test_acc:.2f}%")
        
        times[name] = time.time() - start_time
        results[name] = history
        print(f"--- {name} concluído em {times[name]:.2f}s ---")


    # --- 4. Plotar Resultados ---
    print("\nGerando gráficos de comparação...")
    plt.figure(figsize=(12, 7))
    
    n_epochs_phase1 = n_epochs_total // 2
    for name, history in results.items():
        plt.plot(history['test_acc'], label=name, marker='o', markersize=4, alpha=0.8)
        
    plt.axvline(x=n_epochs_phase1 - 0.5, color='r', linestyle='--', label='Troca (Híbrido)')
    plt.xlabel('Época')
    plt.ylabel('Acurácia no Teste (%)')
    plt.title('Comparação de Otimizadores: AdamS vs Híbrido (MNIST)')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('adams_full_comparison.png', dpi=150)
    print("Gráfico salvo: 'adams_full_comparison.png'")
    
    # --- 5. Resumo Final ---
    print("\n" + "="*80)
    print("RESUMO FINAL DA COMPARAÇÃO")
    print("="*80)
    print(f"{'Otimizador':<24} | {'Acurácia Final':<15} | {'Tempo Total':<12}")
    print("-"*80)
    
    for name, history in results.items():
        acc = history['test_acc'][-1]
        t = times[name]
        print(f"{name:<24} | {acc:<15.2f}% | {t:<12.2f}s")
    print("-"*80)

if __name__ == "__main__":
    run_full_comparison(n_epochs_total=10)