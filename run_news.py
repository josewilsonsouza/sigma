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

# Importar o seu novo otimizador AdamGeoBeta
try:
    from adageobeta import AdamGeoBeta
except ImportError:
    print("="*80)
    print("ERRO: O arquivo 'adamgeobeta.py' não foi encontrado.")
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
    
    return train_loader, test_loader, len(train_loader)

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

        # Otimizador espera o loss_item
        optimizer.step(loss_item=loss_item) 
        
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

# --- 3. Script Principal de Treinamento ---

def run_solo_experiment(n_epochs_total=20):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {DEVICE}")
    
    train_loader, test_loader, steps_per_epoch = get_data_loaders()
    loss_fn = nn.CrossEntropyLoss()
    
    base_model = MNISTNet().to(DEVICE)
    
    results = {}
    times = {}
    
    # --- Configuração do AdamGeoBeta ---
    
    config = {
        "name": "AdamGeoBeta (Modula beta_2)",
        "optimizer": AdamGeoBeta,
        "opt_kwargs": {
            "lr": 0.001, # LR do Adam
            "betas": (0.9, 0.999), # Betas do Adam (beta2 é o *base*)
            "beta_s": 0.9, # Beta do SIGMA (Freio Rápido)
            "alpha_max": 2.0,
            "warmup_steps": 5 * steps_per_epoch # Warmup de 5 épocas
        }
    }

    name = config["name"]
    print("\n" + "="*80)
    print(f"TREINANDO COM: {name}")
    print("="*80)
    
    model = copy.deepcopy(base_model)
    history = {'test_loss': [], 'test_acc': []}
    start_time = time.time()
    
    #
    optimizer = config["optimizer"](model.parameters(), **config["opt_kwargs"])
    
    for epoch in range(n_epochs_total):
        train_loss = train_epoch(model, optimizer, train_loader, DEVICE, loss_fn)
        test_loss, test_acc = evaluate(model, test_loader, DEVICE, loss_fn)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch [{epoch+1:2d}/{n_epochs_total}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.2f}%")
    
    times[name] = time.time() - start_time
    results[name] = history
    print(f"--- {name} concluído em {times[name]:.2f}s ---")


    # --- 4. Plotar Resultados ---
    print("\nGerando gráficos de desempenho...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Acurácia
    ax1.plot(history['test_acc'], label=name, marker='o', markersize=4, color='green')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia no Teste (%)')
    ax1.set_title('Desempenho de Acurácia: AdamGeoBeta')
    ax1.legend()
    ax1.grid(True, alpha=0.4)
    
    # Plot 2: Loss
    ax2.plot(history['test_loss'], label=name, marker='s', markersize=4, color='green')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss no Teste (Log)')
    ax2.set_title('Desempenho de Loss: AdamGeoBeta')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('adamgeobeta_solo.png', dpi=150)
    print("Gráfico salvo: 'adamgeobeta_solo.png'")
    
    # --- 5. Resumo Final ---
    print("\n" + "="*80)
    print("RESUMO FINAL")
    print("="*80)
    print(f"{'Otimizador':<24} | {'Acurácia Final':<15} | {'Tempo Total':<12}")
    print("-"*60)
    
    acc = history['test_acc'][-1]
    t = times[name]
    print(f"{name:<24} | {acc:<15.2f}% | {t:<12.2f}s")
    print("-"*60)

if __name__ == "__main__":
    run_solo_experiment(n_epochs_total=20)