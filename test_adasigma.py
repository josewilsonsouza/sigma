"""
Teste Comparativo: Adam vs SIGMA vs Híbrido vs AdaSIGMA
========================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from adasigma import AdaSIGMA
from adams import AdamS
from sigma import SIGMA
import matplotlib.pyplot as plt
import time
import copy

# Modelo simples
def create_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10)
    )

# Dados
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

loss_fn = nn.CrossEntropyLoss()

# Função de avaliação
def evaluate(model):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            total_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    acc = 100. * correct / len(test_loader.dataset)
    loss = total_loss / len(test_loader)
    return loss, acc

# Função de treino
def train_experiment(name, model, optimizer_config, n_epochs=10):
    print(f"\n{'='*60}")
    print(f"Treinando: {name}")
    print(f"{'='*60}")
    
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    start_time = time.time()
    epoch_count = 0
    
    for optimizer, epochs in optimizer_config:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                
                # SIGMA, AdaSIGMA e AdamS precisam de loss_item
                if isinstance(optimizer, (SIGMA, AdaSIGMA, AdamS)):
                    optimizer.step(loss_item=loss.item())
                else:
                    optimizer.step()
                
                total_loss += loss.item()
            
            # Métricas
            train_loss = total_loss / len(train_loader)
            test_loss, test_acc = evaluate(model)
            
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            epoch_count += 1
            print(f"Época {epoch_count:2d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Acc: {test_acc:.2f}%")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Concluído em {elapsed:.1f}s")
    
    return history, elapsed

# =============================================================================
# EXPERIMENTOS - COMPARAÇÃO FOCADA: Adam vs AdaSIGMA vs AdamS
# =============================================================================

N_EPOCHS = 10

results = {}
times = {}

# 1. Adam (Baseline)
print("\n" + "="*60)
print("EXPERIMENTO 1: Adam (Baseline)")
print("="*60)

model_adam = create_model()
optimizer_adam = optim.Adam(model_adam.parameters(), lr=1e-3)

history_adam, time_adam = train_experiment(
    "Adam",
    model_adam,
    [(optimizer_adam, N_EPOCHS)],
    n_epochs=N_EPOCHS
)

results['Adam'] = history_adam
times['Adam'] = time_adam

# 2. AdaSIGMA (Fusão Paralela)
print("\n" + "="*60)
print("EXPERIMENTO 2: AdaSIGMA (Fusão Paralela)")
print("="*60)

model_adasigma = create_model()
optimizer_adasigma = AdaSIGMA(model_adasigma.parameters(), lr=1e-3)

history_adasigma, time_adasigma = train_experiment(
    "AdaSIGMA",
    model_adasigma,
    [(optimizer_adasigma, N_EPOCHS)],
    n_epochs=N_EPOCHS
)

results['AdaSIGMA'] = history_adasigma
times['AdaSIGMA'] = time_adasigma

# 3. AdamS (Modulação Final)
print("\n" + "="*60)
print("EXPERIMENTO 3: AdamS (Modulação Final)")
print("="*60)

model_adams = create_model()
optimizer_adams = AdamS(model_adams.parameters(), lr=1e-3, 
                        beta_s=0.9, alpha_min=0.5, alpha_max=1.5,
                        warmup_steps=3)

history_adams, time_adams = train_experiment(
    "AdamS",
    model_adams,
    [(optimizer_adams, N_EPOCHS)],
    n_epochs=N_EPOCHS
)

results['AdamS'] = history_adams
times['AdamS'] = time_adams

# =============================================================================
# VISUALIZAÇÃO
# =============================================================================

print("\n" + "="*60)
print("Gerando gráficos...")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = {
    'Adam': '#1f77b4',
    'AdaSIGMA': '#ff7f0e',
    'AdamS': '#2ca02c'
}

markers = {
    'Adam': 'o',
    'AdaSIGMA': 's',
    'AdamS': '^'
}

# Plot 1: Train Loss
ax1 = axes[0, 0]
for name, history in results.items():
    ax1.plot(history['train_loss'], 
             label=name, 
             color=colors[name],
             marker=markers[name],
             markersize=5,
             linewidth=2)

ax1.set_xlabel('Época', fontsize=11)
ax1.set_ylabel('Train Loss', fontsize=11)
ax1.set_title('Loss de Treino', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Test Loss
ax2 = axes[0, 1]
for name, history in results.items():
    ax2.plot(history['test_loss'], 
             label=name,
             color=colors[name],
             marker=markers[name],
             markersize=5,
             linewidth=2)

ax2.set_xlabel('Época', fontsize=11)
ax2.set_ylabel('Test Loss', fontsize=11)
ax2.set_title('Loss de Teste', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Test Accuracy
ax3 = axes[1, 0]
for name, history in results.items():
    ax3.plot(history['test_acc'], 
             label=name,
             color=colors[name],
             marker=markers[name],
             markersize=5,
             linewidth=2)

ax3.set_xlabel('Época', fontsize=11)
ax3.set_ylabel('Acurácia (%)', fontsize=11)
ax3.set_title('Acurácia no Teste', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Tempo de Treinamento
ax4 = axes[1, 1]
names = list(times.keys())
time_values = list(times.values())
bars = ax4.barh(names, time_values, color=[colors[n] for n in names])

for i, (bar, val) in enumerate(zip(bars, time_values)):
    ax4.text(val + 2, i, f'{val:.1f}s', va='center', fontsize=10)

ax4.set_xlabel('Tempo (s)', fontsize=11)
ax4.set_title('Eficiência Computacional', fontsize=13, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('adams_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Gráfico salvo: 'adams_comparison.png'")

# =============================================================================
# RESUMO FINAL
# =============================================================================

print("\n" + "="*60)
print("RESUMO FINAL")
print("="*60)

print(f"\n{'Método':<20} | {'Acc Final':<10} | {'Loss Final':<11} | {'Tempo (s)':<10}")
print("-" * 60)

for name in results.keys():
    acc = results[name]['test_acc'][-1]
    loss = results[name]['test_loss'][-1]
    t = times[name]
    print(f"{name:<20} | {acc:>9.2f}% | {loss:>10.4f} | {t:>9.1f}s")

# Comparação com Adam
print("\n" + "-"*60)
print("COMPARAÇÃO vs Adam (Baseline)")
print("-"*60)

baseline_acc = results['Adam']['test_acc'][-1]
baseline_time = times['Adam']

for name in ['AdaSIGMA', 'AdamS']:
    acc_diff = results[name]['test_acc'][-1] - baseline_acc
    time_diff = times[name] - baseline_time
    
    print(f"\n{name}:")
    print(f"  Δ Acurácia: {acc_diff:+.2f}% {'✓' if acc_diff > 0 else '✗'}")
    print(f"  Δ Tempo: {time_diff:+.1f}s {'✓' if time_diff < 5 else '✗'}")

# Comparação crítica: AdamS vs AdaSIGMA
print("\n" + "="*60)
print("COMPARAÇÃO CRÍTICA: AdamS vs AdaSIGMA")
print("="*60)

acc_adam = results['Adam']['test_acc'][-1]
acc_adams = results['AdamS']['test_acc'][-1]
acc_adasigma = results['AdaSIGMA']['test_acc'][-1]

time_adam = times['Adam']
time_adams = times['AdamS']
time_adasigma = times['AdaSIGMA']

print(f"\nAcurácia Adam (baseline): {acc_adam:.2f}%")
print(f"Acurácia AdamS:           {acc_adams:.2f}%")
print(f"Acurácia AdaSIGMA:        {acc_adasigma:.2f}%")

print(f"\nTempo Adam:      {time_adam:.1f}s")
print(f"Tempo AdamS:     {time_adams:.1f}s")
print(f"Tempo AdaSIGMA:  {time_adasigma:.1f}s")

print("\n" + "-"*60)
print("ANÁLISE:")
print("-"*60)

if acc_adams > acc_adasigma:
    diff = acc_adams - acc_adasigma
    print(f"✓ AdamS é {diff:.2f}% superior ao AdaSIGMA")
    print("  → Modulação final (AdamS) > Fusão paralela (AdaSIGMA)")
else:
    diff = acc_adasigma - acc_adams
    print(f"✗ AdaSIGMA é {diff:.2f}% superior ao AdamS")
    print("  → Fusão paralela funciona melhor")

if acc_adams > acc_adam:
    diff = acc_adams - acc_adam
    print(f"\n✓ AdamS melhora o Adam em {diff:.2f}%")
    print("  → Score σ como freio estabilizador está funcionando!")
else:
    diff = acc_adam - acc_adams
    print(f"\n✗ AdamS é {diff:.2f}% pior que Adam baseline")
    print("  → Score σ não está agregando valor")

overhead = time_adams - time_adam
overhead_pct = 100 * overhead / time_adam
print(f"\nOverhead computacional do AdamS: +{overhead:.1f}s (+{overhead_pct:.1f}%)")

print("\n" + "="*60)
print("✓ Experimento concluído!")
print("="*60)