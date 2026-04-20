import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Model Components ---

class GatedLinear(nn.Module):
    """
    A Linear layer with learnable gates for each output neuron.
    G(x) = (W * x + b) * sigmoid(gate)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Initialize gates to 1.0 (sigmoid(1.0) approx 0.73)
        self.gate = nn.Parameter(torch.ones(out_features) * 0.5)

    def forward(self, x):
        out = self.linear(x)
        mask = torch.sigmoid(self.gate)
        return out * mask

class PrunableNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = GatedLinear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = GatedLinear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_sparsity(self, threshold=0.5):
        """Calculates the percentage of gates that are 'off'."""
        total_gates = 0
        pruned_gates = 0
        for m in self.modules():
            if isinstance(m, GatedLinear):
                mask = torch.sigmoid(m.gate)
                total_gates += mask.numel()
                pruned_gates += torch.sum(mask < threshold).item()
        return (pruned_gates / total_gates) * 100

# --- Training Logic ---

def train_and_eval(sparsity_lambda, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Using small subset for speed in demo
    train_loader = DataLoader(torch.utils.data.Subset(train_set, range(5000)), batch_size=64, shuffle=True)
    test_loader = DataLoader(torch.utils.data.Subset(test_set, range(1000)), batch_size=64)

    model = PrunableNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            # Loss = CrossEntropy + Lambda * L1(sigmoid(gates))
            ce_loss = criterion(output, target)
            
            reg_loss = 0
            for m in model.modules():
                if isinstance(m, GatedLinear):
                    reg_loss += torch.sum(torch.sigmoid(m.gate))
            
            total_loss = ce_loss + sparsity_lambda * reg_loss
            total_loss.backward()
            optimizer.step()

    # Final Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    sparsity = model.get_sparsity()
    
    return accuracy, sparsity, model

# --- Main Execution ---

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    lambdas = [0.0, 0.05, 0.1, 0.5]
    results = []

    print(f"{'Lambda':<10} | {'Accuracy':<10} | {'Sparsity %':<10}")
    print("-" * 35)

    best_model = None
    
    for l in lambdas:
        acc, sp, model = train_and_eval(l)
        results.append((l, acc, sp))
        print(f"{l:<10} | {acc:<10.2f} | {sp:<10.2f}")
        if l == 0.1: # We'll plot the distribution for a medium lambda
            best_model = model

    # 1. Save Results Table
    with open("results/results_table.md", "w") as f:
        f.write("# Pruning Experiments\n\n")
        f.write("| Lambda | Accuracy (%) | Sparsity (%) |\n")
        f.write("|--------|--------------|--------------|\n")
        for l, acc, sp in results:
            f.write(f"| {l} | {acc:.2f} | {sp:.2f} |\n")

    # 2. Save Gate Distribution Plot
    if best_model:
        all_gates = []
        for m in best_model.modules():
            if isinstance(m, GatedLinear):
                all_gates.extend(torch.sigmoid(m.gate).detach().cpu().numpy())
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_gates, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Distribution of Learnable Gates (Lambda=0.01)')
        plt.xlabel('Gate Value (Sigmoid Activation)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig("results/gate_distribution.png")
        print("\nGenerated results/gate_distribution.png")

    print("\nResults table saved to results/results_table.md")
