# ============================================
# SECTION 1: Imports
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set seed so results are reproducible
torch.manual_seed(42)

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================
# SECTION 2: PrunableLinear Layer
# This is the custom layer with gates
# ============================================

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()

        # Standard weight and bias (just like nn.Linear)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores — one for every weight
        # These start at 0 (sigmoid(0) = 0.5, so gates start half open)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        # Step 1: Convert gate_scores to values between 0 and 1
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: Multiply each weight by its gate
        # If gate = 0, that weight is effectively pruned!
        pruned_weights = self.weight * gates

        # Step 3: Do the normal linear operation
        return nn.functional.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        # Helper function to get current gate values
        return torch.sigmoid(self.gate_scores)


# ============================================
# SECTION 3: Neural Network Definition
# A simple network using our PrunableLinear layers
# ============================================

class SelfPruningNetwork(nn.Module):
    def __init__(self):
        super(SelfPruningNetwork, self).__init__()

        # CIFAR-10 images are 32x32 pixels with 3 colors (RGB)
        # So input size = 32 * 32 * 3 = 3072
        self.layer1 = PrunableLinear(3072, 512)
        self.layer2 = PrunableLinear(512, 256)
        self.layer3 = PrunableLinear(256, 128)
        self.layer4 = PrunableLinear(128, 10)  # 10 output classes

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Flatten the image into a 1D vector
        # (batch_size, 3, 32, 32) → (batch_size, 3072)
        x = x.view(x.size(0), -1)

        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.layer4(x)

        return x

    def get_all_gates(self):
        # Collect gate values from ALL layers into one list
        all_gates = []
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            all_gates.append(layer.get_gates().detach().flatten())
        return torch.cat(all_gates)


# ============================================
# SECTION 4: Load CIFAR-10 Dataset
# ============================================

def load_data():
    # Normalize images to have mean=0.5, std=0.5
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load training data
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True
    )

    # Download and load test data
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=128,
        shuffle=False
    )

    return trainloader, testloader


# ============================================
# SECTION 5: Sparsity Loss Calculation
# Penalize the network for keeping gates open
# ============================================

def sparsity_loss(model):
    # Add up ALL gate values across ALL layers
    # The optimizer will try to minimize this → pushes gates toward 0
    total = 0
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        gates = layer.get_gates()
        total += gates.sum()
    return total


# ============================================
# SECTION 6: Training Loop
# ============================================

def train(model, trainloader, optimizer, lambda_val, epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate classification loss (how wrong are the predictions)
            class_loss = criterion(outputs, labels)

            # Calculate sparsity loss (how many gates are still open)
            sparse_loss = sparsity_loss(model)

            # Total loss = accuracy loss + punishment for open gates
            loss = class_loss + lambda_val * sparse_loss

            # Backward pass — calculate gradients
            loss.backward()

            # Update weights AND gate_scores
            optimizer.step()

            total_loss += loss.item()

            # Calculate training accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {total_loss/len(trainloader):.3f} | "
              f"Train Accuracy: {accuracy:.1f}%")


# ============================================
# SECTION 7: Evaluation
# ============================================

def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return accuracy


# ============================================
# SECTION 8: Calculate Sparsity Level
# What % of weights are pruned (gate < 0.01)?
# ============================================

def calculate_sparsity(model):
    all_gates = model.get_all_gates()
    # Count how many gates are basically zero (below threshold)
    pruned = (all_gates < 0.01).sum().item()
    total = all_gates.numel()
    sparsity = 100. * pruned / total
    return sparsity


# ============================================
# SECTION 9: Plot Gate Distribution
# ============================================

def plot_gates(model, lambda_val):
    all_gates = model.get_all_gates().cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.hist(all_gates, bins=100, color='steelblue', edgecolor='black')
    plt.title(f'Gate Value Distribution (λ = {lambda_val})')
    plt.xlabel('Gate Value (0 = pruned, 1 = active)')
    plt.ylabel('Number of Gates')
    plt.axvline(x=0.01, color='red', linestyle='--',
                label='Pruning threshold (0.01)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/gate_distribution_lambda_{lambda_val}.png')
    plt.show()
    print(f"Plot saved!")


# ============================================
# SECTION 10: Main — Run for 3 Lambda Values
# ============================================

import os
os.makedirs('results', exist_ok=True)  # Create results folder

# Load data once — reuse for all experiments
trainloader, testloader = load_data()

# The 3 lambda values to test
lambda_values = [0.0001, 0.001, 0.01]

results = []

for lambda_val in lambda_values:
    print(f"\n{'='*50}")
    print(f"Training with λ = {lambda_val}")
    print(f"{'='*50}")

    # Fresh model for each lambda
    model = SelfPruningNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for 10 epochs
    train(model, trainloader, optimizer, lambda_val, epochs=10)

    # Evaluate on test set
    test_accuracy = evaluate(model, testloader)
    sparsity = calculate_sparsity(model)

    print(f"\nResults for λ = {lambda_val}:")
    print(f"  Test Accuracy : {test_accuracy:.1f}%")
    print(f"  Sparsity Level: {sparsity:.1f}%")

    results.append((lambda_val, test_accuracy, sparsity))

    # Plot gate distribution for this lambda
    plot_gates(model, lambda_val)


# ============================================
# SECTION 11: Print Final Results Table
# ============================================

print("\n")
print("=" * 50)
print("FINAL RESULTS SUMMARY")
print("=" * 50)
print(f"{'Lambda':<12} {'Test Accuracy':<18} {'Sparsity Level'}")
print("-" * 50)
for lam, acc, spar in results:
    print(f"{lam:<12} {acc:<18.1f}% {spar:.1f}%")