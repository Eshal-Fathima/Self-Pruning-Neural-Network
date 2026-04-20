# =============================================================================
# Self-Pruning Neural Network — Tredence AI Engineering Internship Case Study
# Author: [Your Name]
# Description:
#   A feed-forward neural network that learns to prune its own weights
#   during training using learnable sigmoid gates and L1 sparsity
#   regularization, trained and evaluated on the CIFAR-10 dataset.
# =============================================================================


# =============================================================================
# SECTION 1: Imports
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create results folder
os.makedirs("results", exist_ok=True)


# =============================================================================
# SECTION 2: PrunableLinear Layer
#
# How it works:
#   - Every weight has a corresponding learnable "gate_score"
#   - gate_score → Sigmoid → value between 0 and 1 (the gate)
#   - pruned_weight = weight × gate
#   - If gate → 0, that weight is effectively removed (pruned)
#   - Both weight and gate_score are nn.Parameter so PyTorch's autograd
#     tracks gradients through BOTH automatically during backpropagation
# =============================================================================

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Standard weight — Kaiming init works well with ReLU
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

        # Standard bias
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores — same shape as weight
        # sigmoid(0) = 0.5, so gates start half-open
        # We initialize to 2.0 so sigmoid(2) ≈ 0.88 (mostly open at start)
        # This gives the network room to close gates during training
        self.gate_scores = nn.Parameter(
            torch.ones(out_features, in_features) * 2.0
        )

    def forward(self, x):
        # Step 1: Sigmoid squishes gate_scores to (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: Multiply each weight by its gate
        pruned_weights = self.weight * gates

        # Step 3: Standard linear operation
        # Gradients flow back to BOTH self.weight and self.gate_scores
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)

    def sparsity_loss(self):
        # L1 norm: sum of all gate values
        # Minimizing this pushes gates toward 0
        return torch.sigmoid(self.gate_scores).sum()


# =============================================================================
# SECTION 3: Self-Pruning Network
#
# Smaller network (3 layers instead of 4) so it trains faster on CPU
# CIFAR-10: 32×32×3 = 3072 input features, 10 output classes
# =============================================================================

class SelfPruningNetwork(nn.Module):
    def __init__(self):
        super(SelfPruningNetwork, self).__init__()

        # Smaller architecture — faster on CPU, still demonstrates pruning
        self.layer1 = PrunableLinear(3072, 256)
        self.layer2 = PrunableLinear(256, 128)
        self.layer3 = PrunableLinear(128, 10)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Flatten: (batch, 3, 32, 32) → (batch, 3072)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.bn1(self.layer1(x))))
        x = self.dropout(self.relu(self.bn2(self.layer2(x))))
        x = self.layer3(x)
        return x

    def get_prunable_layers(self):
        return [self.layer1, self.layer2, self.layer3]

    def total_sparsity_loss(self):
        """
        Sum of L1 gate norms across ALL layers.
        This is the SparsityLoss in:
            Total Loss = ClassificationLoss + λ × SparsityLoss
        """
        return sum(layer.sparsity_loss() for layer in self.get_prunable_layers())

    def get_all_gates(self):
        all_gates = [
            layer.get_gates().flatten()
            for layer in self.get_prunable_layers()
        ]
        return torch.cat(all_gates)

    def calculate_sparsity_percent(self, threshold=1e-2):
        """
        % of gates below threshold = effectively pruned.
        Higher % = sparser = more compressed model.
        """
        all_gates = self.get_all_gates()
        pruned = (all_gates < threshold).sum().item()
        total = all_gates.numel()
        return 100.0 * pruned / total


# =============================================================================
# SECTION 4: Load CIFAR-10
# Downloads automatically on first run into ./data/
# =============================================================================

def load_data(batch_size=256):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    # num_workers=0 avoids multiprocessing issues on some systems
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Training samples : {len(trainset)}")
    print(f"Test samples     : {len(testset)}")
    return trainloader, testloader


# =============================================================================
# SECTION 5: Training Loop
# =============================================================================

def train(model, trainloader, optimizer, scheduler, lambda_val, epochs=15):
    """
    Trains the self-pruning network.

    Total Loss = CrossEntropyLoss + λ × SparsityLoss

    A higher λ means stronger penalty for keeping gates open,
    resulting in more aggressive pruning but potentially lower accuracy.
    This is the core accuracy vs efficiency trade-off.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "accuracy": [], "sparsity": []}

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            # How wrong are predictions?
            class_loss = criterion(outputs, labels)

            # How many gates are still open?
            sparse_loss = model.total_sparsity_loss()

            # Combined: balance accuracy vs pruning
            loss = class_loss + lambda_val * sparse_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        epoch_acc = 100.0 * correct / total
        epoch_sparsity = model.calculate_sparsity_percent()
        avg_loss = total_loss / len(trainloader)

        history["loss"].append(avg_loss)
        history["accuracy"].append(epoch_acc)
        history["sparsity"].append(epoch_sparsity)

        print(
            f"  Epoch [{epoch+1:02d}/{epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {epoch_acc:.1f}% | "
            f"Sparsity: {epoch_sparsity:.1f}%"
        )

    return history


# =============================================================================
# SECTION 6: Evaluation
# =============================================================================

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
    return 100.0 * correct / total


# =============================================================================
# SECTION 7: Baseline (no pruning) for comparison
# =============================================================================

def run_baseline(trainloader, testloader):
    """
    Standard nn.Linear network with no gates — our reference point.
    Pruned models should stay close to this accuracy while being
    significantly smaller.
    """
    print("\n" + "=" * 55)
    print("  Baseline: Standard Network (No Pruning)")
    print("=" * 55)

    baseline = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 10)
    ).to(device)

    optimizer = optim.Adam(baseline.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        baseline.train()
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(baseline(images), labels)
            loss.backward()
            optimizer.step()
            _, predicted = baseline(images).max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1:02d}/15] | Train Acc: {100.*correct/total:.1f}%")

    baseline.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = baseline(images).max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    print(f"\n  Baseline Test Accuracy: {acc:.2f}% | Sparsity: 0.00%")
    return acc


# =============================================================================
# SECTION 8: Plotting
# =============================================================================

def plot_gate_distribution(model, lambda_val, test_accuracy, sparsity):
    all_gates = model.get_all_gates().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Gate Distribution — λ={lambda_val} | "
        f"Test Accuracy: {test_accuracy:.1f}% | "
        f"Sparsity: {sparsity:.1f}%",
        fontsize=13, fontweight="bold"
    )

    axes[0].hist(all_gates, bins=100, color="steelblue",
                 edgecolor="black", alpha=0.8)
    axes[0].axvline(x=0.01, color="red", linestyle="--",
                    linewidth=1.5, label="Pruning threshold (0.01)")
    axes[0].set_title("Full Gate Distribution")
    axes[0].set_xlabel("Gate Value (0=pruned, 1=active)")
    axes[0].set_ylabel("Number of Gates")
    axes[0].legend()

    active_gates = all_gates[all_gates >= 0.01]
    axes[1].hist(active_gates, bins=80, color="darkorange",
                 edgecolor="black", alpha=0.8)
    axes[1].set_title(f"Active Gates Only ({len(active_gates):,} remaining)")
    axes[1].set_xlabel("Gate Value")
    axes[1].set_ylabel("Number of Gates")

    plt.tight_layout()
    fname = f"results/gate_distribution_lambda_{lambda_val}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Plot saved → {fname}")


def plot_training_curves(all_histories, lambda_values):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves Across Lambda Values",
                 fontsize=13, fontweight="bold")
    colors = ["steelblue", "darkorange", "green"]

    for i, (history, lam) in enumerate(zip(all_histories, lambda_values)):
        epochs = range(1, len(history["accuracy"]) + 1)
        axes[0].plot(epochs, history["accuracy"], color=colors[i],
                     label=f"λ={lam}", linewidth=2)
        axes[1].plot(epochs, history["sparsity"], color=colors[i],
                     label=f"λ={lam}", linewidth=2)

    axes[0].set_title("Training Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Sparsity Level During Training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sparsity (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    plt.close()
    print("  Training curves → results/training_curves.png")


# =============================================================================
# SECTION 9: Main
# =============================================================================

def main():
    print("=" * 55)
    print("  Self-Pruning Neural Network — CIFAR-10")
    print("  Tredence AI Engineering Internship")
    print("=" * 55)

    trainloader, testloader = load_data(batch_size=256)

    # --- Baseline ---
    baseline_acc = run_baseline(trainloader, testloader)

    # --- Three lambda values ---
    # NOTE: Lambda values must be large enough relative to the number of gates
    # With ~800K gates, we need λ in the range 0.001 to 0.1 to see real pruning
    lambda_values = [0.001, 0.01, 0.1]

    results = []
    all_histories = []
    best_model = None
    best_accuracy = 0.0
    best_lambda = None

    for lambda_val in lambda_values:
        print(f"\n{'='*55}")
        print(f"  Experiment: λ = {lambda_val}")
        print(f"{'='*55}")

        model = SelfPruningNetwork().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )

        history = train(model, trainloader, optimizer, scheduler,
                        lambda_val, epochs=15)
        all_histories.append(history)

        test_accuracy = evaluate(model, testloader)
        sparsity = model.calculate_sparsity_percent()

        print(f"\n  Results for λ = {lambda_val}:")
        print(f"     Test Accuracy  : {test_accuracy:.2f}%")
        print(f"     Sparsity Level : {sparsity:.2f}%")

        results.append({
            "lambda": lambda_val,
            "test_accuracy": test_accuracy,
            "sparsity": sparsity
        })

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_lambda = lambda_val
            best_model = model

        plot_gate_distribution(model, lambda_val, test_accuracy, sparsity)

    plot_training_curves(all_histories, lambda_values)

    # --- Final Summary Table ---
    print(f"\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<14} {'Test Accuracy':<18} {'Sparsity':<16} {'Accuracy Drop'}")
    print(f"  {'-'*58}")
    print(f"  {'Baseline':<14} {baseline_acc:<18.2f}% {'0.00%':<16} —")
    for r in results:
        drop = baseline_acc - r["test_accuracy"]
        print(
            f"  {str(r['lambda']):<14} "
            f"{r['test_accuracy']:<18.2f}% "
            f"{r['sparsity']:<16.2f}% "
            f"-{drop:.2f}%"
        )

    print(f"\n  Best Model: λ = {best_lambda} ({best_accuracy:.2f}% accuracy)")
    torch.save(
        best_model.state_dict(),
        f"results/best_model_lambda_{best_lambda}.pth"
    )
    print(f"  Saved → results/best_model_lambda_{best_lambda}.pth")
    print(f"\n  Done! Check the results/ folder for all plots.")


if __name__ == "__main__":
    main()