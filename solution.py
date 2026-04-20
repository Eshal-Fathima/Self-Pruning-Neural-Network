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
import matplotlib.gridspec as gridspec
import numpy as np

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)


# =============================================================================
# SECTION 2: PrunableLinear Layer
#
# This is the heart of the whole solution.
#
# How it works:
#   - Every weight has a corresponding "gate_score" (a learnable number)
#   - gate_score is passed through Sigmoid → gives a value between 0 and 1
#   - pruned_weight = weight × gate
#   - If gate → 0, that weight is effectively removed (pruned)
#   - Both weight and gate_score are nn.Parameter, so PyTorch's autograd
#     tracks gradients through BOTH during backpropagation automatically.
#     This means the optimizer updates both simultaneously.
# =============================================================================

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Standard weight — initialized with small random values (He init)
        # Shape: (out_features, in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=0.01)

        # Standard bias
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores — one per weight, same shape as weight
        # Initialized to 0 so sigmoid(0) = 0.5 (gates start half open)
        # Registered as nn.Parameter so optimizer updates them too
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        # Step 1: Sigmoid squishes gate_scores into (0, 1) range
        # This gives us our "gate" values
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: Element-wise multiply weights by gates
        # Weights with gate ≈ 0 are effectively pruned
        pruned_weights = self.weight * gates

        # Step 3: Standard linear transformation using pruned weights
        # F.linear computes: x @ pruned_weights.T + bias
        # Gradients flow through pruned_weights back to BOTH
        # self.weight and self.gate_scores automatically via autograd
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        """Returns current gate values (detached from computation graph)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)

    def sparsity_loss(self):
        """
        L1 norm of gate values for this layer.
        Since sigmoid output is always positive, L1 = sum of all gate values.
        Minimizing this encourages gates to go toward exactly 0 (pruned).
        """
        return torch.sigmoid(self.gate_scores).sum()


# =============================================================================
# SECTION 3: Self-Pruning Neural Network
#
# A 4-layer feed-forward network built entirely with PrunableLinear layers.
# CIFAR-10 images: 32×32 pixels × 3 channels = 3072 input features
# 10 output classes: airplane, car, bird, cat, deer,
#                    dog, frog, horse, ship, truck
# =============================================================================

class SelfPruningNetwork(nn.Module):
    def __init__(self):
        super(SelfPruningNetwork, self).__init__()

        self.layer1 = PrunableLinear(3072, 1024)
        self.layer2 = PrunableLinear(1024, 512)
        self.layer3 = PrunableLinear(512, 256)
        self.layer4 = PrunableLinear(256, 10)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Flatten image: (batch, 3, 32, 32) → (batch, 3072)
        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.bn1(self.layer1(x))))
        x = self.dropout(self.relu(self.bn2(self.layer2(x))))
        x = self.dropout(self.relu(self.bn3(self.layer3(x))))
        x = self.layer4(x)  # No activation on last layer (CrossEntropy handles it)

        return x

    def get_prunable_layers(self):
        """Returns all PrunableLinear layers for easy iteration."""
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def total_sparsity_loss(self):
        """
        Sum of L1 gate norms across ALL layers.
        This is the SparsityLoss term in:
            Total Loss = ClassificationLoss + λ × SparsityLoss
        """
        return sum(layer.sparsity_loss() for layer in self.get_prunable_layers())

    def get_all_gates(self):
        """Collect and flatten all gate values from all layers into one tensor."""
        all_gates = [
            layer.get_gates().flatten()
            for layer in self.get_prunable_layers()
        ]
        return torch.cat(all_gates)

    def calculate_sparsity_percent(self, threshold=1e-2):
        """
        Sparsity Level = % of gates below threshold (effectively pruned).
        Higher % = more pruned = sparser network.
        """
        all_gates = self.get_all_gates()
        pruned_count = (all_gates < threshold).sum().item()
        total_count = all_gates.numel()
        return 100.0 * pruned_count / total_count


# =============================================================================
# SECTION 4: Load CIFAR-10 Dataset
# =============================================================================

def load_data(batch_size=128):
    """
    Downloads and loads the CIFAR-10 dataset automatically.
    Applies standard normalization (mean=0.5, std=0.5 per channel).
    Training set: 50,000 images
    Test set:     10,000 images
    """
    # Data augmentation for training to improve generalization
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # No augmentation for test — just normalize
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

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"Training samples : {len(trainset)}")
    print(f"Test samples     : {len(testset)}")

    return trainloader, testloader


# =============================================================================
# SECTION 5: Training Loop
# =============================================================================

def train(model, trainloader, optimizer, scheduler, lambda_val, epochs=20):
    """
    Trains the self-pruning network.

    Loss formula:
        Total Loss = CrossEntropyLoss(predictions, labels)
                   + lambda_val × SparsityLoss(all gate values)

    Both the weights and gate_scores are updated by the optimizer.
    Over epochs, gates for unimportant weights are driven toward 0.
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

            # Forward pass
            outputs = model(images)

            # Classification loss — how wrong are the predictions?
            class_loss = criterion(outputs, labels)

            # Sparsity loss — how many gates are still open?
            sparse_loss = model.total_sparsity_loss()

            # Combined loss — network must balance accuracy vs pruning
            loss = class_loss + lambda_val * sparse_loss

            # Backpropagation — gradients flow through weights AND gate_scores
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        epoch_accuracy = 100.0 * correct / total
        epoch_sparsity = model.calculate_sparsity_percent()
        avg_loss = total_loss / len(trainloader)

        history["loss"].append(avg_loss)
        history["accuracy"].append(epoch_accuracy)
        history["sparsity"].append(epoch_sparsity)

        print(
            f"  Epoch [{epoch+1:02d}/{epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {epoch_accuracy:.1f}% | "
            f"Sparsity: {epoch_sparsity:.1f}%"
        )

    return history


# =============================================================================
# SECTION 6: Evaluation
# =============================================================================

def evaluate(model, testloader):
    """Evaluates model on test set. Returns accuracy percentage."""
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
# SECTION 7: Plotting
# =============================================================================

def plot_gate_distribution(model, lambda_val, test_accuracy, sparsity):
    """
    Plots the distribution of all gate values after training.

    A successful result shows:
      - A large spike near 0   → many gates pruned (closed)
      - A smaller cluster > 0  → the important active gates
    """
    all_gates = model.get_all_gates().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Gate Distribution — λ={lambda_val} | "
        f"Test Accuracy: {test_accuracy:.1f}% | "
        f"Sparsity: {sparsity:.1f}%",
        fontsize=13, fontweight="bold"
    )

    # Left plot: Full distribution
    axes[0].hist(all_gates, bins=100, color="steelblue",
                 edgecolor="black", alpha=0.8)
    axes[0].axvline(x=0.01, color="red", linestyle="--",
                    linewidth=1.5, label="Pruning threshold (0.01)")
    axes[0].set_title("Full Gate Distribution")
    axes[0].set_xlabel("Gate Value")
    axes[0].set_ylabel("Number of Gates")
    axes[0].legend()

    # Right plot: Zoomed into active gates (gate > 0.01)
    active_gates = all_gates[all_gates >= 0.01]
    axes[1].hist(active_gates, bins=80, color="darkorange",
                 edgecolor="black", alpha=0.8)
    axes[1].set_title(f"Active Gates Only ({len(active_gates):,} gates)")
    axes[1].set_xlabel("Gate Value")
    axes[1].set_ylabel("Number of Gates")

    plt.tight_layout()
    filename = f"results/gate_distribution_lambda_{lambda_val}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Plot saved → {filename}")


def plot_training_curves(all_histories, lambda_values):
    """Plots training accuracy and sparsity curves for all lambda values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves Across Lambda Values", fontsize=13,
                 fontweight="bold")

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
    print("  Training curves saved → results/training_curves.png")


# =============================================================================
# SECTION 8: Main — Run Experiments for 3 Lambda Values
# =============================================================================

def run_baseline(trainloader, testloader):
    """
    Runs a standard unpruned model (no gates, no sparsity loss).
    This gives us a reference accuracy to compare pruned models against.

    In a production context, this is the 'full model' that would normally
    be deployed — our pruned models should come close to this accuracy
    while being significantly smaller.
    """
    print("\n" + "=" * 60)
    print("  Baseline: Standard Network (No Pruning)")
    print("=" * 60)

    # Simple baseline using standard nn.Linear (no gates)
    baseline = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(1024, 512),  nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(512, 256),   nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 10)
    ).to(device)

    optimizer = optim.Adam(baseline.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        baseline.train()
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = baseline(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1:02d}/20] | "
                  f"Train Acc: {100.*correct/total:.1f}%")

    baseline.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = baseline(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    baseline_acc = 100.0 * correct / total
    print(f"\n  Baseline Test Accuracy: {baseline_acc:.2f}% (0% sparsity)")
    return baseline_acc


def main():
    print("=" * 60)
    print("  Self-Pruning Neural Network — CIFAR-10")
    print("  Tredence AI Engineering Internship — Case Study")
    print("=" * 60)

    # Load data once, reuse across all experiments
    trainloader, testloader = load_data(batch_size=128)

    # Step 1: Run baseline (unpruned) for comparison
    baseline_acc = run_baseline(trainloader, testloader)

    # Three lambda values: low, medium, high
    lambda_values = [0.0001, 0.001, 0.01]

    results = []
    all_histories = []
    best_model = None
    best_accuracy = 0.0
    best_lambda = None

    for lambda_val in lambda_values:
        print(f"\n{'='*60}")
        print(f"  Experiment: λ = {lambda_val}")
        print(f"{'='*60}")

        # Fresh model for each experiment
        model = SelfPruningNetwork().to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001,
                               weight_decay=1e-4)

        # Scheduler reduces learning rate every 7 epochs
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.5
        )

        # Train
        history = train(model, trainloader, optimizer, scheduler,
                        lambda_val, epochs=20)
        all_histories.append(history)

        # Evaluate
        test_accuracy = evaluate(model, testloader)
        sparsity = model.calculate_sparsity_percent()

        print(f"\n  ✅ Results for λ = {lambda_val}:")
        print(f"     Test Accuracy  : {test_accuracy:.2f}%")
        print(f"     Sparsity Level : {sparsity:.2f}%")

        results.append({
            "lambda": lambda_val,
            "test_accuracy": test_accuracy,
            "sparsity": sparsity
        })

        # Track best model (highest accuracy)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_lambda = lambda_val
            best_model = model

        # Plot gate distribution
        plot_gate_distribution(model, lambda_val, test_accuracy, sparsity)

    # Plot training curves for all lambdas
    plot_training_curves(all_histories, lambda_values)

    # ==========================================================================
    # SECTION 9: Final Results Summary
    # ==========================================================================

    print(f"\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Lambda':<12} {'Test Accuracy':<18} {'Sparsity':<16} {'Accuracy Drop'}")
    print(f"  {'-'*60}")
    # Baseline row
    print(f"  {'Baseline':<12} {baseline_acc:<18.2f}% {'0.00%':<16} '-'")
    for r in results:
        drop = baseline_acc - r['test_accuracy']
        print(
            f"  {r['lambda']:<12} "
            f"{r['test_accuracy']:<18.2f}% "
            f"{r['sparsity']:<16.2f}% "
            f"-{drop:.2f}%"
        )
    print(f"\n  Best Model : λ = {best_lambda} "
          f"with {best_accuracy:.2f}% accuracy")

    # Save best model weights
    torch.save(best_model.state_dict(),
               f"results/best_model_lambda_{best_lambda}.pth")
    print(f"  Best model saved → results/best_model_lambda_{best_lambda}.pth")

    # Also save a copy of best model's gate distribution
    best_sparsity = next(
        r["sparsity"] for r in results if r["lambda"] == best_lambda
    )
    plot_gate_distribution(
        best_model, f"{best_lambda}_BEST", best_accuracy, best_sparsity
    )

    print(f"\n{'='*60}")
    print("  All done! Check the results/ folder for plots.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()