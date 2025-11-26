import matplotlib.pyplot as plt
import numpy as np

# === FILL THESE WITH YOUR VALUES ===
# Example:
# train_losses = [...]
# val_losses = [...]
# train_accs = [...]
# val_accs = [...]

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = np.arange(1, len(train_losses) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # LOSS
    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # ACCURACY
    axes[1].plot(epochs, train_accs, label="Train Acc")
    axes[1].plot(epochs, val_accs, label="Val Acc")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("Figure 1 – Training & Validation Curves")
    plt.tight_layout()
    plt.show()


# CALL THE FUNCTION AFTER YOU FILL YOUR LISTS
# plot_training_curves(train_losses, val_losses, train_accs, val_accs)
