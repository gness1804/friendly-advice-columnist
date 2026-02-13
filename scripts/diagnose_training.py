#!/usr/bin/env python3
"""
Diagnostic script to analyze training checkpoints and loss curves.

Inspects:
1. Available checkpoints and their metadata
2. Loss progression across training steps
3. Gradient statistics if available
4. Model parameter counts
5. Training stability metrics
"""

import os
import torch
from pathlib import Path

CHECKPOINT_DIR = "checkpoints"


def format_number(n):
    """Format large numbers with commas."""
    if isinstance(n, float):
        return f"{n:.6f}"
    return f"{n:,}"


def load_checkpoint(checkpoint_path):
    """Load a checkpoint and extract metadata."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint
    except Exception as e:
        print(f"❌ Error loading {checkpoint_path}: {e}")
        return None


def get_checkpoint_info(checkpoint):
    """Extract key info from a checkpoint."""
    info = {
        "step": checkpoint.get("step", "unknown"),
        "has_optimizer": "optimizer_state_dict" in checkpoint,
        "has_model": "model_state_dict" in checkpoint,
    }

    if "hyperparameters" in checkpoint:
        hp = checkpoint["hyperparameters"]
        info["hyperparameters"] = {
            "learning_rate": hp.get("learning_rate"),
            "batch_size": hp.get("batch_size"),
            "block_size": hp.get("block_size"),
            "n_layer": hp.get("n_layer"),
            "n_embd": hp.get("n_embd"),
            "n_head": hp.get("n_head"),
        }

    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        info["num_params"] = sum(
            p.numel() for p in model_state.values() if isinstance(p, torch.Tensor)
        )

    return info


def list_checkpoints():
    """List all available checkpoints sorted by step."""
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Checkpoint directory not found: {CHECKPOINT_DIR}")
        return []

    checkpoints = sorted(Path(CHECKPOINT_DIR).glob("checkpoint_*.pt"))
    if not checkpoints:
        print(f"❌ No checkpoints found in {CHECKPOINT_DIR}")
        return []

    print(f"\n📁 Found {len(checkpoints)} checkpoints:\n")
    print(f"{'Step':<10} {'File':<70} {'Size':<10}")
    print("-" * 90)

    ckpts_with_step = []
    for ckpt_path in checkpoints:
        # Extract step from filename: checkpoint_..._step000500_...
        filename = ckpt_path.name
        try:
            step_part = [p for p in filename.split("_") if p.startswith("step")][0]
            step = int(step_part[4:])  # Remove 'step' prefix
            ckpts_with_step.append((step, ckpt_path))
        except (IndexError, ValueError):
            ckpts_with_step.append((float("inf"), ckpt_path))

    # Sort by step
    ckpts_with_step.sort(key=lambda x: x[0])

    for step, ckpt_path in ckpts_with_step:
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        step_str = str(int(step)) if step != float("inf") else "unknown"
        print(
            f"{step_str:<10} {ckpt_path.name:<70} {size_mb:>6.1f} MB"
        )

    return ckpts_with_step


def analyze_checkpoints(ckpts_with_step):
    """Analyze checkpoint details."""
    if not ckpts_with_step:
        return

    print("\n📊 Checkpoint Analysis:\n")
    print(f"{'Step':<10} {'Params':<15} {'LR':<12} {'Batch':<8} {'Block':<8} {'Layers':<8}")
    print("-" * 80)

    for step, ckpt_path in ckpts_with_step:
        checkpoint = load_checkpoint(ckpt_path)
        if not checkpoint:
            continue

        info = get_checkpoint_info(checkpoint)
        step_str = str(int(step)) if step != float("inf") else "unknown"

        hp = info.get("hyperparameters", {})
        lr = hp.get("learning_rate")
        batch = hp.get("batch_size")
        block = hp.get("block_size")
        layers = hp.get("n_layer")
        params = info.get("num_params", 0)

        lr_str = f"{lr:.2e}" if lr else "N/A"
        batch_str = str(batch) if batch else "N/A"
        block_str = str(block) if block else "N/A"
        layers_str = str(layers) if layers else "N/A"
        params_str = f"{params:,}" if params else "N/A"

        print(f"{step_str:<10} {params_str:<15} {lr_str:<12} {batch_str:<8} {block_str:<8} {layers_str:<8}")


def extract_loss_from_output(output_file):
    """
    Parse training output file to extract loss progression.

    Looks for lines like:
    step 500/5000 (10.0%): train loss 6.9544, val loss 7.1291 | LR: 2.00e-05
    """
    losses = {"steps": [], "train": [], "val": [], "lr": []}

    if not os.path.exists(output_file):
        return losses

    try:
        with open(output_file, "r") as f:
            for line in f:
                # Look for step/loss lines
                if "step" in line and "train loss" in line and "val loss" in line:
                    try:
                        # Parse multiple formats:
                        # Format 1: step 500/5000 (10.0%): train loss 6.9544, val loss 7.1291 | LR: 2.00e-05
                        # Format 2: step 2000/5000 (0.0%): train loss 5.2409, val loss 5.1862 | 13.9s (0.00 steps/sec)
                        
                        # Extract step number
                        step_match = line.split("step")[1].split("/")[0].strip()
                        step_num = int(step_match)
                        
                        # Extract train loss
                        train_idx = line.find("train loss")
                        if train_idx == -1:
                            continue
                        train_str = line[train_idx + 11:].split(",")[0].strip()
                        train_loss = float(train_str)
                        
                        # Extract val loss
                        val_idx = line.find("val loss")
                        if val_idx == -1:
                            continue
                        val_str = line[val_idx + 9:].split("|")[0].strip()
                        val_loss = float(val_str)
                        
                        # Extract LR if available
                        lr = 0.0
                        if "LR:" in line:
                            lr_idx = line.find("LR:")
                            lr_str = line[lr_idx + 4:].split("|")[0].strip()
                            try:
                                lr = float(lr_str)
                            except ValueError:
                                lr = 0.0
                        
                        losses["steps"].append(step_num)
                        losses["train"].append(train_loss)
                        losses["val"].append(val_loss)
                        losses["lr"].append(lr)

                    except (ValueError, IndexError):
                        continue

    except Exception as e:
        print(f"⚠️  Error reading output file: {e}")

    return losses


def find_output_file():
    """Find the most recent output file."""
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        return None

    files = sorted(
        Path(output_dir).glob("build_llm_output_*.txt"), key=os.path.getmtime, reverse=True
    )
    return str(files[0]) if files else None


def plot_loss_curves(losses):
    """Create ASCII plot of loss curves."""
    if not losses["steps"]:
        print("❌ No loss data found")
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Train vs Val loss
        axes[0].plot(losses["steps"], losses["train"], label="Train Loss", marker="o", markersize=4)
        axes[0].plot(losses["steps"], losses["val"], label="Val Loss", marker="s", markersize=4)
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss Progression")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Highlight the catastrophic spike if present
        train_array = np.array(losses["train"])
        
        # Find steps with large loss jumps
        if len(train_array) > 1:
            train_deltas = np.abs(np.diff(train_array))
            spike_threshold = np.median(train_deltas) + 2 * np.std(train_deltas)
            spike_indices = np.where(train_deltas > spike_threshold)[0]
            
            if len(spike_indices) > 0:
                for spike_idx in spike_indices:
                    axes[0].axvline(
                        losses["steps"][spike_idx],
                        color="red",
                        linestyle="--",
                        alpha=0.5,
                        linewidth=2,
                    )
                axes[0].text(
                    0.98,
                    0.05,
                    f"⚠️ Spikes detected at steps: {[losses['steps'][i] for i in spike_indices]}",
                    transform=axes[0].transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
                )

        # Plot 2: Learning rate schedule
        axes[1].plot(losses["steps"], losses["lr"], label="Learning Rate", color="green")
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Learning Rate Schedule")
        axes[1].set_yscale("log")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = "loss_curves_diagnostic.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Loss curves saved to: {output_path}")
        plt.close()

    except ImportError:
        print("⚠️  matplotlib not installed. Showing text-based summary instead.\n")
        print_loss_summary(losses)


def print_loss_summary(losses):
    """Print ASCII-based loss summary."""
    if not losses["steps"]:
        return

    print("\n📈 Loss Summary:\n")
    print(f"{'Step':<10} {'Train Loss':<15} {'Val Loss':<15} {'Δ Train':<12} {'Δ Val':<12} {'LR':<12}")
    print("-" * 90)

    prev_train = losses["train"][0]
    prev_val = losses["val"][0]

    for i, step in enumerate(losses["steps"]):
        train = losses["train"][i]
        val = losses["val"][i]
        lr = losses["lr"][i]

        delta_train = train - prev_train
        delta_val = val - prev_val

        delta_train_str = f"{delta_train:+.4f}"
        delta_val_str = f"{delta_val:+.4f}"

        # Highlight large spikes
        marker = "⚠️ " if abs(delta_train) > 1.0 else "  "

        print(
            f"{step:<10} {train:<15.4f} {val:<15.4f} {delta_train_str:<12} {delta_val_str:<12} {lr:<12.2e} {marker}"
        )

        prev_train = train
        prev_val = val


def analyze_loss_stability(losses):
    """Analyze loss stability metrics."""
    if not losses["train"] or len(losses["train"]) < 2:
        return

    import numpy as np

    train = np.array(losses["train"])
    val = np.array(losses["val"])

    print("\n📊 Loss Stability Analysis:\n")

    # Catastrophic spike detection
    train_deltas = np.abs(np.diff(train))
    median_delta = np.median(train_deltas)
    max_delta = np.max(train_deltas)
    max_delta_idx = np.argmax(train_deltas)

    print(f"Train Loss Range: {train.min():.4f} to {train.max():.4f}")
    print(f"Val Loss Range: {val.min():.4f} to {val.max():.4f}")
    print(f"\nMedian step-to-step change: {median_delta:.4f}")
    print(f"Max step-to-step change: {max_delta:.4f} (at step {losses['steps'][max_delta_idx]})")

    # Divergence: train vs val
    divergence = train - val
    print(f"\nDivergence (Train - Val): {divergence.mean():.4f} (mean), {divergence.std():.4f} (std)")

    # Post-spike analysis (if spike detected)
    if max_delta > median_delta * 3:
        spike_step = losses["steps"][max_delta_idx]
        pre_spike_train = train[max(0, max_delta_idx - 1)]
        post_spike_train = train[max_delta_idx + 1]
        print("\n⚠️  SPIKE DETECTED:")
        print(f"   Step {spike_step}: Train loss jumped from {pre_spike_train:.4f} → {post_spike_train:.4f}")
        print("   Recovery pattern: ", end="")

        # Look at next 5 steps
        recovery_steps = train[max_delta_idx + 1 : min(max_delta_idx + 6, len(train))]
        if len(recovery_steps) > 1:
            recovering = recovery_steps[-1] < post_spike_train
            print(f"{'✅ Recovering' if recovering else '❌ Not recovering'}")
            print(f"   Trajectory: {' → '.join(f'{x:.4f}' for x in recovery_steps)}")
        else:
            print("   (not enough data after spike)")


def main():
    """Main diagnostic routine."""
    print("\n" + "=" * 90)
    print("🔍 Training Diagnostic Report")
    print("=" * 90)

    # Step 1: List checkpoints
    print("\n[1/4] Scanning checkpoints...")
    ckpts_with_step = list_checkpoints()

    if not ckpts_with_step:
        print("❌ No checkpoints to analyze")
        return

    # Step 2: Analyze checkpoint details
    print("\n[2/4] Analyzing checkpoint metadata...")
    analyze_checkpoints(ckpts_with_step)

    # Step 3: Find and parse output file
    print("\n[3/4] Extracting loss data from training output...")
    output_file = find_output_file()

    if output_file:
        print(f"📄 Found output file: {output_file}")
        losses = extract_loss_from_output(output_file)
        if losses["steps"]:
            print(f"   Extracted {len(losses['steps'])} loss measurements")
        else:
            print("   ⚠️  Could not parse loss data from output file")
    else:
        print("⚠️  No output file found in 'outputs/' directory")
        losses = {"steps": [], "train": [], "val": [], "lr": []}

    # Step 4: Analyze stability and plot
    print("\n[4/4] Analyzing loss stability...")
    if losses["steps"]:
        analyze_loss_stability(losses)
        print("\n   Generating visualization...")
        plot_loss_curves(losses)
    else:
        print("   ❌ No loss data available for analysis")

    print("\n" + "=" * 90)
    print("✅ Diagnostic report complete")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
