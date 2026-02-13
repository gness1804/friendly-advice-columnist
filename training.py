"""
Unified Training Script - Fresh and Resume Training
Combines training_legacy.py and training_resume_legacy.py into a single script.
Detects mode based on CHECKPOINT_PATH environment variable.
"""

import os
import sys
import time
import torch
from datetime import datetime

# Helper module imports
from training.config import load_config_fresh_from_env, load_config_from_checkpoint, config_to_dict, _select_device
from training.io_utils import (
    setup_stdout_capture,
    get_data_source_name,
    get_model_name,
    generate_output_filename,
    write_output_file,
)
from training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    create_checkpoint_log_file,
)
from training.data import load_raw_text, prepare_data_and_tokenizer, get_batch

# ============================================================================
# UTILITIES
# ============================================================================


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string (hours, minutes, seconds)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or len(parts) == 0:
        parts.append(f"{secs}s")

    return " ".join(parts)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main training entry point with fresh/resume detection."""
    
    # Step 1: Detect fresh vs resume mode
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", None)
    is_resume = checkpoint_path is not None
    
    # Step 2: Load config
    if is_resume:
        if not checkpoint_path:
            print("❌ Error: CHECKPOINT_PATH must be set for resume training")
            sys.exit(1)
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, device="cpu")
        config = load_config_from_checkpoint(checkpoint)
        print(f"✅ Resumed from step {config.start_step}")
    else:
        config = load_config_fresh_from_env()
        checkpoint = None

    # Set random seed for reproducibility
    torch.manual_seed(1337)
    
    # Step 3: Select device
    device = _select_device(config.device)
    config.device = device
    
    if device == "cuda":
        print("✅ Using NVIDIA GPU (CUDA)")
    elif device == "mps":
        print("✅ Using Apple Silicon GPU (Metal Performance Shaders)")
    else:
        print("⚠️  Using CPU (slow) - consider using MPS if available")

    # Step 4: Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    # Step 5: Set up stdout capture
    tee_output, original_stdout = setup_stdout_capture(config.enable_output_to_file)

    # Step 6: Load data
    print(f"Loading training data from: {config.training_data_source}")
    raw_text = load_raw_text(config.training_data_source)

    # Step 7: Initialize model + tokenizer + data
    if config.model_type == "gpt2":
        # GPT-2 model path
        from models.gpt2_wrapper import GPT2Wrapper

        print(f"🤖 Using GPT-2 model: {config.gpt2_model_name}")
        if config.use_lora:
            print("🔧 Using LoRA for efficient fine-tuning")
            lora_rank = int(os.environ.get("LORA_RANK", "16"))
            lora_alpha = float(os.environ.get("LORA_ALPHA", "32.0"))
            lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.0"))
            print(f"   LoRA rank: {lora_rank}, alpha: {lora_alpha}, dropout: {lora_dropout}")
        else:
            lora_rank = None
            lora_alpha = None
            lora_dropout = 0.0

        model = GPT2Wrapper(
            model_name=config.gpt2_model_name,
            use_lora=config.use_lora,
            lora_rank=lora_rank if config.use_lora else None,
            lora_alpha=lora_alpha if config.use_lora else None,
            lora_dropout=lora_dropout if config.use_lora else None,
            device=device,
        )

        # Use GPT-2's tokenizer
        encode = model.encode
        decode = model.decode
        vocab_size = model.get_vocab_size()

        # Encode dataset
        data = torch.tensor(encode(raw_text), dtype=torch.long)

        # Split data
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]

        # Verify block_size vs max position embeddings
        gpt2_config = model.model.config
        max_pos = gpt2_config.max_position_embeddings
        if config.block_size > max_pos:
            print(f"⚠️  block_size ({config.block_size}) > GPT-2 max_pos ({max_pos}), using {max_pos}")
            config.block_size = max_pos

        model.to(device)

    elif config.model_type == "from_scratch":
        # From-scratch model
        from models.bigram_lm_v2 import BigramLanguageModel
        from models.bigram_lm_v2_lora import BigramLanguageModelLoRA

        # Use helper to prepare data and tokenization
        encode, decode, data, train_data, val_data, vocab_size = prepare_data_and_tokenizer(
            config, raw_text, config.model_type
        )

        if config.use_lora:
            print("🔧 Using LoRA for efficient fine-tuning")
            lora_rank = int(os.environ.get("LORA_RANK", "16"))
            lora_alpha = float(os.environ.get("LORA_ALPHA", "32.0"))
            lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.0"))
            print(f"   LoRA rank: {lora_rank}, alpha: {lora_alpha}, dropout: {lora_dropout}")
            
            model = BigramLanguageModelLoRA(
                vocab_size=vocab_size,
                n_embd=config.n_embd,
                block_size=config.block_size,
                device=device,
                dropout=config.dropout,
                n_head=config.n_head,
                n_layer=config.n_layer,
                use_lora=True,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        else:
            model = BigramLanguageModel(
                vocab_size=vocab_size,
                n_embd=config.n_embd,
                block_size=config.block_size,
                device=device,
                dropout=config.dropout,
                n_head=config.n_head,
                n_layer=config.n_layer,
            )

        model.to(device)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}. Use 'gpt2' or 'from_scratch'")

    # Print parameter statistics
    if config.model_type == "gpt2":
        param_info = model.get_parameter_info()
        print("📊 Parameter Statistics:")
        print(f"   Model: {param_info.get('model_name', config.gpt2_model_name)}")
        print(f"   Total parameters: {param_info['total']:,}")
        if config.use_lora:
            print(f"   Trainable (total): {param_info['trainable']:,}")
            if "lora_only_params" in param_info:
                print(f"     - LoRA adapters: {param_info['lora_only_params']:,} ({param_info.get('lora_only_percentage', 0):.2f}%)")
            print(f"   Frozen (base model): {param_info['frozen']:,}")
            lora_pct = param_info.get("lora_percentage", param_info.get("lora_only_percentage", 0))
            print(f"   💰 LoRA savings: Training only {lora_pct:.2f}% of parameters!")
        else:
            print(f"   Trainable parameters: {param_info['trainable']:,}")
            print("   💡 Tip: Use USE_LORA=True for efficient fine-tuning (90-99% fewer parameters)")
    elif config.use_lora:
        param_info = model.get_parameter_info()
        print("📊 Parameter Statistics:")
        print(f"   Total parameters: {param_info['total']:,}")
        print(f"   Trainable (total): {param_info['trainable']:,}")
        print(f"     - LoRA adapters: {param_info['lora_only_params']:,} ({param_info['lora_only_percentage']:.2f}%)")
        print(f"     - Embeddings: {param_info['embedding_params']:,} ({param_info['embedding_percentage']:.2f}%)")
        print(f"   Frozen (base model): {param_info['frozen']:,}")
        print(f"   💰 LoRA savings: Training only {param_info['lora_only_percentage']:.2f}% of parameters (LoRA-only)!")
        if param_info["embedding_percentage"] > 5.0:
            print(f"   ⚠️  Note: Embeddings are {param_info['embedding_percentage']:.1f}% of total (higher in small models)")
        if param_info["total"] < 5_000_000:
            print("   ⚠️  PERFORMANCE WARNING: LoRA may be SLOWER than full fine-tuning for small models!")
            print("       For models < 5M params, the LoRA overhead can outweigh benefits.")
            print("       Consider using full fine-tuning (USE_LORA=False) for better speed.")
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total model parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    print(f"Model device: {next(model.parameters()).device}")
    if device == "mps":
        print("✅ Model successfully moved to Apple Silicon GPU (MPS)")

    # Step 8: Load model state (if resuming)
    if is_resume and checkpoint:
        print("Loading model state from checkpoint...")
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ Model state loaded")

    # Step 9: Build/load optimizer
    if config.use_lora:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
        print(f"✅ Optimizer initialized with {len(trainable_params)} parameter groups (LoRA only)")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Load optimizer state (if resuming)
    if is_resume and checkpoint:
        print("Loading optimizer state from checkpoint...")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Ensure learning rate matches current config when resuming
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.learning_rate
        print(f"✅ Optimizer state loaded (LR overridden to {config.learning_rate:.2e})")

    # Step 10: Initialize learning rate scheduler (GPT-2 only)
    scheduler = None
    if config.model_type == "gpt2":
        if config.use_lr_warmup:
            warmup_steps = max(50, int(0.02 * config.training_steps))
            if warmup_steps > 0 and warmup_steps < config.training_steps:
                from torch.optim.lr_scheduler import LinearLR
                warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
                scheduler = warmup_scheduler
                print(f"📈 Learning rate schedule: WARMUP ({warmup_steps} steps) → CONSTANT")
                print(f"   LR: {config.learning_rate * 0.1:.2e} → {config.learning_rate:.2e} (then constant)")
            else:
                print("📈 Learning rate schedule: CONSTANT (warmup disabled - too few steps)")
                print(f"   LR: {config.learning_rate:.2e} (constant throughout training)")
        else:
            print("📈 Learning rate schedule: CONSTANT (warmup disabled via USE_LR_WARMUP=False)")
            print(f"   LR: {config.learning_rate:.2e} (constant throughout training)")

    # Collect hyperparameters for output file
    hyperparameters = config_to_dict(config)

    # Step 11: Training loop setup
    print(f"Starting training for {config.training_steps} steps...")
    print(f"Batch size: {config.batch_size}, Block size: {config.block_size}")
    print(f"Vocabulary size: {vocab_size:,} tokens")
    if config.enable_checkpoints:
        print(f"Checkpoints enabled: saving every {config.checkpoint_interval} steps to {config.checkpoint_dir}/")
    print("-" * 50)

    # Get model and source names for checkpoint naming
    model_name = get_model_name(model)
    source_name = get_data_source_name(config.training_data_source)

    # Initialize checkpoint log file
    checkpoint_log_file = None
    checkpoint_log_path = None
    if config.enable_checkpoints:
        checkpoint_log_file, checkpoint_log_path = create_checkpoint_log_file(
            config.log_dir,
            model_name,
            source_name,
            is_resume=is_resume,
            resume_step=config.start_step,
        )
        if checkpoint_log_file:
            print(f"📝 Checkpoint log: {checkpoint_log_path}")

    # Step 12: Training loop
    start_time = time.time()
    initial_train_loss = None
    last_checkpoint_train_loss = None

    @torch.no_grad()
    def estimate_loss():
        """Estimate loss on train and validation sets."""
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(config.eval_iters)
            for k in range(config.eval_iters):
                xb, yb = get_batch(split, train_data, val_data, config.batch_size, config.block_size, device)
                logits, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    for step in range(config.start_step, config.start_step + config.training_steps):
        # Evaluate loss periodically
        if (step - config.start_step) % config.eval_interval == 0:
            losses = estimate_loss()
            current_train_loss = losses["train"].item()

            # Store initial loss at first eval
            if initial_train_loss is None:
                initial_train_loss = current_train_loss
                last_checkpoint_train_loss = current_train_loss

            # Calculate loss changes
            net_loss_change_since_beginning = current_train_loss - initial_train_loss
            net_loss_change_since_last_checkpoint = current_train_loss - last_checkpoint_train_loss

            elapsed = time.time() - start_time
            steps_per_sec = (step - config.start_step) / elapsed if step > config.start_step else 0
            progress_pct = ((step - config.start_step) / config.training_steps) * 100
            current_lr = optimizer.param_groups[0]["lr"]
            
            print(
                f"step {step}/{config.start_step + config.training_steps} ({progress_pct:.1f}%): "
                f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | "
                f"LR: {current_lr:.2e} | {elapsed:.1f}s ({steps_per_sec:.2f} steps/sec) | "
                f"Net loss change since beginning: {net_loss_change_since_beginning:.4f} | "
                f"Net loss change since last checkpoint: {net_loss_change_since_last_checkpoint:.4f}"
            )

            # Save checkpoint if enabled
            if config.enable_checkpoints and (step - config.start_step) % config.checkpoint_interval == 0 and step > config.start_step:
                checkpoint_path = save_checkpoint(
                    step,
                    model,
                    optimizer,
                    config,
                    vocab_size,
                    config.block_size,
                    config.batch_size,
                    checkpoint_dir=config.checkpoint_dir,
                    model_name=model_name,
                    source_name=source_name,
                )
                if checkpoint_path:
                    print(f"   💾 Checkpoint saved: {checkpoint_path}")
                    last_checkpoint_train_loss = current_train_loss
                    if checkpoint_log_file and tee_output:
                        log_content = tee_output.getvalue()
                        checkpoint_log_file.write(f"\n{'='*80}\n")
                        checkpoint_log_file.write(
                            f"CHECKPOINT at step {step} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        )
                        checkpoint_log_file.write(f"{'='*80}\n")
                        checkpoint_log_file.write(log_content)
                        checkpoint_log_file.flush()

        # Print progress more frequently (every 25 steps)
        elif step > config.start_step and (step - config.start_step) % 25 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step - config.start_step) / elapsed
            progress_pct = ((step - config.start_step) / config.training_steps) * 100
            eta_seconds = ((config.training_steps) - (step - config.start_step)) / steps_per_sec if steps_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60
            print(
                f"step {step}/{config.start_step + config.training_steps} ({progress_pct:.1f}%) | "
                f"{steps_per_sec:.2f} steps/sec | ETA: {eta_minutes:.1f}m"
            )

        # Sample batch and train
        xb, yb = get_batch("train", train_data, val_data, config.batch_size, config.block_size, device)
        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        if config.model_type == "gpt2":
            if config.use_lora:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                if len(trainable_params) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update parameters
        optimizer.step()

        # Update learning rate schedule
        if scheduler is not None:
            scheduler.step()

    print("-" * 50)
    print(f"Training complete! Final loss: {loss.item():.4f}")
    total_time = time.time() - start_time
    print(f"Total training time: {format_time(total_time)} ({config.training_steps / total_time:.2f} steps/sec)")

    # Final evaluation
    final_losses = estimate_loss()
    print(f"Final eval - train loss {final_losses['train']:.4f}, val loss {final_losses['val']:.4f}")

    # Save final checkpoint
    if config.enable_checkpoints:
        final_checkpoint_path = save_checkpoint(
            config.start_step + config.training_steps,
            model,
            optimizer,
            config,
            vocab_size,
            config.block_size,
            config.batch_size,
            checkpoint_dir=config.checkpoint_dir,
            model_name=model_name,
            source_name=source_name,
        )
        if final_checkpoint_path:
            print(f"✅ Final model saved: {final_checkpoint_path}")
        if checkpoint_log_file and tee_output:
            log_content = tee_output.getvalue()
            checkpoint_log_file.write(f"\n{'='*80}\n")
            checkpoint_log_file.write(
                f"FINAL CHECKPOINT at step {config.start_step + config.training_steps} - "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            checkpoint_log_file.write(f"{'='*80}\n")
            checkpoint_log_file.write(log_content)
            checkpoint_log_file.write(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            checkpoint_log_file.close()
            print(f"📝 Checkpoint log saved: {checkpoint_log_path}")

    # Step 13: Text generation
    print("\nGenerating text...")
    print("=" * 50)

    if config.model_type == "gpt2":
        # Use in-domain prompt for GPT-2
        prompt = "QUESTION: "
        input_ids = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
        generated_tokens = model.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            temperature=config.generation_temperature,
            top_k=config.generation_top_k,
            do_sample=True,
        )
        generated_text = decode(generated_tokens[0].tolist())
    else:
        # From-scratch: start with zero token
        context = torch.zeros((1, 1), dtype=torch.long).to(device)
        generated_tokens = model.generate(context, max_new_tokens=config.max_new_tokens)
        generated_text = decode(generated_tokens[0].tolist())

    print(generated_text)
    print("=" * 50)

    # Step 14: Write output file
    if config.enable_output_to_file:
        # Restore stdout
        sys.stdout = original_stdout

        # Get captured output
        captured_output = tee_output.getvalue()

        # Generate output filename
        filename = generate_output_filename(
            model_name=model_name,
            source_name=source_name,
            vocab_size=vocab_size,
            training_steps=config.training_steps,
            test_mode=config.test_mode,
            use_lora=config.use_lora,
            lora_rank=int(os.environ.get("LORA_RANK", "16")) if config.use_lora else None,
            lora_alpha=float(os.environ.get("LORA_ALPHA", "32.0")) if config.use_lora else None,
            model_type=config.model_type,
            gpt2_model_name=config.gpt2_model_name if config.model_type == "gpt2" else None,
        )

        # Construct full output path
        output_path = os.path.join(config.output_dir, filename)

        # Write output file
        if write_output_file(output_path, hyperparameters, captured_output):
            print(f"\n✅ Output written to: {output_path}")
        else:
            print("\n⚠️  Failed to write output file")


if __name__ == "__main__":
    main()
