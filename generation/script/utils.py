import os
import torch


def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Loading Checkpoint: {checkpoint_path}.")
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('global_step', 0)
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0


def save_checkpoint(model, optimizer, global_step, checkpoint_path):
    checkpoint = {
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)


def linear_decay_with_warmup(step, max_learning_rate, warmup_steps, hold_steps, decay_steps, min_learning_rate=1e-8):
    if step < warmup_steps:
        lr_lambda = max_learning_rate * (step / warmup_steps)
    elif step < warmup_steps + hold_steps:
        lr_lambda = max_learning_rate
    else:
        offset = warmup_steps + hold_steps
        scale = 1 - (step - offset) / decay_steps
        lr_lambda = max(max_learning_rate * scale, min_learning_rate)
    return lr_lambda
