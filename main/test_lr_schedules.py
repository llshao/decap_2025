#!/usr/bin/env python3
"""
Test script to demonstrate different learning rate schedules.
This helps visualize how the learning rate changes over training updates.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

def test_exponential_schedule(initial_lr, anneal_factor, num_updates):
    """Test exponential learning rate decay."""
    lrs = []
    current_lr = initial_lr
    for update in range(1, num_updates + 1):
        lrs.append(current_lr)
        current_lr *= anneal_factor
    return lrs

def test_step_schedule(initial_lr, step_size, gamma, num_updates):
    """Test step-based learning rate decay."""
    lrs = []
    current_lr = initial_lr
    for update in range(1, num_updates + 1):
        lrs.append(current_lr)
        if update % step_size == 0:
            current_lr *= gamma
    return lrs

def test_cosine_schedule(initial_lr, num_updates, lr_min=1e-6):
    """Test cosine annealing learning rate schedule."""
    lrs = []
    for update in range(1, num_updates + 1):
        # Cosine annealing from initial_lr to lr_min
        progress = (update - 1) / (num_updates - 1)
        lr = lr_min + (initial_lr - lr_min) * (1 + np.cos(np.pi * progress)) / 2
        lrs.append(lr)
    return lrs

def test_linear_schedule(initial_lr, num_updates, warmup_steps=0, lr_min=1e-6):
    """Test linear learning rate schedule with optional warmup."""
    lrs = []
    for update in range(1, num_updates + 1):
        if update <= warmup_steps:
            # Warmup phase: linearly increase from 0 to initial_lr
            lr = initial_lr * (update / warmup_steps) if warmup_steps > 0 else initial_lr
        else:
            # Linear decay phase
            progress = (update - warmup_steps) / max(1, num_updates - warmup_steps)
            lr = initial_lr * (1.0 - progress * 0.9)  # Decay to 10% of initial LR
            lr = max(lr, lr_min)
        lrs.append(lr)
    return lrs

def test_cosine_warmup_schedule(initial_lr, step_size, cycle_mult, num_updates, lr_min=1e-6):
    """Test cosine annealing with warm restarts."""
    lrs = []
    current_lr = initial_lr
    T_0 = step_size
    T_mult = cycle_mult
    
    for update in range(1, num_updates + 1):
        # Calculate current cycle
        cycle = 0
        T_curr = T_0
        while update > T_curr:
            update -= T_curr
            cycle += 1
            T_curr = int(T_curr * T_mult)
        
        # Calculate position within current cycle
        progress = update / T_curr
        lr = lr_min + (initial_lr - lr_min) * (1 + np.cos(np.pi * progress)) / 2
        lrs.append(lr)
    
    return lrs

def test_onecycle_schedule(initial_lr, max_lr, num_updates, pct_start=0.3):
    """Test one-cycle learning rate schedule."""
    lrs = []
    warmup_steps = int(num_updates * pct_start)
    anneal_steps = num_updates - warmup_steps
    
    for update in range(1, num_updates + 1):
        if update <= warmup_steps:
            # Warmup phase: linearly increase from initial_lr to max_lr
            progress = update / warmup_steps
            lr = initial_lr + (max_lr - initial_lr) * progress
        else:
            # Anneal phase: cosine decrease from max_lr to initial_lr/100
            progress = (update - warmup_steps) / anneal_steps
            lr = initial_lr/100 + (max_lr - initial_lr/100) * (1 + np.cos(np.pi * progress)) / 2
        lrs.append(lr)
    
    return lrs

def test_cyclic_schedule(initial_lr, max_lr, step_size, num_updates):
    """Test cyclic learning rate schedule."""
    lrs = []
    current_lr = initial_lr
    direction = 1  # 1 for increasing, -1 for decreasing
    step_count = 0
    
    for update in range(1, num_updates + 1):
        lrs.append(current_lr)
        
        if direction == 1:
            current_lr += (max_lr - initial_lr) / step_size
            if current_lr >= max_lr:
                current_lr = max_lr
                direction = -1
                step_count = 0
        else:
            current_lr -= (max_lr - initial_lr) / step_size
            if current_lr <= initial_lr:
                current_lr = initial_lr
                direction = 1
                step_count = 0
        
        step_count += 1
    
    return lrs

def plot_schedules(args):
    """Plot all learning rate schedules for comparison."""
    num_updates = args.num_updates
    initial_lr = args.learning_rate
    
    # Generate schedules
    exp_lrs = test_exponential_schedule(initial_lr, args.anneal_lr_value, num_updates)
    step_lrs = test_step_schedule(initial_lr, args.lr_step_size, args.lr_gamma, num_updates)
    cosine_lrs = test_cosine_schedule(initial_lr, num_updates, args.lr_min)
    cosine_warmup_lrs = test_cosine_warmup_schedule(initial_lr, args.lr_step_size, args.lr_cycle_mult, num_updates, args.lr_min)
    onecycle_lrs = test_onecycle_schedule(initial_lr, args.lr_max, num_updates)
    cyclic_lrs = test_cyclic_schedule(args.lr_min, args.lr_max, args.lr_step_size, num_updates)
    linear_lrs = test_linear_schedule(initial_lr, num_updates, args.lr_warmup_steps, args.lr_min)
    
    # Create plot
    plt.figure(figsize=(16, 12))
    updates = range(1, num_updates + 1)
    
    plt.subplot(3, 3, 1)
    plt.plot(updates, exp_lrs, 'b-', linewidth=2, label=f'Exponential (γ={args.anneal_lr_value})')
    plt.title('Exponential Learning Rate Schedule')
    plt.xlabel('Update Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 3, 2)
    plt.plot(updates, step_lrs, 'r-', linewidth=2, label=f'Step (step={args.lr_step_size}, γ={args.lr_gamma})')
    plt.title('Step Learning Rate Schedule')
    plt.xlabel('Update Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 3, 3)
    plt.plot(updates, cosine_lrs, 'g-', linewidth=2, label=f'Cosine (min={args.lr_min:.1e})')
    plt.title('Cosine Annealing Learning Rate Schedule')
    plt.xlabel('Update Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 3, 4)
    plt.plot(updates, cosine_warmup_lrs, 'c-', linewidth=2, label=f'Cosine Warmup (T₀={args.lr_step_size})')
    plt.title('Cosine Annealing with Warm Restarts')
    plt.xlabel('Update Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 3, 5)
    plt.plot(updates, onecycle_lrs, 'm-', linewidth=2, label=f'OneCycle (max={args.lr_max:.1e})')
    plt.title('OneCycle Learning Rate Schedule')
    plt.xlabel('Update Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 3, 6)
    plt.plot(updates, cyclic_lrs, 'y-', linewidth=2, label=f'Cyclic (max={args.lr_max:.1e})')
    plt.title('Cyclic Learning Rate Schedule')
    plt.xlabel('Update Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 3, 7)
    plt.plot(updates, linear_lrs, 'orange', linewidth=2, label=f'Linear (warmup={args.lr_warmup_steps})')
    plt.title('Linear Learning Rate Schedule')
    plt.xlabel('Update Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add plateau simulation (simplified)
    plt.subplot(3, 3, 8)
    plateau_lrs = [initial_lr] * num_updates
    # Simulate some plateau reductions
    for i in range(100, num_updates, 150):
        plateau_lrs[i:] = [lr * args.lr_factor for lr in plateau_lrs[i:]]
    plt.plot(updates, plateau_lrs, 'purple', linewidth=2, label=f'Plateau (patience={args.lr_patience})')
    plt.title('ReduceLROnPlateau Schedule')
    plt.xlabel('Update Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add restart simulation
    plt.subplot(3, 3, 9)
    restart_lrs = test_cosine_warmup_schedule(initial_lr, args.lr_step_size, args.lr_cycle_mult, num_updates, args.lr_min)
    plt.plot(updates, restart_lrs, 'brown', linewidth=2, label=f'Restart (T₀={args.lr_step_size})')
    plt.title('Cosine Annealing Restart Schedule')
    plt.xlabel('Update Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_rate_schedules.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nLearning Rate Schedule Summary (over {num_updates} updates):")
    print(f"Initial LR: {initial_lr:.2e}")
    print(f"Exponential: {exp_lrs[-1]:.2e} (final), {np.mean(exp_lrs):.2e} (avg)")
    print(f"Step: {step_lrs[-1]:.2e} (final), {np.mean(step_lrs):.2e} (avg)")
    print(f"Cosine: {cosine_lrs[-1]:.2e} (final), {np.mean(cosine_lrs):.2e} (avg)")
    print(f"Cosine Warmup: {cosine_warmup_lrs[-1]:.2e} (final), {np.mean(cosine_warmup_lrs):.2e} (avg)")
    print(f"OneCycle: {onecycle_lrs[-1]:.2e} (final), {np.mean(onecycle_lrs):.2e} (avg)")
    print(f"Cyclic: {cyclic_lrs[-1]:.2e} (final), {np.mean(cyclic_lrs):.2e} (avg)")
    print(f"Linear: {linear_lrs[-1]:.2e} (final), {np.mean(linear_lrs):.2e} (avg)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test different learning rate schedules")
    parser.add_argument("--num-updates", type=int, default=600, help="Number of training updates")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Initial learning rate")
    parser.add_argument("--anneal-lr-value", type=float, default=0.9997, help="Exponential decay factor")
    parser.add_argument("--lr-step-size", type=int, default=100, help="Step size for step decay")
    parser.add_argument("--lr-gamma", type=float, default=0.9, help="Gamma for step decay")
    parser.add_argument("--lr-warmup-steps", type=int, default=50, help="Warmup steps for linear schedule")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--lr-max", type=float, default=1e-3, help="Maximum learning rate for cyclic and onecycle schedules")
    parser.add_argument("--lr-patience", type=int, default=10, help="Patience for plateau scheduler")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Factor for plateau scheduler reduction")
    parser.add_argument("--lr-cycles", type=int, default=3, help="Number of cycles for cyclic and restart schedulers")
    parser.add_argument("--lr-cycle-mult", type=float, default=2.0, help="Cycle length multiplier for restart scheduler")
    
    args = parser.parse_args()
    plot_schedules(args) 