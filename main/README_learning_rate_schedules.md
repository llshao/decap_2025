# Learning Rate Scheduling for PPO Training

This document describes the enhanced learning rate scheduling system implemented in the PPO training loop.

## Overview

The learning rate scheduling system provides multiple strategies to automatically adjust the learning rate during training, which can help improve convergence and final performance.

## Available Schedules

### 1. Exponential Decay (Default)
- **Strategy**: Multiplies learning rate by a factor after each update
- **Arguments**: `--anneal-lr-value <factor>` (default: 0.9997)
- **Behavior**: Gradual, smooth decay
- **Use case**: General purpose, stable training

### 2. Step Decay
- **Strategy**: Reduces learning rate by a factor every N steps
- **Arguments**: 
  - `--lr-step-size <steps>` (default: 100)
  - `--lr-gamma <factor>` (default: 0.9)
- **Behavior**: Sudden drops at regular intervals
- **Use case**: When you know specific milestones for learning rate reduction

### 3. Cosine Annealing
- **Strategy**: Smoothly decreases learning rate following a cosine curve
- **Arguments**: `--lr-min <min_lr>` (default: 1e-6)
- **Behavior**: Smooth, periodic-like decay
- **Use case**: Exploration-exploitation balance, avoiding local minima

### 4. Cosine Annealing with Warm Restarts
- **Strategy**: Cosine annealing with periodic restarts to higher learning rates
- **Arguments**: 
  - `--lr-step-size <T_0>` (default: 100) - Initial restart period
  - `--lr-cycle-mult <multiplier>` (default: 2.0) - Period multiplier after each restart
  - `--lr-min <min_lr>` (default: 1e-6)
- **Behavior**: Cyclic cosine decay with increasing restart periods
- **Use case**: Escaping local minima, exploration in later training

### 5. OneCycle Policy
- **Strategy**: Single cycle with warmup and anneal phases
- **Arguments**: `--lr-max <max_lr>` (default: 1e-2)
- **Behavior**: Linear warmup followed by cosine annealing
- **Use case**: Fast convergence, super-convergence training

### 6. ReduceLROnPlateau
- **Strategy**: Reduces learning rate when metric stops improving
- **Arguments**:
  - `--lr-patience <patience>` (default: 10) - Updates to wait before reducing
  - `--lr-factor <factor>` (default: 0.5) - Reduction factor
  - `--lr-min <min_lr>` (default: 1e-6)
- **Behavior**: Adaptive reduction based on performance
- **Use case**: When you want automatic adaptation to training progress

### 7. Cyclic Learning Rate
- **Strategy**: Cycles between base and maximum learning rates
- **Arguments**:
  - `--lr-max <max_lr>` (default: 1e-2)
  - `--lr-step-size <step_size>` (default: 100) - Half-cycle length
- **Behavior**: Triangular cycling between min and max learning rates
- **Use case**: Finding optimal learning rate range, escaping local minima

### 8. Cosine Annealing Restart
- **Strategy**: Cosine annealing with fixed-period restarts
- **Arguments**:
  - `--lr-step-size <T_0>` (default: 100) - Restart period
  - `--lr-cycle-mult <multiplier>` (default: 2.0) - Period multiplier
  - `--lr-min <min_lr>` (default: 1e-6)
- **Behavior**: Fixed-period cosine cycles with restarts
- **Use case**: Regular exploration phases, structured training

### 9. Linear Schedule
- **Strategy**: Linear decay with optional warmup phase
- **Arguments**:
  - `--lr-warmup-steps <steps>` (default: 0)
  - `--lr-min <min_lr>` (default: 1e-6)
- **Behavior**: Linear decrease, optionally starting from 0
- **Use case**: When you want gradual, predictable decay

## Usage Examples

### Basic Exponential Decay
```bash
python main_vec.py --case-idx 1 --num-envs 10 --lr-schedule exponential --anneal-lr-value 0.9995
```

### Step Decay
```bash
python main_vec.py --case-idx 1 --num-envs 10 --lr-schedule step --lr-step-size 200 --lr-gamma 0.8
```

### Cosine Annealing
```bash
python main_vec.py --case-idx 1 --num-envs 10 --lr-schedule cosine --lr-min 1e-7
```

### Linear Schedule with Warmup
```bash
python main_vec.py --case-idx 1 --num-envs 10 --lr-schedule linear --lr-warmup-steps 100 --lr-min 1e-6
```

### Cosine Annealing with Warm Restarts
```bash
python main_vec.py --case-idx 1 --num-envs 10 --lr-schedule cosine_warmup --lr-step-size 200 --lr-cycle-mult 1.5 --lr-min 1e-7
```

### OneCycle Policy
```bash
python main_vec.py --case-idx 1 --num-envs 10 --lr-schedule onecycle --lr-max 1e-3
```

### ReduceLROnPlateau
```bash
python main_vec.py --case-idx 1 --num-envs 10 --lr-schedule plateau --lr-patience 15 --lr-factor 0.7 --lr-min 1e-7
```

### Cyclic Learning Rate
```bash
python main_vec.py --case-idx 1 --num-envs 10 --lr-schedule cyclic --lr-max 1e-3 --lr-step-size 150
```

### Cosine Annealing Restart
```bash
python main_vec.py --case-idx 1 --num-envs 10 --lr-schedule restart --lr-step-size 300 --lr-cycle-mult 2.0 --lr-min 1e-7
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lr-schedule` | str | "exponential" | Learning rate scheduling strategy |
| `--anneal-lr-value` | float | 0.9997 | Exponential decay factor |
| `--lr-step-size` | int | 100 | Step size for step decay |
| `--lr-gamma` | float | 0.9 | Multiplicative factor for step decay |
| `--lr-warmup-steps` | int | 0 | Warmup steps for linear schedule |
| `--lr-min` | float | 1e-6 | Minimum learning rate |
| `--lr-max` | float | 1e-2 | Maximum learning rate for cyclic and onecycle schedules |
| `--lr-patience` | int | 10 | Patience for plateau scheduler |
| `--lr-factor` | float | 0.5 | Factor for plateau scheduler reduction |
| `--lr-cycles` | int | 3 | Number of cycles for cyclic and restart schedulers |
| `--lr-cycle-mult` | float | 2.0 | Cycle length multiplier for restart scheduler |

## Monitoring

The system automatically logs:
- Current learning rate at each update
- Learning rate configuration at startup
- Learning rate history saved to `learning_rates.txt`

## Testing and Visualization

Use the test script to visualize different schedules:

```bash
python test_lr_schedules.py --num-updates 600 --learning-rate 2.5e-4
```

This will generate a plot showing all four scheduling strategies for comparison.

## Implementation Details

### PPO Class Changes
- Added `step_lr_scheduler()` method for main loop integration
- Added `get_current_lr()` method for monitoring
- Removed automatic scheduler stepping from update method

### Main Loop Changes
- Learning rate scheduler is stepped after each PPO update
- Current learning rate is logged and tracked
- Learning rate history is saved for analysis

### File Outputs
- `learning_rates.txt`: Learning rate values for each update
- Enhanced logging with current learning rate information

## Best Practices

1. **Start with exponential decay** for most training scenarios
2. **Use step decay** when you have domain knowledge about optimal reduction points
3. **Try cosine annealing** if you experience convergence issues
4. **Use linear schedule with warmup** for very long training runs
5. **Monitor learning rate curves** to ensure they match your expectations

## Advanced Scheduling Strategies

### When to Use Each Method

**Exponential Decay**: Default choice for most training runs. Provides smooth, predictable decay.

**Step Decay**: Use when you know specific milestones (e.g., after certain epochs or when validation plateaus).

**Cosine Annealing**: Excellent for avoiding local minima and providing smooth transitions. Good for long training runs.

**Cosine Annealing with Warm Restarts**: Best for escaping local minima in later training stages. The increasing restart periods provide exploration opportunities.

**OneCycle Policy**: Use for fast convergence and super-convergence training. Excellent for shorter training runs where you want rapid improvement.

**ReduceLROnPlateau**: Ideal when you want automatic adaptation to training progress. Automatically reduces LR when performance stops improving.

**Cyclic Learning Rate**: Great for finding optimal learning rate ranges and escaping local minima. Good for exploration-focused training.

**Cosine Annealing Restart**: Use for structured training with regular exploration phases. Good for maintaining exploration throughout training.

**Linear Schedule**: Best for very long training runs where you want gradual, predictable decay.

### Performance Characteristics

| Method | Convergence Speed | Exploration | Local Minima Escape | Complexity |
|--------|------------------|-------------|---------------------|------------|
| Exponential | Medium | Low | Low | Low |
| Step | Medium | Low | Low | Low |
| Cosine | Medium-High | Medium | Medium | Low |
| Cosine Warmup | High | High | High | Medium |
| OneCycle | Very High | Medium | Medium | Medium |
| Plateau | Variable | Low | Low | Low |
| Cyclic | High | High | High | Medium |
| Restart | High | High | High | Medium |
| Linear | Medium | Low | Low | Low |

## Troubleshooting

### Learning Rate Too Low
- Increase `--anneal-lr-value` (closer to 1.0)
- Decrease `--lr-gamma` for step decay
- Increase `--lr-min` for cosine/linear schedules

### Learning Rate Too High
- Decrease `--anneal-lr-value` (further from 1.0)
- Increase `--lr-gamma` for step decay
- Decrease `--lr-min` for cosine/linear schedules

### Unstable Training
- Try cosine annealing for smoother transitions
- Use linear schedule with warmup for gradual start
- Reduce learning rate step sizes for more frequent updates 