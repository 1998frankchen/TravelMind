# Checkpoints Directory

This directory stores model checkpoints from training runs.

## Directory Structure

```
checkpoints/
├── sft/                     # SFT training checkpoints
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   └── best_model/
├── dpo/                     # DPO training checkpoints
│   └── checkpoint-*/
├── ppo/                     # PPO training checkpoints
│   └── checkpoint-*/
├── grpo/                    # GRPO training checkpoints
│   └── checkpoint-*/
└── README.md               # This file
```

## Checkpoint Management

### Saving Checkpoints

Checkpoints are automatically saved during training based on configuration:

```python
# In training config
save_steps = 500
save_total_limit = 3  # Keep only 3 most recent checkpoints
```

### Loading Checkpoints

```python
from transformers import AutoModelForCausalLM

# Load specific checkpoint
model = AutoModelForCausalLM.from_pretrained("checkpoints/sft/checkpoint-2000")

# Load best model
model = AutoModelForCausalLM.from_pretrained("checkpoints/sft/best_model")
```

### Resume Training

```bash
# Resume from checkpoint
python main.py --function train --resume_from_checkpoint checkpoints/sft/checkpoint-1000
```

## Best Practices

1. **Regular Saves**: Save checkpoints every 500-1000 steps
2. **Limit Storage**: Keep only 3-5 most recent checkpoints
3. **Best Model**: Always save the best performing model separately
4. **Metadata**: Include training config and metrics with checkpoints
5. **Backup**: Backup important checkpoints to cloud storage

## Storage Management

```bash
# Check checkpoint sizes
du -sh checkpoints/*

# Remove old checkpoints
find checkpoints -name "checkpoint-*" -mtime +7 -exec rm -rf {} \;

# Compress checkpoints
tar -czf checkpoint_backup.tar.gz checkpoints/sft/best_model
```

## Notes

- Checkpoints are not tracked by git due to size
- Each checkpoint can be 1-10GB depending on model size
- Use symlinks for frequently accessed checkpoints
- Consider using model compression for production deployment