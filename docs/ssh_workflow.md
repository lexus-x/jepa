# SSH Workflow: Windows Dev → Ubuntu A100 Server

Workflow for developing on Windows and training on a remote Ubuntu server with A100 GPUs.

## 1. SSH Config

`~/.ssh/config` (Windows: `C:\Users\<you>\.ssh\config`):

```
Host a100
    HostName YOUR_SERVER_IP
    User your_username
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

```bash
ssh a100
```

## 2. Code Sync (rsync)

**Push to server:**

```bash
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='checkpoints/' \
    --exclude='data/' --exclude='outputs/' \
    ./ a100:~/vl-jepa/
```

**Pull results:**

```bash
rsync -avz a100:~/vl-jepa/outputs/ ./outputs/
rsync -avz a100:~/vl-jepa/checkpoints/ ./checkpoints/
```

## 3. tmux (Persistent Sessions)

Training must survive SSH disconnects. Use tmux:

```bash
ssh a100
tmux new -s train

# Inside tmux:
conda activate vljepa
torchrun --nproc_per_node=4 scripts/train.py --config configs/phase2_posttrain.yaml

# Detach: Ctrl+B, D
# Reattach: tmux attach -t train
```

| Command | Action |
|---|---|
| `tmux ls` | List sessions |
| `tmux attach -t train` | Reattach |
| `tmux kill-session -t train` | Kill session |
| `Ctrl+B, D` | Detach |
| `Ctrl+B, [` | Scroll mode (q to exit) |

## 4. Remote Training

```bash
ssh a100
tmux new -s train
conda activate vljepa
nvidia-smi  # verify GPUs

torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/phase2_posttrain.yaml \
    --data_path data/droid \
    --output_dir outputs/phase2_run1 \
    --seed 42
```

Monitor:

```bash
watch -n 2 nvidia-smi
tail -f outputs/phase2_run1/train.log
```

## 5. TensorBoard Port Forwarding

```bash
# Local machine:
ssh -L 6006:localhost:6006 a100

# On server:
tensorboard --logdir outputs/ --port 6006 --bind_all
```

Open `http://localhost:6006`.

## 6. Download Checkpoints

```bash
rsync -avz a100:~/vl-jepa/outputs/phase2/checkpoints/best.pt ./checkpoints/
rsync -avz a100:~/vl-jepa/outputs/phase2/logs/ ./outputs/phase2/logs/
```

---

Next: [Training Guide](training.md) | [Setup Guide](setup.md)
