# Setup Guide

## Prerequisites

- Python 3.10+ (main), Python 3.8 (LIBERO only)
- CUDA 12.1+
- conda or mamba

## 1. Main Environment

```bash
git clone https://github.com/lexus-x/jepa.git
cd jepa

conda create -n vljepa python=3.10 -y
conda activate vljepa

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "from vljepa.models import VLJEPA; print('OK')"
```

## 2. V-JEPA 2 Checkpoint

Download Meta's pretrained V-JEPA 2 ViT-L encoder:

```bash
bash scripts/download_vjepa2.sh
```

This downloads `checkpoints/vitl16_vjepa2.pt` (~1.2 GB) from Meta's public release. See the [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985) §2 for architecture details.

## 3. LIBERO (Separate Env)

LIBERO requires Python 3.8 and MuJoCo 210. It has incompatible deps with the main env — use a separate one.

```bash
conda create -n libero python=3.8 -y
conda activate libero

# MuJoCo 2.1.0
mkdir -p ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

# LIBERO
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install "libero @ git+https://github.com/Lifelong-Robot-Learning/LIBERO.git"
pip install -r requirements_libero.txt
```

Verify:

```bash
python -c "import libero; print('LIBERO OK')"
```

## 4. MetaWorld

```bash
conda activate vljepa
pip install "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@master"
```

## 5. Smoke Test

```bash
conda activate vljepa
python tests/smoke_test.py
```

Expected:

```
[1/4] Model construction ........... OK
[2/4] Forward pass ................. OK
[3/4] Flow matching head ........... OK
[4/4] Conformal calibration ........ OK
All checks passed.
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `CUDA out of memory` | Reduce batch size; check `nvidia-smi` for other processes |
| `libcudart.so not found` | `conda install -c conda-forge cudatoolkit=12.1` |
| MuJoCo GL errors (headless) | `export MUJOCO_GL=egl` |
| `ImportError: libero` | Wrong env — `conda activate libero` (Python 3.8) |
| V-JEPA 2 download fails | Check Meta's [repo](https://github.com/facebookresearch/vjepa2) for updated URLs |

---

Next: [Training Guide](training.md) | [Architecture Notes](architecture.md)
