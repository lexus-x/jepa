# tools/

Utility scripts for data preparation, checkpoint management, and environment setup.

## Contents

| Script | Description |
|---|---|
| `download_checkpoints.sh` | Download pre-trained V-JEPA 2 checkpoints from Meta AI |
| `prepare_data.py` | *(planned)* Preprocess and tokenize datasets |
| `export_model.py` | *(planned)* Export trained models to ONNX/TorchScript |

## Usage

```bash
# Download checkpoints
bash tools/download_checkpoints.sh

# All scripts assume you're in the repo root
cd /path/to/vl-jepa
bash tools/<script>.sh
```

## Notes

- Checkpoint downloads require ~20 GB of disk space.
- Scripts are designed to be idempotent — re-running skips already-downloaded files.
