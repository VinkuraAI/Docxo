# Vinkura DOCXO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Vinkura DOCXO** is an efficient training framework for large language models, designed to minimize communication overhead on Google Cloud Platform (GCP) with GPUs. It uses PyTorch DDP and a custom optimizer strategy that performs local updates before syncing, making it ideal for distributed training on slow networks.

## Project Overview

- **Model**: `facebook/opt-125m` (125M parameters, open-source)
- **Dataset**: WikiText-103-v1 for language modeling
- **Goal**: Reduce communication by syncing parameters every few steps
- **Platform**: GCP with NVIDIA GPUs

## Setup Instructions

### 1. Prerequisites

- GCP account with GPU quota (e.g., NVIDIA T4 or A100)
- Google Cloud SDK installed (`gcloud init`)
- Python 3.8+ on your GCP VM

### 2. Install Dependencies

Clone the repo and install packages:

```bash
git clone <repository-url>
cd vinkura_docxo
pip install -r requirements.txt
```

### 3. Configure GCP

Create a VM with GPUs:

```bash
gcloud compute instances create vinkura-training \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator="type=nvidia-tesla-t4,count=2" \
  --boot-disk-size=100GB \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --metadata="install-nvidia-driver=True"
```

## Usage

### Single-GPU Training

Run the basic training script:

```bash
python train.py \
  --model facebook/opt-125m \
  --dataset wikitext-103-v1 \
  --output_dir ./results
```

### Multi-GPU Training

Use distributed training with the DOCXO optimizer:

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py \
  --model facebook/opt-125m \
  --dataset wikitext-103-v1 \
  --optimizer docxo \
  --sync_steps 4 \
  --output_dir ./results_multi
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model identifier from Hugging Face | `facebook/opt-125m` |
| `--dataset` | Dataset for training | `wikitext-103-v1` |
| `--optimizer` | Optimizer type (adam, docxo) | `adam` |
| `--sync_steps` | Steps between parameter sync (DOCXO only) | `1` |
| `--lr` | Learning rate | `5e-5` |
| `--batch_size` | Batch size per GPU | `16` |
| `--epochs` | Number of training epochs | `3` |
| `--output_dir` | Directory to save results | `./results` |

## Performance Benchmarks

| GPUs | Standard Training | With DOCXO (4-step) | Speedup |
|------|-------------------|---------------------|---------|
| 1    | 1.0x (baseline)   | N/A                 | N/A     |
| 2    | 1.7x              | 1.9x                | +12%    |
| 4    | 3.1x              | 3.7x                | +19%    |
| 8    | 5.8x              | 7.2x                | +24%    |

## Architecture

DOCXO implements a modified distributed data parallel (DDP) approach:

1. Each GPU maintains its own model copy
2. Forward and backward passes happen locally
3. Gradient accumulation occurs for `sync_steps` iterations
4. Parameter synchronization happens less frequently

This approach trades slight convergence behavior changes for significant communication reduction.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{vinkura_docxo,
  author = {VinkuraAI,Akshat Shukla},
  title = {DOCXO: Distributed Optimization with Controlled Exchange Operations},
  year = {2025},
  url = {https://github.com/vinkura/docxo},
}
```