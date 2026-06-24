# Task 2: ACT Cross-Environment Generalization with LeRobot

This folder contains the Task 2 materials for the ACT policy generalization experiment on CALVIN-style LeRobot datasets.

PDF reports are generated or stored locally and ignored by Git. The local report path is:

```text
task2/report/main.pdf
```

The GitHub repository keeps the LaTeX source, figures, scripts, and environment file.

The experiment compares:

- `A-only`: trained only on environment A.
- `ABC-joint`: trained jointly on environments A, B, and C.
- `splitD`: unseen environment used for zero-shot offline evaluation.

## Environment

Create the environment from the local file:

```bash
conda env create -f environment.yml
conda activate lerobot
```

Install the CUDA-compatible PyTorch build for your machine if the default solver does not select the desired version:

```bash
pip install torch torchvision torchaudio
```

Install LeRobot:

```bash
cd /path/to/lerobot
pip install -e .
```

## Data

The repository does not redistribute the CALVIN dataset. The experiments use the LeRobot-format dataset from Hugging Face:

```text
https://huggingface.co/datasets/xiaoma26/calvin-lerobot/tree/main
```

Expected local layout:

```text
/path/to/calvin-lerobot/
|-- splitA
|-- splitB
|-- splitC
|-- splitD
`-- splitABC
```

`splitABC` is the local union of `splitA`, `splitB`, and `splitC`. The scripts expect each split to contain:

- `data/`
- `meta/tasks.parquet`

Key fields:

- `observation.images.image`
- `observation.images.wrist_image`
- `observation.state`
- `action`

## Training

Train `A-only@2000`:

```bash
CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --policy.type=act \
  --dataset.repo_id=calvin_splitA \
  --dataset.root=/path/to/calvin-lerobot/splitA \
  --batch_size=32 \
  --steps=2000 \
  --num_workers=4 \
  --save_freq=20000 \
  --wandb.enable=false \
  --output_dir=/path/to/runs/act_splitA_steps2000
```

Train `ABC-joint@2000`:

```bash
CUDA_VISIBLE_DEVICES=3 lerobot-train \
  --policy.type=act \
  --dataset.repo_id=calvin_splitABC \
  --dataset.root=/path/to/calvin-lerobot/splitABC \
  --batch_size=32 \
  --steps=2000 \
  --num_workers=4 \
  --save_freq=20000 \
  --wandb.enable=false \
  --output_dir=/path/to/runs/act_splitABC_steps2000
```

Train `A-only@10000`:

```bash
CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --policy.type=act \
  --dataset.repo_id=calvin_splitA \
  --dataset.root=/path/to/calvin-lerobot/splitA \
  --batch_size=32 \
  --steps=10000 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir=/path/to/runs/act_splitA_steps10000
```

Train `ABC-joint@10000`:

```bash
CUDA_VISIBLE_DEVICES=3 lerobot-train \
  --policy.type=act \
  --dataset.repo_id=calvin_splitABC \
  --dataset.root=/path/to/calvin-lerobot/splitABC \
  --batch_size=32 \
  --steps=10000 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir=/path/to/runs/act_splitABC_steps10000
```

## Evaluation

Visual shift and replan consistency:

```bash
python scripts/eval_shift.py \
  --split-root splitA=/path/to/calvin-lerobot/splitA \
  --split-root splitD=/path/to/calvin-lerobot/splitD \
  --repo-id splitA=calvin_splitA \
  --repo-id splitD=calvin_splitD \
  --checkpoints \
    A10000=checkpoints/A_only_10000/pretrained_model \
    ABC10000=checkpoints/ABC_joint_10000/pretrained_model \
  --samples-per-family 128 \
  --replan-pairs-per-family 64 \
  --batch-size 32 \
  --device cuda:0 \
  --seed 0 \
  --output outputs/results/visual_shift_with_replan_10000.json
```

Offline action error:

```bash
python scripts/eval_metrics.py \
  --dataset-root /path/to/calvin-lerobot/splitD \
  --repo-id splitD \
  --batch-size 32 \
  --max-samples 6400 \
  --device cuda:0 \
  --output outputs/results/eval_task2_metrics_full.json \
  --checkpoints \
    A2000=checkpoints/A_only_2000/pretrained_model \
    ABC2000=checkpoints/ABC_joint_2000/pretrained_model \
    A10000=checkpoints/A_only_10000/pretrained_model \
    ABC10000=checkpoints/ABC_joint_10000/pretrained_model
```

## Report

The LaTeX source is kept under `report/` for transparency. The original machine used XeLaTeX with the local NeurIPS-style template. The generated PDF is ignored by Git and should be uploaded separately.
