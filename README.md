# HW3 Deep Learning and Spatial Intelligence

This repository is the unified submission workspace for HW3. It contains the deliverables for Task 1 and Task 2, together with reproducibility notes, environment files, reports, figures, and evaluation scripts.

Submission repository:

https://github.com/chen12138-123/HW3--

Group members:

- 周湘洋
- 毛琦骏
- 陈希

## Repository Layout

```text
.
├── task1/
│   ├── README.md
│   ├── environment.yml
│   ├── src/
│   ├── figures/
│   └── report/
│       ├── main.pdf
│       └── HW3_Report.pdf
├── task2/
│   ├── README.md
│   ├── environment.yml
│   ├── scripts/
│   └── report/
│       ├── main.pdf
│       ├── main.tex
│       └── figures/
├── SUBMISSION.md
├── report/
│   └── HW3_深度学习与空间智能_综合提交.pdf
├── environment.yml
└── HW3_深度学习与空间智能.pdf
```

The combined submission PDF is:

```text
report/HW3_深度学习与空间智能_综合提交.pdf
```

## Task 1

Task 1 builds a multi-source 3D asset generation and real-scene fusion pipeline:

- Background scene reconstruction with GraphDECO/Inria 3D Gaussian Splatting.
- Object A reconstruction from real multi-view data with COLMAP and 3DGS.
- Object B text-to-3D generation with threestudio DreamFusion/SDS, plus a stable deliverable asset generated with TripoSR.
- Object C single-image-to-3D generation with Zero123-style supervision, plus a stable deliverable asset generated with TripoSR.
- Data-level fusion by merging 3DGS-compatible PLY vertex fields after 3D scale, rotation, and translation transforms.

The final Chinese report is:

```text
task1/report/main.pdf
```

The compatibility copy is:

```text
task1/report/HW3_Report.pdf
```

See `task1/README.md` for the full reproduction workflow and expected external assets.

## Task 2

Task 2 evaluates ACT policy generalization with LeRobot on CALVIN splits:

- `A-only`: trained on environment A only.
- `ABC-joint`: trained jointly on environments A, B, and C.
- `splitD`: held-out zero-shot evaluation environment.

The submitted Chinese report is:

```text
task2/report/main.pdf
```

See `task2/README.md` for training and evaluation commands.

## Environments

Task-specific environment files are provided:

```bash
conda env create -f task1/environment.yml
conda env create -f task2/environment.yml
```

The root `environment.yml` is kept as a Task 1 compatible environment for legacy scripts.

## Large Files and Weights

Large datasets, pretrained checkpoints, generated PLY/GLB assets, videos, and training outputs are intentionally excluded from Git. Upload them separately to a cloud drive or course submission system.

See `SUBMISSION.md` for the exact upload list for Task 1 and Task 2.
