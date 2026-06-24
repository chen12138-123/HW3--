# HW3 Deep Learning and Spatial Intelligence

This repository contains the code, figures, environment files, and reproduction notes for HW3 Task 1 and Task 2.

PDF reports are not committed to GitHub. They are generated locally and ignored by Git because the final submission PDF should be uploaded separately to the course system or cloud drive.

Repository:

```text
https://github.com/chen12138-123/HW3--
```

Group members:

- Zhou Xiangyang
- Mao Qijun
- Chen Xi

## Repository Layout

```text
.
|-- task1/
|   |-- README.md
|   |-- environment.yml
|   |-- src/
|   `-- figures/
|-- task2/
|   |-- README.md
|   |-- environment.yml
|   |-- scripts/
|   `-- report/
|       |-- main.tex
|       |-- neurips_2025.sty
|       |-- refs.bib
|       `-- figures/
|-- scripts/
|   `-- build_combined_submission.py
|-- SUBMISSION.md
|-- environment.yml
`-- .gitignore
```

Local PDF outputs are generated but not tracked:

```text
task1/report/main.pdf
task2/report/main.pdf
report/HW3_combined_submission.pdf
report/HW3_*.pdf
```

## Task 1

Task 1 builds a multi-source 3D asset generation and real-scene fusion pipeline:

- Background scene reconstruction with GraphDECO/Inria 3D Gaussian Splatting.
- Object A reconstruction from real multi-view data with COLMAP and 3DGS.
- Object B text-to-3D generation with threestudio DreamFusion/SDS, plus a stable deliverable asset generated with TripoSR.
- Object C single-image-to-3D generation with Zero123-style supervision, plus a stable deliverable asset generated with TripoSR.
- Data-level fusion by merging 3DGS-compatible PLY vertex fields after 3D scale, rotation, and translation transforms.

See `task1/README.md` for the detailed workflow.

## Task 2

Task 2 evaluates ACT policy generalization with LeRobot on CALVIN splits:

- `A-only`: trained only on environment A.
- `ABC-joint`: trained jointly on environments A, B, and C.
- `splitD`: unseen environment used for zero-shot offline evaluation.

See `task2/README.md` for training and evaluation commands.

## Build Local Reports

Task 1 report:

```bash
python task1/src/report_builder.py
```

Combined submission report:

```bash
python scripts/build_combined_submission.py
```

The combined report is written locally to:

```text
report/HW3_combined_submission.pdf
```

It is ignored by Git and should be uploaded separately.

## Environments

Task-specific environment files:

```bash
conda env create -f task1/environment.yml
conda env create -f task2/environment.yml
```

The root `environment.yml` is kept as a Task 1 compatible environment for legacy scripts.

## Large Files and Weights

Large datasets, pretrained checkpoints, generated PLY/GLB assets, videos, training outputs, and PDF reports are intentionally excluded from Git.

See `SUBMISSION.md` for the exact upload list.
