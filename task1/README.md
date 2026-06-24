# Task 1: 3DGS and AIGC Multi-Source Asset Fusion

This folder contains the Task 1 pipeline for multi-source 3D asset generation and fusion in a reconstructed real scene.

PDF reports are generated locally and ignored by Git. The local report path is:

```text
task1/report/main.pdf
```

## Pipeline

The final pipeline contains five stages:

- Reconstruct the background scene from the Mip-NeRF 360 `garden` dataset with GraphDECO/Inria 3D Gaussian Splatting.
- Reconstruct Object A from real multi-view `truck` data with COLMAP and 3DGS.
- Generate Object B from a text prompt with threestudio DreamFusion/SDS, then produce a stable deliverable mesh and 3DGS-compatible PLY with TripoSR.
- Generate Object C from a single foreground image with Zero123-style supervision, then produce a stable deliverable mesh and 3DGS-compatible PLY with TripoSR.
- Fuse background, Object A, Object B, and Object C by merging PLY vertex fields after 3D coordinate transforms.

The final fusion is data-level 3D fusion. It is not a 2D screenshot composition.

## Environment

Create the Task 1 environment:

```bash
conda env create -f environment.yml
conda activate hw3-task1-3dgs-aigc
```

The full pipeline requires:

- NVIDIA GPU and CUDA 11.8-compatible PyTorch.
- COLMAP.
- Compiled GraphDECO/Inria 3DGS CUDA extensions.
- AIGC model weights for the selected generation branches.

## External Assets

The repository excludes large external files. Prepare the following assets locally before running the full pipeline:

- Mip-NeRF 360 `garden` dataset.
- GraphDECO/Inria 3DGS source tree and compiled extensions.
- Tanks and Temples `truck` multi-view data or an equivalent real multi-view object capture.
- threestudio source tree and diffusion-model checkpoints.
- TripoSR checkpoint for final feed-forward mesh generation.

Expected local paths used by the current scripts:

```text
data/mipnerf360/garden/
data/official_3dgs/tandt/truck/
data/models/TripoSR/
submodules/gaussian-splatting/
submodules/threestudio/
```

## Reproduction

Run the official 3DGS background and Object A experiments:

```bash
python src/real_3dgs_pipeline.py \
  --source data/mipnerf360/garden \
  --model_path output/real_3dgs_bg \
  --scene_name garden \
  --images images_4 \
  --iterations 7000 \
  --resolution 4 \
  --eval
```

Generate final Object B and Object C assets:

```bash
python src/final_asset_pipeline.py
```

Fuse all assets in 3D:

```bash
python src/final_fusion.py
```

Rebuild the local Chinese report:

```bash
python src/report_builder.py
```

The report builder regenerates visualization figures and writes ignored local PDF files under:

```text
report/
```

## Main Outputs

Generated outputs are intentionally ignored by Git. Upload these files separately when submitting:

```text
output/real_3dgs_bg/point_cloud/iteration_7000/point_cloud.ply
output/object_a_official_3dgs/point_cloud/iteration_7000/point_cloud.ply
output/final_assets/object_b_final/mesh.glb
output/final_assets/object_b_final/model.ply
output/final_assets/object_c_final/mesh.glb
output/final_assets/object_c_final/model.ply
output/final_fused/model.ply
output/final_fused/roaming_video.mp4
data/models/TripoSR/model.ckpt
```

## Notes

The report records attempted advanced 3D generation routes such as Hunyuan3D-2.1, TRELLIS/TRELLIS.2, and Stable Fast 3D. Routes blocked by model access, network, Windows compatibility, or GPU memory are recorded as attempts rather than final successful results.
