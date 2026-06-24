# Submission Checklist

Unified GitHub repository:

```text
https://github.com/chen12138-123/HW3--
```

Group members:

- Zhou Xiangyang
- Mao Qijun
- Chen Xi

## PDF Reports

PDF files are not committed to GitHub. They are generated or kept locally and should be uploaded separately.

Local combined report:

```text
report/HW3_combined_submission.pdf
```

Local Task 1 report:

```text
task1/report/main.pdf
```

Local Task 2 report:

```text
task2/report/main.pdf
```

## Files Excluded from Git

The repository intentionally excludes:

- PDF reports.
- Raw datasets.
- Pretrained model weights.
- Generated PLY/GLB assets.
- Videos.
- Training output folders and checkpoints.

## Large Files to Upload Separately

Task 1:

```text
task1/output/real_3dgs_bg/point_cloud/iteration_7000/point_cloud.ply
task1/output/object_a_official_3dgs/point_cloud/iteration_7000/point_cloud.ply
task1/output/final_assets/object_b_final/mesh.glb
task1/output/final_assets/object_b_final/model.ply
task1/output/final_assets/object_c_final/mesh.glb
task1/output/final_assets/object_c_final/model.ply
task1/output/final_fused/model.ply
task1/output/final_fused/roaming_video.mp4
task1/data/models/TripoSR/model.ckpt
```

Task 2:

```text
task2/checkpoints/A_only_2000/pretrained_model
task2/checkpoints/ABC_joint_2000/pretrained_model
task2/checkpoints/A_only_10000/pretrained_model
task2/checkpoints/ABC_joint_10000/pretrained_model
```

Task 2 weight link currently recorded in `task2/LINKS.txt`:

```text
https://pan.baidu.com/s/1voGnHTbvz8gjdoWDuJbzug?pwd=mjxn
```

The Task 2 checkpoint folders are referenced by the evaluation scripts but are not present in this local workspace. Upload them from the machine or cloud workspace where Task 2 training was run.
