#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


def parse_args():
    parser = argparse.ArgumentParser(description="Offline splitD action-error evaluation for LeRobot ACT checkpoints.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default="splitD")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=6400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="checkpoint_name=pretrained_model_dir")
    return parser.parse_args()


def move_tensors(batch, device):
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def main():
    args = parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    device = torch.device(args.device)
    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        video_backend="pyav",
    )
    total = min(args.max_samples, len(ds))
    subset = Subset(ds, list(range(total)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    results = {}
    for spec in args.checkpoints:
        name, ckpt_dir = spec.split("=", 1)
        cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
        cfg.device = str(device)
        policy = get_policy_class(cfg.type).from_pretrained(ckpt_dir, config=cfg, strict=False)
        preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=ckpt_dir)

        policy.eval()
        total_abs = 0.0
        total_arm_abs = 0.0
        total_gripper_abs = 0.0
        total_count = 0
        total_arm_count = 0
        total_gripper_count = 0
        batch_count = 0

        for raw_batch in loader:
            batch_count += 1
            batch = move_tensors(raw_batch, device)
            input_batch = {key: batch[key] for key in cfg.input_features.keys()}
            target_action = batch[ACTION]

            with torch.inference_mode():
                processed = preprocessor(input_batch)
                pred_action = policy.predict_action_chunk(processed)[:, 0]
                pred_action = postprocessor(pred_action)
                if isinstance(pred_action, torch.Tensor):
                    pred_action = pred_action.to(device)

            diff = (pred_action - target_action).abs()
            total_abs += diff.sum().item()
            total_arm_abs += diff[:, :6].sum().item()
            total_gripper_abs += diff[:, 6].sum().item()
            total_count += diff.numel()
            total_arm_count += diff[:, :6].numel()
            total_gripper_count += diff[:, 6].numel()

        results[name] = {
            "checkpoint_dir": ckpt_dir,
            "dataset_root": args.dataset_root,
            "eval_subset_samples": total,
            "batch_size": args.batch_size,
            "num_batches": batch_count,
            "avg_l1_loss": total_abs / total_count,
            "avg_arm_l1_loss": total_arm_abs / total_arm_count,
            "avg_gripper_l1_loss": total_gripper_abs / total_gripper_count,
        }

    output = {
        "dataset_root": args.dataset_root,
        "repo_id": args.repo_id,
        "eval_subset_samples": total,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
