#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
import random

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

ONE_STEP_KEYS = ["avg_l1_loss", "avg_arm_l1_loss", "avg_gripper_l1_loss"]
REPLAN_KEYS = ["avg_replan_l1_loss", "avg_replan_arm_l1_loss", "avg_replan_gripper_l1_loss"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stratified source-target visual shift evaluation on CALVIN LeRobot splits."
    )
    parser.add_argument("--split-root", action="append", required=True, help="name=/abs/path/to/split")
    parser.add_argument("--repo-id", action="append", required=True, help="name=repo_id_for_lerobot")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=/abs/path/to/pretrained_model")
    parser.add_argument("--samples-per-family", type=int, default=128)
    parser.add_argument(
        "--replan-pairs-per-family",
        type=int,
        default=64,
        help="Number of consecutive-frame pairs per family for chunk replanning consistency.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_mapping(items):
    out = {}
    for item in items:
        name, value = item.split("=", 1)
        out[name] = value
    return out


def task_family(task_text: str) -> str:
    return task_text.split(":", 1)[0].strip()


def load_task_family_map(split_root: Path):
    table = pq.read_table(split_root / "meta" / "tasks.parquet")
    rows = table.to_pylist()
    return {row["task_index"]: task_family(row["task"]) for row in rows}


def build_family_index(split_root: Path):
    family_map = load_task_family_map(split_root)
    parquet_ds = ds.dataset(split_root / "data", format="parquet")
    table = parquet_ds.scanner(columns=["index", "task_index"]).to_table()
    families = defaultdict(list)
    for row in table.to_pylist():
        families[family_map[row["task_index"]]].append(row["index"])
    return families


def build_family_pair_index(split_root: Path):
    family_map = load_task_family_map(split_root)
    parquet_ds = ds.dataset(split_root / "data", format="parquet")
    table = parquet_ds.scanner(columns=["index", "task_index", "episode_index", "frame_index"]).to_table()
    rows = sorted(table.to_pylist(), key=lambda row: row["index"])
    families = defaultdict(list)
    for curr, nxt in zip(rows, rows[1:]):
        curr_family = family_map[curr["task_index"]]
        next_family = family_map[nxt["task_index"]]
        if curr["episode_index"] != nxt["episode_index"]:
            continue
        if nxt["frame_index"] != curr["frame_index"] + 1:
            continue
        if curr_family != next_family:
            continue
        families[curr_family].append((curr["index"], nxt["index"]))
    return families


def select_indices_by_family(family_to_indices, samples_per_family, seed):
    rng = random.Random(seed)
    selected = {}
    for family, indices in sorted(family_to_indices.items()):
        if len(indices) < samples_per_family:
            raise ValueError(f"Family {family} only has {len(indices)} samples, need {samples_per_family}")
        picked = list(indices)
        rng.shuffle(picked)
        selected[family] = sorted(picked[:samples_per_family])
    return selected


def select_aligned_pairs_by_family(source_pairs, target_pairs, pairs_per_family, seed):
    source_selected = {}
    target_selected = {}
    selected_counts = {}

    for family in sorted(set(source_pairs) & set(target_pairs)):
        count = min(len(source_pairs[family]), len(target_pairs[family]), pairs_per_family)
        if count == 0:
            continue

        src_rng = random.Random(f"{seed}:{family}:source")
        tgt_rng = random.Random(f"{seed}:{family}:target")
        src_candidates = list(source_pairs[family])
        tgt_candidates = list(target_pairs[family])
        src_rng.shuffle(src_candidates)
        tgt_rng.shuffle(tgt_candidates)

        source_selected[family] = sorted(src_candidates[:count])
        target_selected[family] = sorted(tgt_candidates[:count])
        selected_counts[family] = count

    return source_selected, target_selected, selected_counts


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


class SequentialPairDataset:
    def __init__(self, dataset, pair_indices):
        self.dataset = dataset
        self.pair_indices = pair_indices

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        curr_idx, next_idx = self.pair_indices[idx]
        return {"curr": self.dataset[curr_idx], "next": self.dataset[next_idx]}


def load_policy_bundle(ckpt_dir, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    cfg.device = str(device)
    policy = get_policy_class(cfg.type).from_pretrained(ckpt_dir, config=cfg, strict=False)
    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=ckpt_dir)
    policy.eval()
    return cfg, policy, preprocessor, postprocessor


def postprocess_actions(postprocessor, action_tensor, device):
    original_shape = action_tensor.shape
    if action_tensor.ndim == 3:
        action_tensor = action_tensor.reshape(-1, original_shape[-1])
    processed = postprocessor(action_tensor)
    if isinstance(processed, torch.Tensor):
        processed = processed.to(device)
    if len(original_shape) == 3:
        processed = processed.reshape(original_shape)
    return processed


def predict_action_chunk(bundle, batch, device):
    cfg, policy, preprocessor, postprocessor = bundle
    input_batch = {key: batch[key] for key in cfg.input_features.keys()}
    processed = preprocessor(input_batch)
    pred_chunk = policy.predict_action_chunk(processed)
    pred_chunk = postprocess_actions(postprocessor, pred_chunk, device)
    return pred_chunk


def summarize_metric_totals(
    total_abs,
    total_arm_abs,
    total_gripper_abs,
    total_count,
    total_arm_count,
    total_gripper_count,
    prefix="avg",
):
    return {
        f"{prefix}_l1_loss": total_abs / total_count,
        f"{prefix}_arm_l1_loss": total_arm_abs / total_arm_count,
        f"{prefix}_gripper_l1_loss": total_gripper_abs / total_gripper_count,
    }


def evaluate_one_step(bundle, dataset, selected_by_family, device, batch_size):
    from lerobot.utils.constants import ACTION

    family_results = {}
    all_indices = []
    for family in sorted(selected_by_family):
        family_indices = selected_by_family[family]
        all_indices.extend(family_indices)
        loader = DataLoader(
            Subset(dataset, family_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        total_abs = 0.0
        total_arm_abs = 0.0
        total_gripper_abs = 0.0
        total_count = 0
        total_arm_count = 0
        total_gripper_count = 0
        batch_count = 0

        for raw_batch in loader:
            batch_count += 1
            batch = move_to_device(raw_batch, device)
            target_action = batch[ACTION]
            with torch.inference_mode():
                pred_action = predict_action_chunk(bundle, batch, device)[:, 0]
            diff = (pred_action - target_action).abs()
            total_abs += diff.sum().item()
            total_arm_abs += diff[:, :6].sum().item()
            total_gripper_abs += diff[:, 6].sum().item()
            total_count += diff.numel()
            total_arm_count += diff[:, :6].numel()
            total_gripper_count += diff[:, 6].numel()

        family_results[family] = {
            "num_samples": len(family_indices),
            "num_batches": batch_count,
            **summarize_metric_totals(
                total_abs,
                total_arm_abs,
                total_gripper_abs,
                total_count,
                total_arm_count,
                total_gripper_count,
            ),
        }

    overall = summarize_family_results(family_results, ONE_STEP_KEYS)
    overall["num_samples"] = len(all_indices)
    return {"overall": overall, "families": family_results}


def evaluate_replan(bundle, dataset, pair_selection_by_family, device, batch_size):
    family_results = {}
    total_pairs = 0

    for family in sorted(pair_selection_by_family):
        family_pairs = pair_selection_by_family[family]
        total_pairs += len(family_pairs)
        loader = DataLoader(
            SequentialPairDataset(dataset, family_pairs),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        total_abs = 0.0
        total_arm_abs = 0.0
        total_gripper_abs = 0.0
        total_count = 0
        total_arm_count = 0
        total_gripper_count = 0
        batch_count = 0
        overlap_steps = None

        for raw_batch in loader:
            batch_count += 1
            batch = move_to_device(raw_batch, device)
            curr_batch = batch["curr"]
            next_batch = batch["next"]
            with torch.inference_mode():
                curr_chunk = predict_action_chunk(bundle, curr_batch, device)
                next_chunk = predict_action_chunk(bundle, next_batch, device)
            overlap = (curr_chunk[:, 1:, :] - next_chunk[:, :-1, :]).abs()
            overlap_steps = overlap.shape[1]
            total_abs += overlap.sum().item()
            total_arm_abs += overlap[:, :, :6].sum().item()
            total_gripper_abs += overlap[:, :, 6].sum().item()
            total_count += overlap.numel()
            total_arm_count += overlap[:, :, :6].numel()
            total_gripper_count += overlap[:, :, 6].numel()

        family_results[family] = {
            "num_pairs": len(family_pairs),
            "num_batches": batch_count,
            "overlap_steps": overlap_steps,
            **summarize_metric_totals(
                total_abs,
                total_arm_abs,
                total_gripper_abs,
                total_count,
                total_arm_count,
                total_gripper_count,
                prefix="avg_replan",
            ),
        }

    overall = summarize_family_results(family_results, REPLAN_KEYS)
    overall["num_pairs"] = total_pairs
    return {"overall": overall, "families": family_results}


def summarize_family_results(family_results, metric_keys):
    if not family_results:
        raise ValueError("No family results were collected for this metric.")
    weighted = {}
    for key in metric_keys:
        weight_name = "num_samples" if "replan" not in key else "num_pairs"
        total_weight = sum(v[weight_name] for v in family_results.values())
        weighted[key] = sum(v[key] * v[weight_name] for v in family_results.values()) / total_weight
    return weighted


def compute_gap(source_result, target_result):
    overall_keys = sorted(
        key for key in source_result["overall"].keys() if key.startswith("avg_") and key in target_result["overall"]
    )
    overall_gap = {
        key.replace("avg_", "delta_"): target_result["overall"][key] - source_result["overall"][key]
        for key in overall_keys
    }
    family_gap = {}
    for family in sorted(set(source_result["families"]) & set(target_result["families"])):
        family_keys = sorted(
            key
            for key in source_result["families"][family].keys()
            if key.startswith("avg_") and key in target_result["families"][family]
        )
        family_gap[family] = {
            key.replace("avg_", "delta_"): target_result["families"][family][key] - source_result["families"][family][key]
            for key in family_keys
        }
    return {"overall": overall_gap, "families": family_gap}


def merge_metric_results(one_step_result, replan_result):
    merged = {"overall": dict(one_step_result["overall"]), "families": {}}
    merged["overall"].update(replan_result["overall"])
    for family, metrics in one_step_result["families"].items():
        merged["families"][family] = dict(metrics)
        if family in replan_result["families"]:
            merged["families"][family].update(replan_result["families"][family])
    for family, metrics in replan_result["families"].items():
        merged["families"].setdefault(family, {}).update(metrics)
    return merged


def main(args):
    split_roots = {k: Path(v) for k, v in parse_mapping(args.split_root).items()}
    repo_ids = parse_mapping(args.repo_id)
    checkpoints = parse_mapping(args.checkpoints)

    source_name = "splitA"
    target_name = "splitD"
    if source_name not in split_roots or target_name not in split_roots:
        raise ValueError("Expected splitA and splitD in --split-root")

    source_family_index = build_family_index(split_roots[source_name])
    target_family_index = build_family_index(split_roots[target_name])
    common_families = sorted(set(source_family_index) & set(target_family_index))
    source_family_pairs = build_family_pair_index(split_roots[source_name])
    target_family_pairs = build_family_pair_index(split_roots[target_name])

    source_selected = select_indices_by_family(
        {family: source_family_index[family] for family in common_families},
        args.samples_per_family,
        args.seed,
    )
    target_selected = select_indices_by_family(
        {family: target_family_index[family] for family in common_families},
        args.samples_per_family,
        args.seed,
    )
    source_replan_selected, target_replan_selected, replan_counts = select_aligned_pairs_by_family(
        source_family_pairs,
        target_family_pairs,
        args.replan_pairs_per_family,
        args.seed,
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets = {
        name: LeRobotDataset(repo_id=repo_ids[name], root=str(root), video_backend="pyav")
        for name, root in split_roots.items()
    }
    device = torch.device(args.device)

    output = {
        "source_split": source_name,
        "target_split": target_name,
        "samples_per_family": args.samples_per_family,
        "replan_pairs_per_family": args.replan_pairs_per_family,
        "seed": args.seed,
        "families": common_families,
        "replan_definition": "Compare overlapping actions between consecutive predicted chunks: chunk_t[1:] vs chunk_t+1[:-1]. Lower is more consistent.",
        "selection": {
            source_name: {family: len(indices) for family, indices in source_selected.items()},
            target_name: {family: len(indices) for family, indices in target_selected.items()},
        },
        "replan_selection": {
            source_name: {family: len(indices) for family, indices in source_replan_selected.items()},
            target_name: {family: len(indices) for family, indices in target_replan_selected.items()},
            "aligned_counts": replan_counts,
        },
        "results": {},
    }

    for ckpt_name, ckpt_dir in checkpoints.items():
        bundle = load_policy_bundle(ckpt_dir, device)
        source_one_step = evaluate_one_step(bundle, datasets[source_name], source_selected, device, args.batch_size)
        target_one_step = evaluate_one_step(bundle, datasets[target_name], target_selected, device, args.batch_size)
        source_replan = evaluate_replan(bundle, datasets[source_name], source_replan_selected, device, args.batch_size)
        target_replan = evaluate_replan(bundle, datasets[target_name], target_replan_selected, device, args.batch_size)
        source_result = merge_metric_results(source_one_step, source_replan)
        target_result = merge_metric_results(target_one_step, target_replan)
        output["results"][ckpt_name] = {
            source_name: source_result,
            target_name: target_result,
            f"{target_name}_minus_{source_name}": compute_gap(source_result, target_result),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)
