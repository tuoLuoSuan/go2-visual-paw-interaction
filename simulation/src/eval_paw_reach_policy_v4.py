#!/usr/bin/env python3
"""策略评测：加载 checkpoint，在仿真里跑正式指标（无训练随机性干扰）。

用法：
  python simulation/src/eval_paw_reach_policy.py \
      --checkpoint simulation/output/policies/best_gru_standing.pt \
      --scene D:/Project/robot_dog/unitree_mujoco/unitree_robots/go2/scene.xml
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import standing_paw_lift_common as common
from train_paw_reach_policy_v4 import (
    MLPPolicy, GRUPolicy, obs_dim_for, act_dim_for,
    EPISODE_STEPS, make_envs,
)


def build_envs(args, n_envs):
    a = argparse.Namespace(scene=args.scene, envs=n_envs, task=args.task,
                           latency_range=args.latency_range)
    _, envs = make_envs(a)
    return envs


def eval_policy(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu",
                      weights_only=False)
    obs_dim = ckpt.get("obs_dim", obs_dim_for(args.task))
    act_dim = ckpt.get("act_dim", act_dim_for(args.task))
    backbone = ckpt.get("backbone", args.backbone)
    policy = (GRUPolicy(obs_dim, act_dim) if backbone == "gru"
              else MLPPolicy(obs_dim, act_dim))
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    envs = build_envs(args, args.episodes)
    h = None
    if backbone == "gru":
        h = torch.zeros(1, args.episodes, 128)

    dists = []            # 全部时间步（跨回合）
    step_rows = []        # 逐时间步记录（R2-P0-1 方案A：step_metrics.csv.gz）
    targets_log = []      # 每步前腿目标角（整臂协同指标）
    ep_records = []       # 逐回合聚合（GPT P0-3：CSV 可复算）
    ep_dists = [[] for _ in range(args.episodes)]
    obs_batch = np.zeros((args.episodes, obs_dim), dtype=np.float32)
    for e, env in enumerate(envs):
        obs_batch[e] = env.obs()
    completions = 0
    for step in range(EPISODE_STEPS):
        x = torch.tensor(obs_batch, dtype=torch.float32)
        with torch.no_grad():
            if backbone == "gru":
                mean, _, _, h = policy(x, h)
            else:
                mean, _, _, _ = policy(x)
            act = mean.numpy()
        for e, env in enumerate(envs):
            obs, r, d, info = env.step(act[e])
            dists.append(info["dist"])
            step_rows.append([completions, step, e, info["dist"]])
            targets_log.append(list(env.targets))
            ep_dists[e].append(info["dist"])
            if d:
                # 回合结束：聚合该回合指标（GPT P0-3 方案 B）
                arr = np.asarray(ep_dists[e])
                term = ("time_limit" if env.t >= EPISODE_STEPS
                        else "collapse")
                ep_records.append({
                    "episode": completions + 1,
                    "env_index": e,
                    "steps": int(env.t),
                    "mean_dist": float(arr.mean()),
                    "median_dist": float(np.median(arr)),
                    "p90_dist": float(np.percentile(arr, 90)),
                    "final_dist": float(env.done_info()["dist"]),
                    "close_rate_steps": float((arr < 0.05).mean()),
                    "termination": term,
                })
                ep_dists[e] = []
                completions += 1
                env.reset()
                obs_batch[e] = env.obs()
                if h is not None:
                    h[:, e] = 0.0
            else:
                obs_batch[e] = obs
        if completions >= args.episodes:
            break
    dists = np.asarray(dists)
    ep_final = np.asarray([r["final_dist"] for r in ep_records])
    survivals = np.asarray([r["steps"] for r in ep_records])
    # 整臂协同指标：前腿各关节目标的活动范围（max-min），rad
    tlog = np.asarray(targets_log, dtype=float)
    joint_names = ["FR_h", "FR_t", "FR_c", "FL_h", "FL_t", "FL_c"]
    joint_range = {}
    for i, name in enumerate(joint_names[:6]):
        joint_range[name] = float(tlog[:, i].max() - tlog[:, i].min())
    thigh_range = max(joint_range["FR_t"], joint_range["FL_t"])
    print(f"[EVAL] {args.checkpoint}")
    print(f"[EVAL] backbone={backbone} task={args.task} "
          f"episodes={args.episodes}")
    print(f"[EVAL] mean_dist={dists.mean():.4f}m "
          f"median={np.median(dists):.4f}m "
          f"p90={np.percentile(dists, 90):.4f}m")
    print(f"[EVAL] final_dist={ep_final.mean():.4f}m")
    print(f"[EVAL] close_rate(<5cm)={float((dists < 0.05).mean()):.1%}")
    print(f"[EVAL] mean_episode_steps={survivals.mean():.1f}/"
          f"{EPISODE_STEPS} (survival={float((survivals >= EPISODE_STEPS).mean()):.1%})")
    print(f"[EVAL] joint_target_range_rad={joint_range}")
    print(f"[EVAL] thigh_range_rad={thigh_range:.4f}（整臂协同指标，"
          f"越大说明大臂参与越多）")

    # ---- 证据留存：逐回合聚合 CSV + summary.json（论文证据包）----
    import csv
    import hashlib
    import json
    import platform
    import subprocess
    from datetime import datetime
    from pathlib import Path
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"eval_{args.task}_{backbone}_{now}_seed{args.seed}"
    out_dir = Path(args.evidence_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "episode_metrics.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "env_index", "steps", "mean_dist_m",
                    "median_dist_m", "p90_dist_m", "final_dist_m",
                    "close_rate_steps", "termination"])
        for r in ep_records:
            w.writerow([r["episode"], r["env_index"], r["steps"],
                        f"{r['mean_dist']:.17g}",
                        f"{r['median_dist']:.17g}",
                        f"{r['p90_dist']:.17g}",
                        f"{r['final_dist']:.17g}",
                        f"{r['close_rate_steps']:.17g}", r["termination"]])
    # 逐时间步原始数据（R2-P0-1 方案A；P0-1：.17g 全精度不提前舍入）
    import gzip
    with gzip.open(out_dir / "step_metrics.csv.gz", "wt", newline="",
                   encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "step", "env_index", "distance_m"])
        for r in step_rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.17g}"])
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True,
            text=True, timeout=10).stdout.strip()
        git_dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True,
            text=True, timeout=10).stdout.strip()
    except Exception:
        git_commit, git_dirty = "unknown", ""
    sha = hashlib.sha256()
    with open(args.checkpoint, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "git_dirty": bool(git_dirty),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha.hexdigest(),
        "task": args.task,
        "backbone": backbone,
        "episodes": args.episodes,
        "eval_seed": args.seed,
        "latency_range": list(args.latency_range),
        "schema_version": 2,
        "python": platform.python_version(),
        "data_dictionary": {
            "mean_dist_m": "全部时间步(跨回合)的爪-手距离均值；可由 step_metrics.csv.gz 精确重建",
            "median_dist_m": "全部时间步的距离中位数；可由 step_metrics.csv.gz 精确重建",
            "p90_dist_m": "全部时间步的距离 P90；可由 step_metrics.csv.gz 精确重建",
            "mean_final_dist_m": "各回合终止时刻距离的跨回合均值；可由 episode_metrics.csv 重建",
            "close_rate_5cm": "全部时间步中距离<0.05m 的比例；可由 step_metrics.csv.gz 精确重建",
            "mean_survival_steps": "回合存活步数的跨回合均值(满分200=4s)；可由 episode_metrics.csv 重建",
            "survival_rate": "存活满 200 步的回合比例；可由 episode_metrics.csv 重建",
            "thigh_range_rad": "前腿大腿关节目标在全部时间步的活动范围"
                              "（只支持'大腿目标发生变化'，不单独证明"
                              "平滑/准确/握手成功）",
            "reconstruction_note": "step_metrics.csv.gz 与 episode_metrics.csv "
                                  "以 .17g 全精度保存，summary 各指标可由其"
                                  "精确重建（重建脚本见 "
                                  "simulation/src/reconstruct_metrics.py）。"
                                  "历史 run（2026-08-22 早批，导出精度 4-5 位"
                                  "小数）仅可'在导出精度内重建'，详见其自身"
                                  "summary 的 reconstruction_note。",
        },
        "metrics": {
            "mean_dist_m": float(dists.mean()),
            "median_dist_m": float(np.median(dists)),
            "p90_dist_m": float(np.percentile(dists, 90)),
            "mean_final_dist_m": float(ep_final.mean()),
            "close_rate_5cm": float((dists < 0.05).mean()),
            "mean_survival_steps": float(survivals.mean()),
            "survival_rate": float((survivals >= EPISODE_STEPS).mean()),
            "joint_target_range_rad": joint_range,
            "thigh_range_rad": thigh_range,
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[EVIDENCE] {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--task", choices=("prone", "prone_px", "standing",
                                      "standing_px"),
                   required=True)
    p.add_argument("--backbone", choices=("mlp", "gru"), default="gru")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--latency-range", type=float, nargs=2, default=(0.03, 0.15))
    p.add_argument("--tau-clip", type=float, default=60.0)
    p.add_argument("--mask-active-only", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--evidence-dir", default="evidence/experiments",
                   help="证据输出目录（每 run 一个子目录）")
    args = p.parse_args()
    eval_policy(args)


if __name__ == "__main__":
    main()
