"""从 step/episode 原始文件重建 summary 指标并比对（GPT P0-1 第4条）。

用法：
  python simulation/src/reconstruct_metrics.py <evidence_dir> [--tol 1e-9]
退出码 0=重建一致；1=不一致。
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np


def reconstruct(evidence_dir):
    d = Path(evidence_dir)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    # 逐时间步
    steps = []
    with gzip.open(d / "step_metrics.csv.gz", "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(float(row["distance_m"]))
    dists = np.asarray(steps)
    # 逐回合
    finals, surv, ep_mean = [], [], []
    with open(d / "episode_metrics.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            finals.append(float(row["final_dist_m"]))
            surv.append(int(row["steps"]))
            ep_mean.append(float(row["mean_dist_m"]))
    finals = np.asarray(finals)
    surv = np.asarray(surv)
    out = {
        "mean_dist_m": float(dists.mean()),
        "median_dist_m": float(np.median(dists)),
        "p90_dist_m": float(np.percentile(dists, 90)),
        "mean_final_dist_m": float(finals.mean()),
        "close_rate_5cm": float((dists < 0.05).mean()),
        "mean_survival_steps": float(surv.mean()),
        "survival_rate": float((surv >= 200).mean()),
    }
    return summary, out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence_dir")
    p.add_argument("--tol", type=float, default=1e-9)
    args = p.parse_args()
    summary, out = reconstruct(args.evidence_dir)
    bad = []
    for key, value in out.items():
        ref = summary["metrics"][key]
        if abs(value - ref) > args.tol:
            bad.append(f"{key}: rebuilt={value!r} summary={ref!r}")
    if bad:
        print("[RECON] MISMATCH:")
        for line in bad:
            print("  " + line)
        return 1
    print("[RECON] OK: 全部指标在容差内与 summary 一致")
    for key, value in out.items():
        print(f"  {key}={value!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
