#!/usr/bin/env python3
"""三种子评测汇总：从多个 evidence 目录的 summary.json 自动生成统计表。

可比性校验（GPT 审查 R2-P0-5）：任一关键条件不一致或输入重复时中止，
不继续计算均值。校验项：task/backbone/eval_seed/episodes/schema_version/
latency_range/eval 代码 commit 一致；checkpoint_sha256 与 run_id 不重复。

用法：
  python simulation/src/summarize_seeds.py \
      --dirs <evidence目录1> <目录2> <目录3> --out evidence/experiments/seeds_summary
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

COMPARABLE_FIELDS = ("task", "backbone", "eval_seed", "episodes",
                     "schema_version", "git_commit")
REQUIRED_FIELDS = COMPARABLE_FIELDS + ("latency_range", "checkpoint_sha256",
                                       "run_id", "metrics")


def validate_summaries(rows):
    """rows: 从 summary.json 读取的 dict 列表。返回 (task, backbone)。"""
    if len(rows) < 1:
        raise ValueError("至少需要一个 summary")
    for i, r in enumerate(rows):
        for field in REQUIRED_FIELDS:
            if field not in r:
                raise ValueError(f"第 {i + 1} 个 summary 缺少字段 {field}")
    for field in COMPARABLE_FIELDS:
        values = {r[field] for r in rows}
        if len(values) != 1:
            raise ValueError(f"{field} 不一致: {values}")
    lat = {tuple(r["latency_range"]) for r in rows}
    if len(lat) != 1:
        raise ValueError(f"latency_range 不一致: {lat}")
    shas = [r["checkpoint_sha256"] for r in rows]
    if len(set(shas)) != len(shas):
        raise ValueError("checkpoint_sha256 重复（同一模型被重复评测）")
    run_ids = [r["run_id"] for r in rows]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("run_id 重复")
    return rows[0]["task"], rows[0]["backbone"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows_in = [json.loads((Path(d) / "summary.json").read_text(
        encoding="utf-8")) for d in args.dirs]
    task, backbone = validate_summaries(rows_in)

    rows = []
    for s in rows_in:
        m = s["metrics"]
        rows.append({
            "run_id": s["run_id"],
            "checkpoint": s["checkpoint"],
            "checkpoint_sha256": s["checkpoint_sha256"],
            "eval_seed": s["eval_seed"],
            "mean_dist_m": m["mean_dist_m"],
            "median_dist_m": m["median_dist_m"],
            "p90_dist_m": m["p90_dist_m"],
            "mean_final_dist_m": m["mean_final_dist_m"],
            "close_rate_5cm": m["close_rate_5cm"],
            "mean_survival_steps": m["mean_survival_steps"],
            "survival_rate": m["survival_rate"],
            "thigh_range_rad": m["thigh_range_rad"],
        })
    keys = ["mean_dist_m", "median_dist_m", "p90_dist_m",
            "mean_final_dist_m", "close_rate_5cm", "mean_survival_steps",
            "survival_rate", "thigh_range_rad"]
    agg = {}
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=float)
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                  "n": int(len(vals)), "values": [float(v) for v in vals]}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"task": task, "backbone": backbone, "rows": rows,
                   "aggregates": agg}, f, ensure_ascii=False, indent=2)
    print(f"[SEEDS] task={task} backbone={backbone} per-run + aggregates:")
    for k, v in agg.items():
        print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f} "
              f"(n={v['n']}, values={[round(x, 4) for x in v['values']]})")
    print(f"[SEEDS] saved: {out.with_suffix('.csv')} / "
          f"{out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
