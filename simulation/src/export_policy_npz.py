#!/usr/bin/env python3
"""把训练出的 .pt 策略导出为纯 numpy .npz（VM 部署无需 torch）。

用法：
  python simulation/src/export_policy_npz.py \
      simulation/output/policies/prone_px/best_mlp_prone_px.pt \
      simulation/output/policies/prone_px/best_mlp_prone_px.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("out", nargs="?", default="")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu",
                      weights_only=False)
    out = args.out or str(Path(args.checkpoint).with_suffix(".npz"))
    payload = {"obs_dim": np.int64(ckpt.get("obs_dim", 27)),
               "act_dim": np.int64(ckpt.get("act_dim", 6)),
               "task": np.array(ckpt.get("task", "prone_px")),
               "backbone": np.array(ckpt.get("backbone", "mlp"))}
    for key, value in ckpt["policy"].items():
        payload[key] = value.numpy().astype(np.float64)
    np.savez_compressed(out, **payload)
    print(f"[EXPORT] {args.checkpoint} -> {out} "
          f"({len(payload) - 4} 个张量)")


if __name__ == "__main__":
    main()
