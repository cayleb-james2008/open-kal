"""Genuine RLCD fine-tune of Laya on the recorded production labels, or an honest no.

The labels are the bot's own state at each recorded tick, labelled by the realised
post-fee 30s return — not a curated benchmark. `laya.train.finetune` runs the package's
own `rlcd` loss (Log + Spherical proper scoring) with its `vegaml`-style conformal
calibration split, so what comes out is a real fine-tune of the published checkpoint.

There is no GPU on this host. That is a constraint on wall-clock, not a reason to
pretend: the script reports exactly what it trained, for how long, and refuses to write
an ONNX file it has not measured.

    .venv/bin/python scripts/laya_finetune.py            # dry run + 1 epoch
    .venv/bin/python scripts/laya_finetune.py --epochs 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "laya-rlcd" / "labels.jsonl"
OUT = ROOT / "data" / "laya-rlcd"


def to_csv(path: Path) -> Path:
    target = path.with_suffix(".csv")
    rows = 0
    with path.open() as src, target.open("w", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(["text", "label"])
        for line in src:
            row = json.loads(line)
            writer.writerow([row["text"], row["label"]])
            rows += 1
    print(f"wrote {target} ({rows} rows)", flush=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=25.0)
    args = parser.parse_args()

    from laya import train as laya_train

    data = to_csv(LABELS)
    model_dir = str(Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya")
    # `resolve_checkpoint_dir` understands the hub cache layout; the pinned revision
    # carries both the base and typed-decisions heads.
    snapshot = sorted(Path(model_dir, "snapshots").iterdir())[-1]
    base = str(snapshot / "typed-decisions")

    cfg = laya_train.TrainConfig(epochs=args.epochs, micro_batch=args.micro_batch,
                                 grad_accum=args.grad_accum, loss="rlcd",
                                 freeze_encoder=True)
    dry = laya_train.dry_run(str(data), base, cfg)
    print("dry_run:", json.dumps(dry, indent=2), flush=True)
    if dry.get("valid_items", 0) < 1000:
        print("refusing to train: not enough valid items", flush=True)
        return 2

    started = time.time()
    try:
        result = laya_train.finetune(str(data), base, str(OUT / "model"), config=cfg,
                                     device="cpu")
    except Exception as exc:  # noqa: BLE001 - a failed fine-tune is a real answer
        elapsed = time.time() - started
        payload = {"trained": False, "reason": f"{type(exc).__name__}: {exc}",
                   "elapsed_s": round(elapsed, 1), "dry_run": dry}
        (OUT / "train-result.json").write_text(json.dumps(payload, indent=2, default=str))
        print(json.dumps(payload, indent=2, default=str)[:1500], flush=True)
        return 1
    elapsed = time.time() - started
    payload = {"trained": True, "elapsed_s": round(elapsed, 1), "epochs": args.epochs,
               "device": "cpu", "loss": "rlcd", "dry_run": dry, "result": result}
    (OUT / "train-result.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str)[:3000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
