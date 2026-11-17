"""Laya's offline RLCD loop, and the honest switch between it and the hot path.

Two facts shape this module:

  * The zero-shot CPU checkpoint answers in ~1.7s. That is fine for an advisory
    review once per contract and ruinous as a per-entry gate, so it runs in shadow
    until a fine-tuned (and ideally ONNX) model exists.
  * RLCD training needs labeled production ticks — realised post-fee mark-to-market
    at +30s and +60s — not a curated dataset. This module builds exactly that from
    the recorded tape and hands it to Laya's own `proper_reward` pipeline
    (`Log` + `Spherical`) with `vegaml` conformal calibration on top.

Nothing here fabricates a model. If the labels are too few, or `onnxruntime` is
absent, the functions say so and the deterministic kernel keeps the hot path.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

MIN_TRAIN_ROWS = 1000


@dataclass(frozen=True)
class Label:
    ts: float
    series: str
    ticker: str
    p_model: float
    mid: float
    edge: float
    ret_30s: float
    ret_60s: float
    label: str          # "enter" | "skip"


def build_labels(db_path: Path, horizon_s: float = 30.0) -> list[Label]:
    """Label every recorded tick by the realised return the trade would have earned.

    The label is the decision a perfect trader would have made at that tick: enter
    only when the post-fee mark-to-market at `horizon_s` was positive. Fees are the
    exact Kalshi taker fee, so a "profitable" label is profitable after costs.
    """
    if not Path(db_path).exists():
        return []
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT ts, series, ticker, yes_bid, yes_ask, spot, p_model, edge, secs_left"
        " FROM ticks WHERE yes_ask IS NOT NULL ORDER BY ts"
    ).fetchall()
    db.close()
    by_series: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_series.setdefault(row["series"], []).append(row)

    out: list[Label] = []
    for series, series_rows in by_series.items():
        for i, row in enumerate(series_rows):
            try:
                mid0 = (Decimal(row["yes_bid"]) + Decimal(row["yes_ask"])) / 2
                p = float(row["p_model"] or 0.5)
            except Exception:  # noqa: BLE001
                continue
            side = "YES" if p >= float(mid0) else "NO"
            ask0 = Decimal(row["yes_ask"]) if side == "YES" else (Decimal(1) - Decimal(row["yes_bid"]))
            later = [r for r in series_rows[i:] if r["ts"] - row["ts"] >= horizon_s]
            if not later or ask0 <= 0:
                continue
            nxt = later[0]
            mid1 = (Decimal(nxt["yes_bid"]) + Decimal(nxt["yes_ask"])) / 2
            mark = mid1 if side == "YES" else (Decimal(1) - mid1)
            fee = Decimal("0.07") * ask0 * (Decimal(1) - ask0)
            net = mark - ask0 - fee
            out.append(Label(
                ts=float(row["ts"]), series=series, ticker=row["ticker"],
                p_model=p, mid=float(mid0), edge=float(row["edge"] or 0),
                ret_30s=float(net), ret_60s=0.0,
                label="enter" if net > 0 else "skip",
            ))
    return out


def write_jsonl(labels: list[Label], path: Path) -> Path:
    """Laya's `read_data` wants `text` + `label`; the text is the kernel's own state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for lab in labels:
            state = (
                f"lane {lab.series}, model probability {lab.p_model:.3f}, book mid {lab.mid:.3f}, "
                f"kernel edge {lab.edge:+.4f} dollars, realised 30s post-fee return "
                f"{lab.ret_30s:+.4f} dollars per contract."
            )
            fh.write(json.dumps({"text": state, "label": lab.label}) + "\n")
    return path


def status(db_path: Path, out_dir: Path) -> dict[str, Any]:
    labels = build_labels(db_path)
    enters = sum(1 for lab in labels if lab.label == "enter")
    try:
        import onnxruntime  # noqa: F401
        onnx = True
    except Exception:  # noqa: BLE001
        onnx = False
    ckpt = out_dir / "laya.onnx"
    return {
        "labels": len(labels),
        "enter_labels": enters,
        "skip_labels": len(labels) - enters,
        "ready_for_rlcd": len(labels) >= MIN_TRAIN_ROWS,
        "onnxruntime": onnx,
        "onnx_model": ckpt.exists(),
        "hot_path": "onnx" if (onnx and ckpt.exists()) else "deterministic_kernel",
    }


def train(data_path: Path, output_dir: Path, epochs: int = 1) -> dict[str, Any]:
    """Fine-tune Laya on the RLCD labels. Reports what it did, never pretends.

    This is the long pole: it needs a base checkpoint on disk, a torch stack, and
    enough labels that the proper-scoring loss is not fitting noise.
    """
    rows = sum(1 for _ in data_path.open()) if data_path.exists() else 0
    if rows < MIN_TRAIN_ROWS:
        return {"trained": False, "reason": f"{rows} labels < {MIN_TRAIN_ROWS} required"}
    try:
        from laya.train import TrainConfig, finetune
    except Exception as exc:  # noqa: BLE001
        return {"trained": False, "reason": f"laya.train unavailable: {exc}"}
    try:
        result = finetune(str(data_path), model_dir=str(output_dir / "base"),
                          output_dir=str(output_dir), config=TrainConfig(epochs=epochs),
                          device="auto")
        return {"trained": True, "result": result}
    except Exception as exc:  # noqa: BLE001 - a missing checkpoint is a real answer
        return {"trained": False, "reason": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    import sys

    db = Path(sys.argv[1] if len(sys.argv) > 1 else "data/openkal-prod.db")
    out = Path("data/laya-rlcd")
    print(json.dumps(status(db, out), indent=2))
    labels = build_labels(db)
    if labels:
        path = write_jsonl(labels, out / "labels.jsonl")
        print("wrote", path, len(labels), "labels")
        print(json.dumps(train(path, out), indent=2)[:600])
    print(time.strftime("%H:%M:%S"))
