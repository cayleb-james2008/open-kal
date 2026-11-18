#!/usr/bin/env python3
"""Diagnose payoff calibration on existing sim_core fills, without replaying any trade."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openkal import features, sim_core as S
from scripts import f2_structural as F2, kurtosis_fv_snipe as KF, rnd_common as RC

DATA = ROOT / "data"
STEM = "track2b-flow-student-t"
FIT = DATA / f"{STEM}-is.json"
CHAMPION = "h0.05_w5_15_th0.30_taker"
BINS = 10
SEED = 1794943521
DRAWS = 10_000


def trade_path(split):
    return DATA / f"{STEM}-{split}-trades.jsonl"


def trades(split):
    rows = [json.loads(line) for line in trade_path(split).read_text().splitlines()]
    if (not rows or len({(r["ticker"], r["trigger_ts"]) for r in rows}) != len(rows)
            or any(r["config"] != CHAMPION or r["reason"] != "settle" or r["crossed"] != 0
                   or r["entry_style"] != S.TAKER or Decimal(r["entry"]) > S.CAP
                   or r["order_ts"] - r["trigger_ts"] != S.LAG_S or r["exit_ts"] != r["close"]
                   or r["p_side"] != r["feat"][0] or r["close"] - r["trigger_ts"] != r["feat"][2]
                   for r in rows)):
        raise ValueError(f"{split}: not a unique set of champion sim_core settlement fills")
    return rows


def outcomes(rows):
    return np.array([Decimal(r["pnl"]) > 0 for r in rows], dtype=np.int64)


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def platt(p, slope):
    return sigmoid(slope * logit(p))


def fit_platt(p, y):
    """Fit one nonnegative slope to IS log loss, fixing the intercept to zero."""
    x = logit(p)

    def gradient(slope):
        return float(x @ (sigmoid(slope * x) - y))

    if gradient(0) >= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while gradient(hi) < 0:
        hi *= 2
        if hi > 65_536:
            raise ValueError("one-parameter Platt fit did not have a finite solution")
    for _ in range(55):
        mid = (lo + hi) / 2
        if gradient(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def features4(rows, params):
    # x was not serialized in Track 2B. Invert the actual interpolated CDF,
    # not a different distribution implementation or the rounded fill price.
    tables = {lane: (KF.t_cdf_table(fit["nu"]), fit["s"]) for lane, fit in params.items()}
    result = []
    for r in rows:
        p, edge, secs = r["feat"][:3]
        if not 0 < p < 1 or r["side"] not in (S.YES, S.NO):
            raise ValueError("cannot recover a finite abs_x from a clipped or invalid probability")
        cdf = p if r["side"] == S.NO else 1 - p
        table, scale = tables[r["lane"]]
        x = float(np.interp(cdf, table, KF.CDF_X)) * scale
        result.append((p, edge, secs, abs(x)))
    return np.array(result)


def bin_ids(p):
    return np.clip((p * BINS).astype(int), 0, BINS - 1)


def metrics(p, y):
    if len(p) == 0:
        return {"trades": 0, "ece": None, "brier": None, "log_loss": None, "buckets": []}
    buckets = []
    ids = bin_ids(p)
    for b in np.unique(ids):
        mask = ids == b
        n, wins = int(mask.sum()), int(y[mask].sum())
        rate, predicted = wins / n, float(p[mask].mean())
        z2 = 1.95996398454 ** 2
        center = (rate + z2 / (2 * n)) / (1 + z2 / n)
        half = math.sqrt(z2 * (rate * (1 - rate) / n + z2 / (4 * n * n))) / (1 + z2 / n)
        buckets.append({"bin": int(b), "trades": n, "wins": wins, "mean_p": predicted,
                        "win_rate": rate, "gap_realized_minus_predicted": rate - predicted,
                        "wilson_95": [max(0.0, center - half), min(1.0, center + half)]})
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return {"trades": len(p), "ece": features.expected_calibration_error(list(zip(p.tolist(), y.tolist()))),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
            "mean_p": float(p.mean()), "win_rate": float(y.mean()),
            "gap_realized_minus_predicted": float(y.mean() - p.mean()),
            "prediction_sd": float(p.std()), "buckets": buckets}


def simulated_ece(truth, ids, uniforms):
    """Hold bin membership fixed; predict the true mean within each bin."""
    residuals = (uniforms < truth) - truth
    return sum(np.abs(residuals[:, ids == b].sum(axis=1)) for b in np.unique(ids)) / len(truth)


def null_stats(samples, observed):
    return {"mean": float(samples.mean()), "sd": float(samples.std()),
            "q025_q975": np.quantile(samples, [.025, .975]).tolist(),
            "prob_gt_gate_0_08": float(np.mean(samples > .08)),
            "prob_ge_observed_ece": float(np.mean(samples >= observed))}


def file_digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def economics(rows):
    return RC.summarize(rows)


def audit_frozen():
    train, test = trades("is"), trades("oos")
    fit = json.loads(FIT.read_text())
    if (len(train), len(test)) != (44, 36) or fit["champion"]["config"] != CHAMPION:
        raise ValueError("frozen champion inputs changed")
    if not max(r["close"] for r in train) < fit["cut"] <= min(r["close"] for r in test):
        raise ValueError("chronological IS/OOS boundary violated")
    y_is, y_oos = outcomes(train), outcomes(test)
    x_is, x_oos = (features4(rows, fit["fit_params"]) for rows in (train, test))
    p_is, p_oos = x_is[:, 0], x_oos[:, 0]
    slope = fit_platt(p_is, y_is)
    ridge4 = F2.fit_logistic(x_is, y_is)
    ridge5 = RC.fit_payoff_model(train)
    p5 = [F2.apply_logistic(np.array([r["feat"] for r in rows]), *ridge5) for rows in (train, test)]
    predictions = {
        "raw_student_t": (p_is, p_oos),
        "platt_1p": (platt(p_is, slope), platt(p_oos, slope)),
        "ridge_logistic_4f": (F2.apply_logistic(x_is, *ridge4), F2.apply_logistic(x_oos, *ridge4)),
        "historical_ridge_logistic_5f_control": tuple(p5),
    }
    uniforms = np.random.default_rng(SEED).random((DRAWS, len(test)))
    obs_rate = float(y_oos.mean())
    models = {}
    for name, (p_fit, p_eval) in predictions.items():
        fit_metrics, eval_metrics = metrics(p_fit, y_is), metrics(p_eval, y_oos)
        ids = bin_ids(p_eval)
        models[name] = {
            "is": fit_metrics, "oos": eval_metrics,
            "flat_win_rate_binning_noise": null_stats(
                simulated_ece(np.full(len(test), obs_rate), ids, uniforms), eval_metrics["ece"]),
            "perfectly_calibrated_conditional_on_model_p": null_stats(
                simulated_ece(p_eval, ids, uniforms), eval_metrics["ece"]),
        }
    if round(models["historical_ridge_logistic_5f_control"]["oos"]["ece"], 4) != .1581:
        raise ValueError("could not reproduce historical payoff-ECE 0.1581")
    paths = [FIT, trade_path("is"), trade_path("oos"), ROOT / "openkal/sim_core.py"]
    return {
        "champion": CHAMPION, "cut": fit["cut"],
        "sha256": {str(p.relative_to(ROOT)): file_digest(p) for p in paths},
        "economics": {"is": economics(train), "oos": economics(test)},
        "fit_on_is_only": {"platt_logit_slope": slope, "platt_intercept": 0.0,
                           "ridge": 1.0,
                           "features4": ["p_side", "p_side_minus_trigger_cost", "secs", "abs_x"],
                           "ridge4_weights": ridge4[0].tolist(), "ridge4_norm": ridge4[1].tolist(),
                           "historical_features5": ["p_side", "edge", "secs_left", "flow_fast", "flow_slow"],
                           "historical_ridge5_weights": ridge5[0].tolist(), "historical_ridge5_norm": ridge5[1].tolist()},
        "bootstrap": {"draws": DRAWS, "seed": SEED, "trades_per_draw": len(test), "bins": BINS,
                      "fixed_win_rate": obs_rate,
                      "assumption": "Independent labels conditionally on fixed OOS bin membership. "
                                    "Flat null uses the same 21/36 truth in each occupied bin; "
                                    "conditional null draws at each predictor's own fitted probabilities. "
                                    "The i.i.d. null does not model shared window effects."},
        "models": models,
    }


if __name__ == "__main__":
    out = DATA / "track2b-calibration-audit.json"
    out.write_text(json.dumps(audit_frozen(), indent=2, allow_nan=False) + "\n")
    print(out)
