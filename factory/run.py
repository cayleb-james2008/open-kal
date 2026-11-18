"""One walk-forward run: for each epoch, evolve on the train book, judge the survivors on the epoch's test blocks, log everything."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from factory import indicators as IND  # noqa: E402
from factory import judge as J  # noqa: E402
from factory import leaderboard as LB  # noqa: E402
from factory import screen, search, sim  # noqa: E402
from factory.book import load  # noqa: E402

TOP_K = 500                                  # train survivors judged out of sample per epoch


def window_order(book, win: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Chronological order of trades by entry time (close minus seconds left)."""
    return np.argsort(book.wclose[win] - book.X[book.secs_idx, row], kind="stable")


def run_epoch(db, book, epoch: int, train: list[int], test: list[int], wblock: np.ndarray, rng, *, pop: int, gens: int, top_k: int = TOP_K) -> dict:
    t0 = time.time()
    label = J.labels(book.wclose, wblock, train, test)
    train_book = search.subset(book, label == 1)
    qt = search.quantile_table(train_book)
    space = search.Space(book.names, qt)
    G, fams, acc_train = search.evolve(train_book, space, rng, pop=pop, gens=gens)
    LB.add_trials(db, epoch, fams)
    n_tried = LB.trials_in_epoch(db, epoch)
    fit = search.fitness(acc_train)
    hashes = np.array([search.code_hash(g) for g in G])
    # distinct survivors: best fitness per hash, then the top K by train fitness
    order = np.argsort(-fit)
    seen, keep = set(), []
    for i in order:
        if hashes[i] not in seen:
            seen.add(hashes[i])
            keep.append(i)
        if len(keep) >= top_k:
            break
    keep = np.array(keep)
    Gk = G[keep]
    train_end = float(book.wclose[label == 1].max())
    # tier 1 on the test blocks: base and stress
    acc, _ = screen.screen(book, Gk, label, wblock, int(wblock.max()) + 1)
    acc_s, _ = screen.screen(book, Gk, label, wblock, int(wblock.max()) + 1, lat=10, slip=0.01)
    tr, te, tes = screen.summarize(acc[:, 1]), screen.summarize(acc[:, 2]), screen.summarize(acc_s[:, 2])
    bu, bc = J.bounds(te["mean"], te["se"], n_tried)
    # PBO of the selection procedure over every block up to the epoch's last test block
    upto = wblock <= max(test)
    lab_all = np.where(upto, 1, 0)
    _, blk = screen.screen(book, Gk, lab_all, np.where(upto, wblock, -1), int(wblock.max()) + 1)
    pbo = J.pbo(blk[:, : max(test) + 1].T)
    # deflated Sharpe on per-window OOS PnL; the null Sharpe variance is the spread across the judged candidates
    test_w = np.flatnonzero(label == 2)
    srs, win_pnls = [], []
    rows_out = []
    for j, g in enumerate(Gk):
        pnl, win, row, qty = sim.simulate(book, g)
        m = label[win] == 2
        per_w = np.zeros(len(test_w))
        pos = np.searchsorted(test_w, win[m])
        np.add.at(per_w, pos, pnl[m])
        win_pnls.append(per_w)
        srs.append(per_w.mean() / per_w.std(ddof=1) if per_w.std(ddof=1) > 0 else 0.0)
        o = window_order(book, win[m], row[m])
        rows_out.append(J.max_drawdown(pnl[m][o]))
    var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.01
    recs = []
    for j, g in enumerate(Gk):
        dsr = J.deflated_sharpe(win_pnls[j], n_tried, var_sr)
        status, reason = J.verdict(oos_n=te["n"][j], oos_mean=te["mean"][j], bound_unc=bu[j], bound_corr=bc[j], dd=rows_out[j],
                                   stress_mean=tes["mean"][j] if tes["n"][j] > 0 else -1.0, dsr=dsr, pbo_epoch=pbo["pbo"])
        i = keep[j]
        recs.append({"code_hash": hashes[i], "genome": g.tolist(), "family": fams[i], "epoch": epoch, "train_end": train_end,
                     "registry_version": IND.VERSION, "train_metrics": {k: float(v[j]) for k, v in tr.items()},
                     "oos_metrics": {**{k: float(v[j]) for k, v in te.items()}, "max_dd": rows_out[j], "dsr": dsr},
                     "stress_metrics": {k: float(v[j]) for k, v in tes.items()}, "bound_uncorrected": float(bu[j]), "bound_corrected": float(bc[j]),
                     "n_candidates_in_epoch": n_tried, "status": status, "failure_reason": reason, "tier": "screen"})
    # the non-survivors are logged too (train metrics only): every candidate is a row
    rest = [i for i in order if hashes[i] not in {r["code_hash"] for r in recs}]
    seen_rest, extra = set(), []
    for i in rest:
        if hashes[i] in seen_rest:
            continue
        seen_rest.add(hashes[i])
        s = screen.summarize(acc_train[i:i + 1])
        extra.append({"code_hash": hashes[i], "genome": G[i].tolist(), "family": fams[i], "epoch": epoch, "train_end": train_end,
                      "registry_version": IND.VERSION, "train_metrics": {k: float(v[0]) for k, v in s.items()}, "status": "train_only",
                      "failure_reason": "not among the top train survivors"})
    LB.log_candidates(db, recs + extra)
    db.execute("INSERT OR REPLACE INTO epochs VALUES (?,?,?,?,?,?,?,?,?,?)", (epoch, t0, time.time(), json.dumps(train), json.dumps(test), n_tried,
                                                                                len(seen) + len(seen_rest), pbo["pbo"], time.time() - t0, ""))
    db.commit()
    return {"epoch": epoch, "tried": len(G), "distinct": len(seen) + len(seen_rest), "judged": len(recs), "pbo": pbo["pbo"], "seconds": time.time() - t0,
            "passed": sum(r["status"] == "passed_screen" for r in recs), "train_windows": int((label == 1).sum()), "test_windows": int((label == 2).sum())}


def main(feature_dir: Path, db_path: Path, pop: int, gens: int, seed: int = 0) -> list[dict]:
    book = load(feature_dir)
    wblock = J.block_ids(book.wclose)
    db = LB.connect(db_path)
    rng = np.random.default_rng(seed)
    out = []
    for e, (tr, te) in enumerate(J.epochs(int(wblock.max()) + 1)):
        res = run_epoch(db, book, e, tr, te, wblock, rng, pop=pop, gens=gens)
        print(json.dumps(res), flush=True)
        out.append(res)
    return out


if __name__ == "__main__":
    fd = sorted((ROOT / "data" / "factory").glob("features-*"))[-1]
    main(fd, ROOT / "data" / "factory" / "leaderboard.db", pop=int(sys.argv[1]) if len(sys.argv) > 1 else 2000, gens=int(sys.argv[2]) if len(sys.argv) > 2 else 8)
