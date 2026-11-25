#!/usr/bin/env python
"""Aggregate native evaluation cells written by run_eval.sh / robotwin_eval.py into a table (episodes, successes, Wilson 95% interval, skipped seeds).

usage: aggregate_results.py OUT_ROOT [--csv FILE] [--episodes-csv FILE]
Reads OUT_ROOT/<clean|rand>/<task>/seed*.json. Success is the simulator's own `eval_success` flag recorded per episode; unresolved infrastructure failures
(`infra_failure`) count as failures and are reported separately.
"""
import argparse, csv, glob, json, math, os


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_cell(d):
    eps = []
    for f in sorted(glob.glob(os.path.join(d, "seed*.json"))):
        r = json.load(open(f)); eps.append({"seed": r["seed"], "success": bool(r.get("success")), "infra_failure": bool(r.get("infra_failure")), "n_calls": r.get("n_calls")})
    skipped = len(glob.glob(os.path.join(d, "skip*.json")))
    return eps, skipped


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--csv"); ap.add_argument("--episodes-csv"); a = ap.parse_args()
    rows, ep_rows = [], []
    for cfg in ("clean", "rand"):
        for d in sorted(glob.glob(os.path.join(a.root, cfg, "*"))):
            task = os.path.basename(d); eps, sk = load_cell(d)
            if not eps:
                continue
            k = sum(e["success"] for e in eps); n = len(eps); lo, hi = wilson(k, n)
            rows.append({"task": task, "config": cfg, "n": n, "success": k, "rate_pct": round(100 * k / n, 1), "wilson95_lo": round(100 * lo, 1), "wilson95_hi": round(100 * hi, 1), "skipped_seeds": sk, "unresolved_infra": sum(e["infra_failure"] for e in eps)})
            ep_rows += [{"task": task, "config": cfg, **e} for e in eps]
    for cfg in ("clean", "rand"):
        s = [r for r in rows if r["config"] == cfg]
        if s:
            k = sum(r["success"] for r in s); n = sum(r["n"] for r in s); print(f"{cfg}: {k}/{n} = {100 * k / n:.2f} %")
    print("task,config,n,success,rate_pct,wilson95,skipped_seeds,unresolved_infra")
    for r in rows:
        print(f'{r["task"]},{r["config"]},{r["n"]},{r["success"]},{r["rate_pct"]},({r["wilson95_lo"]}, {r["wilson95_hi"]}),{r["skipped_seeds"]},{r["unresolved_infra"]}')
    for path, data in ((a.csv, rows), (a.episodes_csv, ep_rows)):
        if path and data:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


if __name__ == "__main__":
    main()
