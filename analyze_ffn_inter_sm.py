import argparse
import csv
import os
import statistics
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_csv(path):
    data = defaultdict(list)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            b = row["backend"].strip()
            lat = float(row["latency_ms"])
            data[b].append(lat)
    return data


def summarize(vals):
    vals = sorted(vals)
    n = len(vals)
    p50 = vals[n // 2] if n else 0.0
    p95 = vals[min(n - 1, int(0.95 * (n - 1)))] if n else 0.0
    return {
        "count": n,
        "mean": statistics.mean(vals) if n else 0.0,
        "stdev": statistics.pstdev(vals) if n > 1 else 0.0,
        "p50": p50,
        "p95": p95,
        "min": min(vals) if n else 0.0,
        "max": max(vals) if n else 0.0,
    }


def write_summary(data, out_csv):
    backends = sorted(data.keys())
    rows = []
    for b in backends:
        s = summarize(data[b])
        s["backend"] = b
        rows.append(s)

    base_map = {r["backend"]: r["mean"] for r in rows}
    pyt = base_map.get("pytorch", None)
    trt = base_map.get("tensorrt", None)

    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "backend",
            "count",
            "mean_ms",
            "stdev_ms",
            "p50_ms",
            "p95_ms",
            "min_ms",
            "max_ms",
            "speedup_vs_pytorch",
            "speedup_vs_tensorrt",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for r in rows:
            mean = r["mean"]
            out = {
                "backend": r["backend"],
                "count": r["count"],
                "mean_ms": f"{r['mean']:.6f}",
                "stdev_ms": f"{r['stdev']:.6f}",
                "p50_ms": f"{r['p50']:.6f}",
                "p95_ms": f"{r['p95']:.6f}",
                "min_ms": f"{r['min']:.6f}",
                "max_ms": f"{r['max']:.6f}",
                "speedup_vs_pytorch": f"{(pyt / mean) if (pyt and mean > 0) else 0.0:.4f}",
                "speedup_vs_tensorrt": f"{(trt / mean) if (trt and mean > 0) else 0.0:.4f}",
            }
            w.writerow(out)


def make_plots(data, outdir):
    if plt is None:
        print("matplotlib is not installed; skipping plot generation")
        return

    backends = sorted(data.keys())
    means = [statistics.mean(data[b]) for b in backends]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(backends, means, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(backends)])
    plt.ylabel("Latency (ms)")
    plt.title("FFN latency comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "latency_bar.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    series = [data[b] for b in backends]
    plt.boxplot(series, tick_labels=backends, showfliers=True)
    plt.ylabel("Latency (ms)")
    plt.title("FFN latency distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "latency_box.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = load_csv(args.csv)

    summary_csv = os.path.join(args.outdir, "summary.csv")
    write_summary(data, summary_csv)
    make_plots(data, args.outdir)

    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote plots: {os.path.join(args.outdir, 'latency_bar.png')}, {os.path.join(args.outdir, 'latency_box.png')}")


if __name__ == "__main__":
    main()
