#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot transpose benchmark CSV")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="docs/benchmark_results.csv",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/images/transpose_bandwidth.png"),
    )
    args = parser.parse_args()

    series = defaultdict(list)
    labels = []
    with args.csv_path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            label = f'{row["size"]}x{row["size"]}'
            if label not in labels:
                labels.append(label)
            series[row["kernel"]].append(float(row["bandwidth_gbps"]))

    x_positions = range(len(labels))
    figure, axis = plt.subplots(figsize=(11, 6))
    for kernel, bandwidths in series.items():
        axis.plot(x_positions, bandwidths, marker="o", label=kernel)

    axis.set_title("CUDA Matrix Transpose Effective Bandwidth")
    axis.set_xlabel("Input shape (m x m)")
    axis.set_ylabel("Effective bandwidth (GB/s)")
    axis.set_xticks(list(x_positions), labels, rotation=25, ha="right")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
