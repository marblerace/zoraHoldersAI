"""Render the retrieval ablation (Recall@5 + MRR per config) as a PNG for the README."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def main() -> None:
    result_files = sorted(glob.glob("eval/results/retrieval-*.json"))
    if not result_files:
        raise FileNotFoundError("No retrieval evaluation JSON files found")

    latest = result_files[-1]
    with Path(latest).open(encoding="utf-8") as handle:
        data = json.load(handle)

    labels = {
        "dense": "Dense",
        "hybrid": "Hybrid RRF",
        "hybrid_rerank": "Hybrid + rerank",
    }
    order = ("dense", "hybrid", "hybrid_rerank")
    by_mode = {summary["mode"]: summary for summary in data["summaries"]}
    modes = [mode for mode in order if mode in by_mode]
    if not modes:
        raise ValueError(f"No recognized retrieval modes in {latest}")

    recall = [by_mode[mode]["recall_at_k"] for mode in modes]
    mrr = [by_mode[mode]["mrr"] for mode in modes]

    x = list(range(len(modes)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    recall_bars = ax.bar(
        [index - width / 2 for index in x],
        recall,
        width,
        label="Recall@5",
    )
    mrr_bars = ax.bar(
        [index + width / 2 for index in x],
        mrr,
        width,
        label="MRR",
    )
    ax.set_xticks(x, [labels[mode] for mode in modes])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval ablation (higher is better)")
    ax.grid(axis="y", alpha=0.2, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    ax.bar_label(recall_bars, labels=[f"{value:.2f}" for value in recall], padding=3)
    ax.bar_label(mrr_bars, labels=[f"{value:.2f}" for value in mrr], padding=3)

    output = Path("docs/assets/retrieval_ablation.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"source={latest}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
