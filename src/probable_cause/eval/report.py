"""Tables and charts from experiment summaries (matplotlib PNGs, light theme).

Palette: the dataviz reference categorical slots, in fixed order, so a variant keeps its
color in every chart.
"""

import json
from pathlib import Path
from typing import Any

SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # blue, orange, aqua, yellow
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
VARIANTS = (("", "base"), ("+rerank", "+ rerank"), ("+rewrite", "+ rewrite"),
            ("+rerank+rewrite", "+ rerank + rewrite"))  # fmt: skip
STRATEGIES = ("fixed", "section", "parent_child")
METHODS = ("bm25", "dense", "hybrid")


def load_summary(run_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return data


def grid_table(configs: dict[str, Any], metric: str = "recall@10") -> str:
    """Rows: strategy x method; columns: the four rerank/rewrite variants."""
    header = "| chunking | retrieval | " + " | ".join(label for _, label in VARIANTS) + " |"
    lines = [header, "|---|---|" + "---:|" * len(VARIANTS)]
    for strategy in STRATEGIES:
        for method in METHODS:
            cells = []
            for suffix, _ in VARIANTS:
                summary = configs.get(f"{strategy}+{method}{suffix}")
                cells.append(f"{summary['all'][metric]:.3f}" if summary else "-")
            lines.append(f"| {strategy} | {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ranked_table(configs: dict[str, Any], top: int = 10) -> str:
    rows = sorted(configs.items(), key=lambda kv: -kv[1]["all"]["recall@10"])[:top]
    lines = [
        "| config | R@5 | R@10 | R@20 | MRR | nDCG@10 | lookup R@10 | structured R@10 "
        "| analytical R@10 | p50 ms |",
        "|---|" + "---:|" * 9,
    ]
    for name, s in rows:
        a = s["all"]
        lines.append(
            f"| {name} | {a['recall@5']:.3f} | {a['recall@10']:.3f} | {a['recall@20']:.3f} | "
            f"{a['mrr']:.3f} | {a['ndcg@10']:.3f} | {s['lookup']['recall@10']:.3f} | "
            f"{s['structured']['recall@10']:.3f} | {s['analytical']['recall@10']:.3f} | "
            f"{s['latency_ms']['p50']:.0f} |"
        )
    return "\n".join(lines)


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grid_chart(configs: dict[str, Any], path: Path, metric: str = "recall@10") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, facecolor=SURFACE)
    width = 0.2
    for ax, strategy in zip(axes, STRATEGIES, strict=True):
        _style(ax)
        for v, (suffix, label) in enumerate(VARIANTS):
            values = [configs[f"{strategy}+{m}{suffix}"]["all"][metric] for m in METHODS]
            xs = [i + (v - 1.5) * (width + 0.02) for i in range(len(METHODS))]
            bars = ax.bar(xs, values, width, color=SLOTS[v], label=label,
                          edgecolor=SURFACE, linewidth=1)  # fmt: skip
            if suffix == "+rerank+rewrite":  # label only the strongest variant
                for bar, value in zip(bars, values, strict=True):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}",
                            ha="center", va="bottom", fontsize=8, color=INK)  # fmt: skip
        ax.set_xticks(range(len(METHODS)), METHODS)
        ax.set_title(strategy.replace("_", "-"), color=INK, fontsize=11, loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel(metric, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=4,
               labelcolor=INK)  # fmt: skip
    fig.suptitle(f"Retrieval {metric} by chunking, retriever and add-ons (284 questions)",
                 color=INK, fontsize=12, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def by_type_chart(configs: dict[str, Any], names: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = ("lookup", "structured", "analytical")
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=SURFACE)
    _style(ax)
    width = 0.26
    for i, name in enumerate(names):
        values = [configs[name][t]["recall@10"] for t in types]
        xs = [j + (i - (len(names) - 1) / 2) * (width + 0.02) for j in range(len(types))]
        bars = ax.bar(xs, values, width, color=SLOTS[i], label=name, edgecolor=SURFACE)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK)  # fmt: skip
    ax.set_xticks(range(len(types)), types)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall@10", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    ax.set_title("Recall@10 by question type", color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
