"""Generate the GO2 manuscript diagrams and descriptive outcome figure.

All quantitative values are read from source_data_outcomes.csv. The script does
not perform inferential statistics and does not merge observer, execution, and
recovery endpoints into one success label.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)

INK = "#18324A"
BLUE = "#5B8DB8"
BLUE_LIGHT = "#E8F1F7"
TEAL = "#4F9C91"
TEAL_LIGHT = "#E6F2EF"
AMBER = "#C5903D"
AMBER_LIGHT = "#F8EFD9"
RED = "#B65D5D"
GRAY = "#6B7785"
GRID = "#D9E1E8"


def save_all(fig: plt.Figure, stem: str) -> None:
    fig.savefig(ROOT / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(ROOT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(ROOT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(ROOT / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    color: str,
    fill: str,
) -> tuple[FancyBboxPatch, mpl.text.Text]:
    x, y = xy
    patch = FancyBboxPatch(
        (x - 0.135, y - 0.075),
        0.27,
        0.15,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=color,
        facecolor=fill,
    )
    ax.add_patch(patch)
    text_artist = ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=6.7,
        linespacing=1.15,
    )
    return patch, text_artist


def assert_box_labels_fit(
    fig: plt.Figure,
    artists: list[tuple[FancyBboxPatch, mpl.text.Text]],
    *,
    margin_px: float = 2.0,
) -> None:
    """Fail figure generation when a label extends beyond its node box."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    failures: list[str] = []
    for patch, text_artist in artists:
        box_bounds = patch.get_window_extent(renderer)
        text_bounds = text_artist.get_window_extent(renderer)
        if (
            text_bounds.x0 < box_bounds.x0 + margin_px
            or text_bounds.x1 > box_bounds.x1 - margin_px
            or text_bounds.y0 < box_bounds.y0 + margin_px
            or text_bounds.y1 > box_bounds.y1 - margin_px
        ):
            failures.append(text_artist.get_text().replace("\n", " / "))
    if failures:
        raise ValueError(f"box label overflow: {failures}")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, dashed: bool = False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            color=GRAY,
            linestyle="--" if dashed else "-",
            connectionstyle="arc3,rad=0",
        )
    )


def build_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(3.5039, 2.9921))  # 89 x 76 mm
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    positions = [
        (0.17, 0.84), (0.50, 0.84), (0.83, 0.84),
        (0.83, 0.52), (0.50, 0.52), (0.17, 0.52),
        (0.17, 0.20), (0.50, 0.20), (0.83, 0.20),
    ]
    labels = [
        "内置单目\nRGB 图像",
        "MediaPipe\n21 点关键点",
        "归一化掌部\n二维目标",
        "EMA + Kalman\n平滑与预测",
        "27 维\n策略观测",
        "MLP 输出 6 维\n前腿关节增量",
        "50 Hz\n策略更新",
        "500 Hz PD 跟踪\n与安全看门狗",
        "接触保持、收回\n与控制权恢复",
    ]
    colors = [BLUE, BLUE, BLUE, TEAL, TEAL, TEAL, AMBER, AMBER, AMBER]
    fills = [BLUE_LIGHT] * 3 + [TEAL_LIGHT] * 3 + [AMBER_LIGHT] * 3
    for position, label, color, fill in zip(positions, labels, colors, fills):
        box(ax, position, label, color, fill)

    connections = [
        ((0.305, 0.84), (0.365, 0.84)),
        ((0.635, 0.84), (0.695, 0.84)),
        ((0.83, 0.765), (0.83, 0.595)),
        ((0.695, 0.52), (0.635, 0.52)),
        ((0.365, 0.52), (0.305, 0.52)),
        ((0.17, 0.445), (0.17, 0.275)),
        ((0.305, 0.20), (0.365, 0.20)),
        ((0.635, 0.20), (0.695, 0.20)),
    ]
    for start, end in connections:
        arrow(ax, start, end)

    ax.text(0.5, 0.985, "受约束图像空间感知—控制—恢复链", ha="center", va="top", fontsize=8, color=INK, weight="bold")
    ax.text(0.5, 0.03, "二维图像坐标不被解释为机器人坐标系中的一般三维位置", ha="center", va="bottom", fontsize=6.4, color=GRAY)
    save_all(fig, "fig2_system_pipeline")


def build_state_machine() -> None:
    fig, ax = plt.subplots(figsize=(3.5039, 3.0709))  # 89 x 78 mm
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    main = [
        ((0.5, 0.84), "现场检查\n口令/遥控器\n净空/温度", BLUE, BLUE_LIGHT),
        ((0.5, 0.66), "Reach\n冻结趴姿\n缓慢伸足", TEAL, TEAL_LIGHT),
        ((0.5, 0.48), "Track\n接收目标\n并跟踪", TEAL, TEAL_LIGHT),
        ((0.5, 0.30), "Retreat\n收回关节\n停止低层命令", AMBER, AMBER_LIGHT),
        ((0.5, 0.12), "Restore\n恢复运动服务\n与控制权", AMBER, AMBER_LIGHT),
    ]
    boxed_artists: list[tuple[FancyBboxPatch, mpl.text.Text]] = []
    for position, label, color, fill in main:
        boxed_artists.append(box(ax, position, label, color, fill))
    for y1, y2 in [(0.765, 0.735), (0.585, 0.555), (0.405, 0.375), (0.225, 0.195)]:
        arrow(ax, (0.5, y1), (0.5, y2))

    boxed_artists.append(
        box(ax, (0.84, 0.48), "安全门触发\n跟踪/姿态\n或温度越界", RED, "#F7E8E8")
    )
    arrow(ax, (0.635, 0.66), (0.705, 0.52), dashed=True)
    arrow(ax, (0.635, 0.48), (0.705, 0.48), dashed=True)
    arrow(ax, (0.84, 0.405), (0.635, 0.30), dashed=True)

    ax.text(0.14, 0.48, "无手约 3 s\n或达到 90 s 上限", ha="center", va="center", fontsize=6.4, color=GRAY)
    arrow(ax, (0.365, 0.48), (0.28, 0.48), dashed=True)
    arrow(ax, (0.14, 0.405), (0.365, 0.30), dashed=True)
    assert_box_labels_fit(fig, boxed_artists)
    save_all(fig, "fig3_state_machine")


def load_outcomes() -> list[dict[str, str]]:
    with (ROOT / "source_data_outcomes.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_outcomes() -> None:
    rows = load_outcomes()
    labels = [row["endpoint"] for row in rows]
    values = [int(row["count"]) for row in rows]
    totals = [int(row["total"]) for row in rows]

    fig, ax = plt.subplots(figsize=(3.5039, 2.6378))  # 89 x 67 mm
    y = list(range(len(labels)))
    ax.barh(y, totals, color="#EDF1F4", height=0.55, edgecolor="none")
    colors = [BLUE, BLUE, TEAL, AMBER]
    ax.barh(y, values, color=colors, height=0.55, edgecolor="none")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 10.55)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("次数（n = 10 次被命令执行的真机尝试）")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", color=GRID)
    for yy, value, total in zip(y, values, totals):
        ax.text(value + 0.12, yy, f"{value}/{total}", va="center", ha="left", fontsize=7, color=INK, weight="bold")
    ax.annotate(
        "另有 1/10 在接触后因 TRACKING_ERROR 中止",
        xy=(9.1, 2),
        xytext=(5.5, 2.55),
        fontsize=6.2,
        color=RED,
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.8},
    )
    ax.set_title("FORMAL-02 的外部观察、执行与恢复端点", loc="left", color=INK, weight="bold", pad=8)
    fig.subplots_adjust(left=0.34, right=0.96, top=0.84, bottom=0.18)
    save_all(fig, "fig4_outcomes")


def build_standing_extension() -> None:
    """Create one readable, un-cropped representative standing-handshake frame."""
    fig, ax = plt.subplots(figsize=(3.5039, 1.9843))  # 89 x 50.4 mm
    ax.imshow(plt.imread(ROOT / "source_standing_t065.png"))
    ax.axis("off")
    ax.text(
        0.025,
        0.955,
        "站姿录制段 2",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color="white",
        weight="bold",
        bbox={"boxstyle": "round,pad=0.20", "facecolor": INK, "edgecolor": "none", "alpha": 0.82},
    )
    ax.text(
        0.975,
        0.045,
        "65 s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="white",
        bbox={"boxstyle": "round,pad=0.17", "facecolor": "black", "edgecolor": "none", "alpha": 0.68},
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    save_all(fig, "fig5_standing_extension")


def main() -> None:
    build_pipeline()
    build_state_machine()
    build_outcomes()
    build_standing_extension()
    print("generated=fig2_system_pipeline,fig3_state_machine,fig4_outcomes,fig5_standing_extension")


if __name__ == "__main__":
    main()
