"""Plot regional temporal-window RMSE differences for manuscript use."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


DEFAULT_EXPERIMENT = Path(
    "outputs/ten_region_bathymetry/20261008T010000Z_tutuila/"
    "experiment_4_temporal_window/temporal_window_robustness_v1"
)
REFERENCE_WINDOW = 15
MAIN_WINDOWS = [7, 15, 30, 45, 60]
ALL_WINDOWS = [7, 15, 30, 45, 60, 90, 180]

DESIGN_LABELS = {
    "equal_sample": "Equal-sample",
    "natural": "Natural-sample",
}

REGION_LABELS = {
    "anegada": "Anegada",
    "cocos_guam": "Cocos Guam",
    "florida_bay": "Florida Bay",
    "kahului": "Kahului",
    "kaneohe": "Kaneohe",
    "key_largo": "Key Largo",
    "key_west": "Key West",
    "marathon": "Marathon",
    "saipan": "Saipan",
    "tutuila": "Tutuila",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create regional RMSE heatmaps relative to the 15-day window."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_EXPERIMENT / "metrics" / "summary_mean_sd.csv",
        help="Path to summary_mean_sd.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT / "figures",
        help="Directory for PNG, PDF, and SVG outputs.",
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "font.weight": "bold",
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_region_data(summary_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_path)
    required = {
        "design",
        "window_days",
        "scope",
        "region_id",
        "n_mean",
        "rmse_mean",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    region = frame.loc[
        frame["scope"].eq("region")
        & frame["design"].isin(DESIGN_LABELS)
        & frame["window_days"].isin(ALL_WINDOWS),
        list(required),
    ].copy()
    if region.empty:
        raise ValueError("No regional rows were found in the summary file.")
    return region


def natural_best_order(frame: pd.DataFrame) -> list[str]:
    natural = frame.loc[frame["design"].eq("natural")]
    matrix = natural.pivot(index="region_id", columns="window_days", values="rmse_mean")
    matrix = matrix.reindex(columns=MAIN_WINDOWS)
    best_window = matrix.idxmin(axis=1)
    return sorted(matrix.index, key=lambda region: (best_window[region], REGION_LABELS[region]))


def prepare_matrices(
    frame: pd.DataFrame, design: str, windows: list[int], region_order: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    subset = frame.loc[frame["design"].eq(design)]
    absolute = subset.pivot(index="region_id", columns="window_days", values="rmse_mean")
    absolute = absolute.reindex(index=region_order, columns=windows)
    if absolute.isna().any().any():
        raise ValueError(f"Incomplete RMSE matrix for {design} and windows {windows}.")
    delta = absolute.sub(absolute[REFERENCE_WINDOW], axis=0)
    sample_size = subset.groupby("region_id")["n_mean"].first().reindex(region_order)
    return absolute, delta, sample_size


def draw_heatmap(
    frame: pd.DataFrame,
    windows: list[int],
    output_stem: Path,
    include_plateau_note: bool,
) -> None:
    region_order = natural_best_order(frame)
    prepared = {
        design: prepare_matrices(frame, design, windows, region_order)
        for design in DESIGN_LABELS
    }
    color_limit = max(
        float(np.nanmax(np.abs(delta.to_numpy())))
        for _, delta, _ in prepared.values()
    )
    color_limit = max(0.05, np.ceil(color_limit * 20.0) / 20.0)
    cmap = LinearSegmentedColormap.from_list(
        "better_worse", ["#2166AC", "#F7F7F7", "#D6604D"]
    )
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)

    width = 7.15 if len(windows) <= 5 else 7.35
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(width, 4.7),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.04},
    )

    image = None
    for panel_index, (ax, design) in enumerate(zip(axes, DESIGN_LABELS)):
        absolute, delta, sample_size = prepared[design]
        values = delta.to_numpy()
        image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(len(windows)), [str(window) for window in windows])
        ax.set_xlabel("Temporal window (days)")
        title = f"({chr(97 + panel_index)}) {DESIGN_LABELS[design]}"
        if include_plateau_note:
            plateau = (
                "30-180 d: identical effective set"
                if design == "equal_sample"
                else "60-180 d: identical effective set"
            )
            title = f"{title}\n{plateau}"
        ax.set_title(title, loc="left", pad=6)
        ax.set_yticks(range(len(region_order)))
        ax.tick_params(length=0)

        if panel_index == 0:
            labels = [
                f"{REGION_LABELS[region]}  (n={int(round(sample_size[region])):,})"
                for region in region_order
            ]
            ax.set_yticklabels(labels)
            ax.set_ylabel("Region and evaluation sample size")
        else:
            ax.tick_params(axis="y", labelleft=False)

        for row, region in enumerate(region_order):
            region_values = absolute.loc[region].to_numpy()
            minimum = float(np.nanmin(region_values))
            tied_best = np.flatnonzero(np.isclose(region_values, minimum, atol=0.005, rtol=0.0))
            best_column = int(tied_best[0])
            ax.add_patch(
                Rectangle(
                    (best_column - 0.48, row - 0.48),
                    0.96,
                    0.96,
                    fill=False,
                    edgecolor="#111111",
                    linewidth=1.2,
                )
            )
            for column, value in enumerate(values[row]):
                text_color = "white" if abs(value) > 0.58 * color_limit else "#222222"
                label = "0.00" if abs(value) < 0.005 else f"{value:+.2f}"
                ax.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7.2,
                )

        ax.set_xticks(np.arange(-0.5, len(windows), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(region_order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

    assert image is not None
    colorbar = fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        fraction=0.065,
        pad=0.035,
        aspect=35,
    )
    colorbar.set_label("ΔRMSE relative to 15 d (m); negative values favor the current window")
    colorbar.ax.tick_params(length=2)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(output_stem.with_suffix(suffix), facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    frame = load_region_data(args.summary)
    draw_heatmap(
        frame,
        MAIN_WINDOWS,
        args.output_dir / "regional_rmse_delta_heatmap_main_upto60d",
        include_plateau_note=False,
    )
    draw_heatmap(
        frame,
        ALL_WINDOWS,
        args.output_dir / "regional_rmse_delta_heatmap_supplement_all_windows",
        include_plateau_note=True,
    )


if __name__ == "__main__":
    main()
