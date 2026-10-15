"""Create publication figures for spatially independent regional adaptation.

The main figure reports region-specific RMSE on fixed CUDEM support outside all
candidate target-adaptation tracks by 500 m. A CUDEM reference-disagreement band
is optional and is drawn only from an explicitly validated same-support file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = Path(r"H:\ATL24\PhisicKPConvNet")
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "20261011T000000Z_selected_r2"
    / "regional_adaptation"
    / "leave_one_region_out_spatially_independent_cudem_v2"
)
SUPPORT_PROTOCOL = "fixed_CUDEM_support_outside_all_candidate_adaptation_tracks_500m"
PROPORTIONS = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]
PERCENT_LABELS = ["0", "5", "10", "25", "50", "100"]
REGIONS = [
    ("florida_bay", "Florida Bay"),
    ("key_largo", "Key Largo"),
    ("key_west", "Key West"),
    ("marathon", "Marathon"),
    ("anegada", "Anegada"),
    ("tutuila", "Tutuila"),
    ("cocos_guam", "Cocos (Guam)"),
    ("kahului", "Kahului"),
    ("saipan", "Saipan"),
    ("kaneohe", "Kaneohe Bay"),
]
DISPLAY_NAMES = dict(REGIONS)
REGION_ORDER = [region_id for region_id, _ in REGIONS]

BLUE = "#1F4E79"
GREY = "#6E6E6E"
GRID = "#D9D9D9"
TIMES_BOLD_PATH = Path(r"C:\Windows\Fonts\timesbd.ttf")
IMPROVEMENT_CMAP = LinearSegmentedColormap.from_list(
    "improvement", ["#B2182B", "#F7F7F7", "#2166AC"]
)


def configure_style() -> None:
    for font_file in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        path = Path(r"C:\Windows\Fonts") / font_file
        if path.exists():
            font_manager.fontManager.addfont(path)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-error-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with target_region, rmse, ci_low, ci_high, "
            "support_protocol, and independent_reference columns."
        ),
    )
    return parser.parse_args()


def load_metrics(result_dir: Path) -> pd.DataFrame:
    path = result_dir / "metrics" / "adaptation_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {
        "status",
        "target_region",
        "nominal_proportion",
        "target_adaptation_rows",
        "target_adaptation_tracks",
        "independent_validation_pixels",
        "test_support",
        "rmse",
        "mae",
        "bias",
        "r2",
        "rmse_ci_low",
        "rmse_ci_high",
        "mae_ci_low",
        "mae_ci_high",
        "bias_ci_low",
        "bias_ci_high",
        "r2_ci_low",
        "r2_ci_high",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not df["status"].eq("OK").all():
        raise ValueError("Not all adaptation tasks have status OK")
    if set(df["target_region"]) != set(REGION_ORDER):
        raise ValueError("The result does not contain the expected ten regions")
    observed_proportions = sorted(df["nominal_proportion"].astype(float).unique())
    if not np.allclose(observed_proportions, PROPORTIONS):
        raise ValueError(f"Unexpected proportions: {observed_proportions}")
    if len(df) != len(REGION_ORDER) * len(PROPORTIONS):
        raise ValueError(f"Expected 60 rows, found {len(df)}")
    if not df["test_support"].eq(SUPPORT_PROTOCOL).all():
        raise ValueError("Metrics do not all use the required fixed 500 m support")
    validation_counts = df.groupby("target_region")["independent_validation_pixels"].nunique()
    if not validation_counts.eq(1).all():
        raise ValueError("Validation support changes across proportions")
    for metric in ("rmse", "mae", "bias", "r2"):
        if (df[f"{metric}_ci_low"] > df[metric]).any() or (df[f"{metric}_ci_high"] < df[metric]).any():
            raise ValueError(f"Invalid confidence interval for {metric}")
    df["target_region"] = pd.Categorical(df["target_region"], REGION_ORDER, ordered=True)
    return df.sort_values(["target_region", "nominal_proportion"]).reset_index(drop=True)


def load_assignments(result_dir: Path) -> dict[tuple[str, float], dict[str, Any]]:
    assignments: dict[tuple[str, float], dict[str, Any]] = {}
    for region_id in REGION_ORDER:
        region_dir = result_dir / "assignments" / region_id
        for proportion in PROPORTIONS:
            path = region_dir / f"p{int(round(proportion * 100)):03d}.json"
            if not path.exists():
                if np.isclose(proportion, 1.0):
                    # The workflow treats 100% as the complete candidate pool
                    # and does not write a redundant assignment file.
                    continue
                raise FileNotFoundError(path)
            assignments[(region_id, proportion)] = json.loads(path.read_text(encoding="utf-8"))
    return assignments


def load_reference_error(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"target_region", "rmse", "ci_low", "ci_high", "support_protocol", "independent_reference"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Reference-error CSV is missing: {missing}")
    independent = df["independent_reference"].astype(str).str.lower().isin({"true", "1", "yes"})
    same_support = df["support_protocol"].eq(SUPPORT_PROTOCOL)
    if not independent.all() or not same_support.all():
        raise ValueError(
            "Reference-error rows must be independent and use the exact fixed 500 m support protocol"
        )
    if set(df["target_region"]) != set(REGION_ORDER):
        raise ValueError("Reference-error CSV must contain exactly the ten target regions")
    if (df["ci_low"] > df["rmse"]).any() or (df["ci_high"] < df["rmse"]).any():
        raise ValueError("Reference-error confidence intervals do not contain their RMSE estimates")
    return df.set_index("target_region")


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    if TIMES_BOLD_PATH.exists():
        for text_object in fig.findobj(match=mpl.text.Text):
            properties = text_object.get_fontproperties().copy()
            properties.set_file(str(TIMES_BOLD_PATH))
            properties.set_weight("bold")
            text_object.set_fontproperties(properties)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def plot_main_rmse(
    df: pd.DataFrame,
    output_dir: Path,
    reference_error: pd.DataFrame | None,
) -> None:
    x = np.arange(len(PROPORTIONS))
    fig, axes = plt.subplots(2, 5, figsize=(7.16, 4.75))
    panel_labels = [f"({chr(97 + i)})" for i in range(len(REGIONS))]

    for ax, (region_id, display_name), panel_label in zip(axes.flat, REGIONS, panel_labels):
        region = df[df["target_region"].astype(str).eq(region_id)].copy()
        y = region["rmse"].to_numpy(float)
        low = region["rmse_ci_low"].to_numpy(float)
        high = region["rmse_ci_high"].to_numpy(float)
        yerr = np.vstack([np.maximum(y - low, 0), np.maximum(high - y, 0)])

        if reference_error is not None:
            ref = reference_error.loc[region_id]
            ax.axhspan(float(ref.ci_low), float(ref.ci_high), color="#BDBDBD", alpha=0.25, zorder=0)
            ax.axhline(float(ref.rmse), color="#555555", linestyle=":", linewidth=1.0, zorder=1)

        benchmark = float(region.loc[np.isclose(region["nominal_proportion"], 1.0), "rmse"].iloc[0])
        ax.axhline(benchmark, color=GREY, linestyle="--", linewidth=1.0, zorder=1)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=BLUE,
            marker="o",
            markersize=3.8,
            markerfacecolor=BLUE,
            markeredgecolor=BLUE,
            linewidth=1.25,
            elinewidth=0.8,
            capsize=2.0,
            zorder=3,
        )
        ax.plot(
            x[0],
            y[0],
            marker="o",
            markersize=4.3,
            markerfacecolor="white",
            markeredgecolor=BLUE,
            markeredgewidth=1.1,
            linestyle="None",
            zorder=4,
        )

        upper_candidates = [float(np.nanmax(high)), float(np.nanmax(y)), benchmark]
        if reference_error is not None:
            upper_candidates.append(float(reference_error.loc[region_id, "ci_high"]))
        ax.set_ylim(0, max(upper_candidates) * 1.09)
        ax.set_xlim(-0.28, len(x) - 0.72)
        ax.set_xticks(x)
        ax.set_xticklabels(PERCENT_LABELS)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
        ax.tick_params(width=0.75, length=3.0)
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontweight("bold")

        delta_pct = (y[-1] - y[0]) / y[0] * 100.0
        ax.text(
            0.97,
            0.95,
            f"Change: {delta_pct:+.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.6,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.5},
        )
        ax.set_title(f"{panel_label} {display_name}", fontsize=8.2, pad=3.0, fontweight="bold")

    fig.supylabel("RMSE (m)", x=0.006, y=0.54, fontsize=9.0, fontweight="bold")
    fig.supxlabel("Target-region adaptation share (%)", x=0.52, y=0.067, fontsize=9.0, fontweight="bold")
    handles = [
        Line2D([], [], color=BLUE, marker="o", markerfacecolor="white", markeredgewidth=1.1,
               linewidth=0, markersize=4.5, label="Zero-shot (0%)"),
        Line2D([], [], color=BLUE, marker="o", linewidth=1.25, markersize=4.0,
               label="Target-region adaptation"),
        Line2D([], [], color=GREY, linestyle="--", linewidth=1.0,
               label="100% target-data benchmark"),
    ]
    if reference_error is not None:
        handles.append(Patch(facecolor="#BDBDBD", alpha=0.25, edgecolor="none",
                             label="Independent CUDEM-reference disagreement (95% CI)"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=len(handles),
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.965, bottom=0.145, wspace=0.42, hspace=0.24)
    save_figure(fig, output_dir, "Fig_Regional_Adaptation_RMSE")


def unchanged_assignment_cells(
    assignments: dict[tuple[str, float], dict[str, Any]],
) -> set[tuple[int, int]]:
    unchanged: set[tuple[int, int]] = set()
    displayed = PROPORTIONS[1:]
    for row, region_id in enumerate(REGION_ORDER):
        previous = assignments[(region_id, PROPORTIONS[0])].get("groups", [])
        for col, proportion in enumerate(displayed):
            assignment = assignments.get((region_id, proportion))
            if assignment is None:
                previous = None
                continue
            current = assignment.get("groups", [])
            if col > 0 and previous is not None and current == previous:
                unchanged.add((row, col))
            previous = current
    return unchanged


def annotate_heatmap(
    ax: mpl.axes.Axes,
    values: np.ndarray,
    norm: Normalize,
    formatter: str,
    unchanged: set[tuple[int, int]] | None = None,
) -> None:
    unchanged = unchanged or set()
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = float(values[row, col])
            rgba = ax.images[-1].cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "black" if luminance > 0.58 else "white"
            suffix = " =" if (row, col) in unchanged else ""
            ax.text(
                col,
                row,
                formatter.format(value) + suffix,
                ha="center",
                va="center",
                fontsize=6.6,
                fontweight="bold",
                color=color,
            )


def style_heatmap_axis(ax: mpl.axes.Axes, columns: list[str]) -> None:
    ax.set_xticks(np.arange(len(columns)), labels=columns)
    ax.set_yticks(np.arange(len(REGIONS)), labels=[display_name for _, display_name in REGIONS])
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(REGIONS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", width=0, length=0, labelsize=7.5)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")


def plot_relative_rmse(
    df: pd.DataFrame,
    assignments: dict[tuple[str, float], dict[str, Any]],
    output_dir: Path,
) -> None:
    table = df.pivot(index="target_region", columns="nominal_proportion", values="rmse").reindex(REGION_ORDER)
    baseline = table[0.0]
    improvement = table[PROPORTIONS[1:]].sub(baseline, axis=0).mul(-100).div(baseline, axis=0)
    values = improvement.to_numpy(float)
    limit = max(10.0, float(np.ceil(np.nanmax(np.abs(values)) / 5.0) * 5.0))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, ax = plt.subplots(figsize=(7.16, 4.2))
    image = ax.imshow(values, cmap=IMPROVEMENT_CMAP, norm=norm, aspect="auto")
    annotate_heatmap(ax, values, norm, "{:+.1f}%", unchanged_assignment_cells(assignments))
    style_heatmap_axis(ax, ["5%", "10%", "25%", "50%", "100%"])
    ax.set_xlabel("Target-region adaptation share", fontsize=9.0, fontweight="bold", labelpad=6)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.027, pad=0.025)
    colorbar.set_label("RMSE improvement relative to zero-shot (%)", fontsize=8.5, fontweight="bold")
    colorbar.ax.tick_params(labelsize=7.0, width=0.6)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.legend(
        handles=[Patch(facecolor="white", edgecolor="black", label="=  Same selected tracks as preceding share")],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        frameon=False,
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.18, right=0.91, top=0.99, bottom=0.17)
    save_figure(fig, output_dir, "Fig_S1_Relative_RMSE_Heatmap")


def plot_additional_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    specifications = [
        ("mae", "(a) MAE (m)", mpl.colormaps["cividis"], None, "{:.2f}"),
        ("bias", "(b) Bias (m)", mpl.colormaps["RdBu_r"], "center", "{:+.2f}"),
        ("r2", "(c) R²", mpl.colormaps["viridis"], None, "{:.2f}"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 8.35))
    for ax, (metric, title, cmap, mode, formatter) in zip(axes, specifications):
        table = df.pivot(index="target_region", columns="nominal_proportion", values=metric).reindex(REGION_ORDER)
        values = table[PROPORTIONS].to_numpy(float)
        if mode == "center":
            limit = float(np.nanmax(np.abs(values)))
            norm: Normalize = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        else:
            norm = Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
        image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        annotate_heatmap(ax, values, norm, formatter)
        style_heatmap_axis(ax, ["0%", "5%", "10%", "25%", "50%", "100%"])
        ax.set_title(title, fontsize=9.0, fontweight="bold", loc="left", pad=4)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.020, pad=0.025)
        colorbar.ax.tick_params(labelsize=7.0, width=0.6)
        for label in colorbar.ax.get_yticklabels():
            label.set_fontweight("bold")
    axes[-1].set_xlabel("Target-region adaptation share", fontsize=9.0, fontweight="bold", labelpad=6)
    fig.subplots_adjust(left=0.18, right=0.93, top=0.985, bottom=0.075, hspace=0.32)
    save_figure(fig, output_dir, "Fig_S2_Additional_Metrics_Heatmaps")


def export_tables(
    df: pd.DataFrame,
    assignments: dict[tuple[str, float], dict[str, Any]],
    output_dir: Path,
) -> None:
    table = df.copy()
    table["region"] = table["target_region"].astype(str).map(DISPLAY_NAMES)
    table["target_share_percent"] = table["nominal_proportion"].astype(float) * 100.0
    candidate_rows = table["candidate_adaptation_rows"].astype(float)
    table["actual_share_percent"] = np.where(
        candidate_rows > 0,
        100.0 * table["target_adaptation_rows"].astype(float) / candidate_rows,
        np.nan,
    )
    columns = [
        "region",
        "target_share_percent",
        "actual_share_percent",
        "target_adaptation_rows",
        "target_adaptation_tracks",
        "independent_validation_pixels",
        "rmse",
        "rmse_ci_low",
        "rmse_ci_high",
        "mae",
        "mae_ci_low",
        "mae_ci_high",
        "bias",
        "bias_ci_low",
        "bias_ci_high",
        "r2",
        "r2_ci_low",
        "r2_ci_high",
        "selection_seed",
        "model_seed",
        "bootstrap_blocks",
        "bootstrap_repeats",
        "validation_protocol",
        "validation_buffer_m",
    ]
    table[columns].to_csv(output_dir / "Table_S1_Regional_Adaptation_Metrics.csv", index=False)

    definition = table[
        [
            "region",
            "target_share_percent",
            "actual_share_percent",
            "target_adaptation_rows",
            "target_adaptation_tracks",
            "candidate_adaptation_rows",
            "candidate_adaptation_tracks",
        ]
    ].copy()
    definition["same_selected_tracks_as_preceding_share"] = False
    unchanged = unchanged_assignment_cells(assignments)
    for row, col in unchanged:
        region_id = REGION_ORDER[row]
        proportion = PROPORTIONS[1:][col]
        mask = table["target_region"].astype(str).eq(region_id) & np.isclose(
            table["nominal_proportion"].astype(float), proportion
        )
        definition.loc[mask, "same_selected_tracks_as_preceding_share"] = True
    definition.to_csv(output_dir / "Table_S2_Few_Shot_Definition.csv", index=False)


def write_reference_audit(result_dir: Path, output_dir: Path, reference_error: pd.DataFrame | None) -> None:
    figure05 = (
        PROJECT_ROOT
        / "Plot"
        / "Manuscript_Figures_Tables"
        / "Figure05_ICESat2_Bathymetric_Validation"
        / "figure05_bathymetric_validation_self_atl24_combined_all_tracks_tide_normalized_stats.csv"
    )
    paired = (
        PROJECT_ROOT
        / "outputs"
        / "ten_region_bathymetry"
        / "20261011T000000Z_selected_r2"
        / "report_paired_reference_metrics.csv"
    )
    audit = {
        "required_support_protocol": SUPPORT_PROTOCOL,
        "reference_error_band_drawn": reference_error is not None,
        "decision": (
            "draw validated same-support independent reference-disagreement band"
            if reference_error is not None
            else "omit reference-error band and use the 100% target-data empirical benchmark"
        ),
        "candidates": [
            {
                "path": str(figure05),
                "exists": figure05.exists(),
                "decision": "reject_for_main_figure",
                "reason": (
                    "Direct ICESat-2/CUDEM disagreement is measured on track support excluded by the fixed "
                    "500 m validation mask; it is not a same-support estimate."
                ),
            },
            {
                "path": str(paired),
                "exists": paired.exists(),
                "decision": "reject_for_main_figure",
                "reason": (
                    "The file contains model-to-ICESat-2 and model-to-CUDEM metrics, not direct "
                    "CUDEM-reference disagreement."
                ),
            },
            {
                "path": str(result_dir / "reference"),
                "exists": (result_dir / "reference").exists(),
                "decision": "insufficient",
                "reason": "The directory contains CUDEM depth rasters but no independent error or uncertainty layer.",
            },
            {
                "path": str(PROJECT_ROOT / "validation_area" / "data"),
                "exists": (PROJECT_ROOT / "validation_area" / "data").exists(),
                "decision": "reject_for_main_figure",
                "reason": (
                    "Independent IRMA DEM rasters are available for only four Florida regions. They were not "
                    "used because the ten-region figure requires one consistent benchmark definition."
                ),
            },
        ],
    }
    (output_dir / "CUDEM_reference_error_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


def write_captions(df: pd.DataFrame, output_dir: Path, reference_error: pd.DataFrame | None) -> None:
    total_pixels = int(df.groupby("target_region", observed=True)["independent_validation_pixels"].first().sum())
    benchmark_note = (
        "The gray band denotes an independently estimated CUDEM-reference disagreement on the exact evaluation support."
        if reference_error is not None
        else (
            "No same-support independent CUDEM reference-error estimate was available; dashed lines therefore show "
            "the empirical 100% target-data benchmark and are not theoretical lower bounds."
        )
    )
    caption = f"""# Figure captions

## Main figure

**Regional adaptation under spatially independent CUDEM evaluation.** Each panel shows RMSE for one leave-one-region-out task as complete target-region ICESat-2 tracks are added to the nine-region source training set. CUDEM evaluation uses a fixed region-specific mask outside a 500 m buffer around every candidate target-region adaptation track; the same pixels are used at all adaptation shares ({total_pixels:,} pixels across ten regions). Error bars are 95% spatial-block bootstrap intervals conditional on the selected track set and model run. Open circles denote zero-shot transfer, and dashed lines denote the 100% target-data empirical benchmark. {benchmark_note} Panel-specific vertical scales start at zero. The annotation gives the RMSE change from 0% to 100%, with negative values indicating lower error.

## Supplementary Figure S1

**RMSE change relative to zero-shot transfer.** Cells report 100 x (RMSE_0% - RMSE_p%) / RMSE_0%; positive values indicate improvement and negative values indicate degradation. An equals sign indicates that the selected target tracks and observations are identical to those in the preceding adaptation condition. Results are conditional on selection seed 42 and model seed 42.

## Supplementary Figure S2

**Additional spatially independent CUDEM metrics across regional adaptation conditions.** Cells show MAE, signed bias (prediction minus CUDEM), and R^2 on the same fixed region-specific evaluation support used in the main figure. The spatial-block confidence intervals and exact sample counts are reported in Supplementary Table S1.

## Supplementary Table S1

Exact target-region observation counts, complete-track counts, fixed validation-pixel counts, evaluation metrics, and 95% spatial-block bootstrap intervals. The intervals do not include track-selection or model-training variability.

## Supplementary Table S2

Exact nominal and realized adaptation shares, target-region point counts, complete-track counts, and full candidate-pool sizes for every region. The final column identifies conditions that reused the same selected tracks as the preceding nominal share.
"""
    (output_dir / "figure_captions.md").write_text(caption, encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = (args.output_dir or (result_dir / "publication_figures")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    metrics = load_metrics(result_dir)
    assignments = load_assignments(result_dir)
    reference_error = load_reference_error(args.reference_error_csv)
    plot_main_rmse(metrics, output_dir, reference_error)
    plot_relative_rmse(metrics, assignments, output_dir)
    plot_additional_metrics(metrics, output_dir)
    export_tables(metrics, assignments, output_dir)
    write_reference_audit(result_dir, output_dir, reference_error)
    write_captions(metrics, output_dir, reference_error)
    print(f"Wrote publication figures and supplements to {output_dir}")


if __name__ == "__main__":
    main()
