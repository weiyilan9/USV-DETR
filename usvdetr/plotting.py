# -*- coding: utf-8 -*-
"""Figures for USV-DETR results.

This module is the presentation stage of the pipeline. It draws detection
boxes on a single spectrogram and draws call timelines over a whole
recording.

Every colour, size and font setting is a keyword argument, so a demo
notebook can restyle any figure without editing this file. Pass save_path
to write the figure to disk at SAVE_DPI.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from usvdetr.analysis import (
    DEFAULT_CONTAINMENT_THRESH,
    DEFAULT_GAP_THRESH_SEC,
    DEFAULT_RATIO_MAX,
    DEFAULT_RATIO_MIN,
    DEFAULT_TIME_OVERLAP_THRESH,
    LABEL_HARMONIC,
    LABEL_USV,
    analyze_result,
    merge_into_bouts,
)


# ---------------------------------------------------------------------------
# Default style. Every function below takes these as keyword arguments.
# ---------------------------------------------------------------------------

USV_COLOR = "lime"
HARMONIC_COLOR = "#ccff00"
BG_COLOR = "black"
FG_COLOR = "white"

SAVE_DPI = 300              # resolution used when save_path is given

DEFAULT_SPECTROGRAM_FIGSIZE = (12, 4)
DEFAULT_TIMELINE_FIGSIZE = (24, 5)

DEFAULT_TITLE_FONTSIZE = 18
DEFAULT_LABEL_FONTSIZE = 16
DEFAULT_TICK_FONTSIZE = 14
DEFAULT_LEGEND_FONTSIZE = 14

DEFAULT_TIME_LABEL = "Time (s)"
DEFAULT_FREQ_LABEL = "Frequency (kHz)"
DEFAULT_MEAN_FREQ_LABEL = "Mean Frequency (kHz)"


def _label_colors(usv_color, harmonic_color):
    return {LABEL_USV: usv_color, LABEL_HARMONIC: harmonic_color}


def _finish(fig, save_path, dpi, bg_color, show):
    """Save and display a finished figure."""
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, facecolor=bg_color, bbox_inches="tight")
        print("saved figure: %s" % save_path)
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Single spectrogram with detection boxes
# ---------------------------------------------------------------------------

def show_prediction(
    result,
    title="Prediction",
    containment_thresh=DEFAULT_CONTAINMENT_THRESH,
    time_overlap_thresh=DEFAULT_TIME_OVERLAP_THRESH,
    ratio_min=DEFAULT_RATIO_MIN,
    ratio_max=DEFAULT_RATIO_MAX,
    xlabel=DEFAULT_TIME_LABEL,
    ylabel=DEFAULT_FREQ_LABEL,
    usv_color=USV_COLOR,
    harmonic_color=HARMONIC_COLOR,
    figsize=DEFAULT_SPECTROGRAM_FIGSIZE,
    box_linewidth=1.0,
    show_labels=True,
    show_scores=False,
    label_fontsize=7,
    title_fontsize=None,
    axis_label_fontsize=None,
    save_path=None,
    dpi=SAVE_DPI,
    show=True,
):
    """Draw one spectrogram with its detection boxes.

    Axes are in seconds and kHz rather than pixels, so a box can be read off
    the figure directly.

    Args:
        result: the dict returned by usvdetr.model.predict_on_rgb_image.
        title: figure title, or None for no title.
        containment_thresh, time_overlap_thresh, ratio_min, ratio_max:
            post-processing settings, see usvdetr.analysis.analyze_result.
        show_labels: draw the USV or Harmonic tag above each box.
        show_scores: append the confidence to that tag.
        save_path: write the figure here, or None to skip saving.
        dpi: resolution used when saving.
        show: call plt.show, set False in scripts that only save.

    Returns:
        (fig, ax) so the caller can adjust the figure further.
    """
    analysis = analyze_result(
        result, containment_thresh, time_overlap_thresh, ratio_min, ratio_max
    )
    colors = _label_colors(usv_color, harmonic_color)

    freq_min_khz = result["fmin_khz"]
    freq_max_khz = result["fmax_khz"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        result["image_rgb"],
        origin="lower",
        aspect="auto",
        extent=[result["start_time"], result["end_time"], freq_min_khz, freq_max_khz],
    )

    for i, label in enumerate(analysis["labels"]):
        onset = analysis["onset_s"][i]
        offset = analysis["offset_s"][i]
        low = analysis["freq_min_khz"][i]
        high = analysis["freq_max_khz"][i]

        ax.add_patch(patches.Rectangle(
            (onset, low),
            offset - onset,
            high - low,
            linewidth=box_linewidth,
            edgecolor=colors[label],
            facecolor="none",
        ))

        if show_labels:
            text = label
            if show_scores:
                text = "%s %.2f" % (label, analysis["scores"][i])
            ax.text(
                onset, high, text,
                color=FG_COLOR, fontsize=label_fontsize,
                verticalalignment="bottom",
                bbox=dict(facecolor=BG_COLOR, alpha=0.45, pad=1),
            )

    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    _finish(fig, save_path, dpi, "white", show)
    return fig, ax


# ---------------------------------------------------------------------------
# Timelines over a whole recording
# ---------------------------------------------------------------------------

def _setup_timeline_ax(
    ax,
    duration_s,
    freq_min_khz,
    freq_max_khz,
    usv_color,
    harmonic_color,
    bg_color,
    fg_color,
    xlabel,
    ylabel,
    show_legend,
    legend_fontsize,
    label_fontsize,
    tick_fontsize,
    y_pad_bottom,
    y_pad_top,
):
    """Apply the shared dark styling used by both timeline figures."""
    ax.set_facecolor(bg_color)
    ax.set_xlabel(xlabel, color=fg_color, fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel(ylabel, color=fg_color, fontsize=label_fontsize, fontweight="bold")
    ax.set_xlim(0, duration_s)
    ax.set_ylim(freq_min_khz - y_pad_bottom, freq_max_khz + y_pad_top)
    ax.tick_params(colors=fg_color, labelsize=tick_fontsize)
    ax.spines[:].set_color(fg_color)

    if show_legend:
        handles = [
            plt.Line2D([0], [0], color=harmonic_color, linewidth=5,
                       label=LABEL_HARMONIC),
            plt.Line2D([0], [0], color=usv_color, linewidth=5,
                       label=LABEL_USV),
        ]
        ax.legend(
            handles=handles, facecolor=bg_color, labelcolor=fg_color,
            framealpha=1, prop={"size": legend_fontsize, "weight": "bold"},
        )


def plot_detection_timeline(
    df,
    duration_s,
    freq_min_khz=20,
    freq_max_khz=100,
    title="USV and Harmonic Timeline",
    xlabel=DEFAULT_TIME_LABEL,
    ylabel=DEFAULT_MEAN_FREQ_LABEL,
    usv_color=USV_COLOR,
    harmonic_color=HARMONIC_COLOR,
    bg_color=BG_COLOR,
    fg_color=FG_COLOR,
    figsize=DEFAULT_TIMELINE_FIGSIZE,
    line_width=6,
    dot_size=60,
    alpha=1.0,
    show_dots=True,
    show_legend=True,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
    label_fontsize=DEFAULT_LABEL_FONTSIZE,
    tick_fontsize=DEFAULT_TICK_FONTSIZE,
    legend_fontsize=DEFAULT_LEGEND_FONTSIZE,
    y_pad_bottom=5,
    y_pad_top=5,
    save_path=None,
    dpi=SAVE_DPI,
    show=True,
):
    """Draw one horizontal line per detected call.

    Each line spans the call in time and sits at its mean frequency.

    Args:
        df: detection DataFrame from usvdetr.analysis.
        duration_s: length of the recording, used for the x axis limit.
        freq_min_khz, freq_max_khz: y axis range before padding.
        show_dots: mark the centre of each call.
        save_path, dpi, show: see show_prediction.

    Returns:
        (fig, ax).
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)

    _setup_timeline_ax(
        ax, duration_s, freq_min_khz, freq_max_khz,
        usv_color, harmonic_color, bg_color, fg_color,
        xlabel, ylabel, show_legend, legend_fontsize,
        label_fontsize, tick_fontsize, y_pad_bottom, y_pad_top,
    )

    for label, color in _label_colors(usv_color, harmonic_color).items():
        subset = df[df["label"] == label]
        for _, row in subset.iterrows():
            ax.plot(
                [row["onset_time_s"], row["offset_time_s"]],
                [row["freq_mean_khz"], row["freq_mean_khz"]],
                color=color, linewidth=line_width, alpha=alpha,
                solid_capstyle="round",
            )
        if show_dots and len(subset) > 0:
            centers = (subset["onset_time_s"] + subset["offset_time_s"]) / 2
            ax.scatter(
                centers, subset["freq_mean_khz"],
                color=color, s=dot_size, zorder=3, linewidths=0,
            )

    if title:
        ax.set_title(title, color=fg_color, fontsize=title_fontsize,
                     fontweight="bold")

    _finish(fig, save_path, dpi, bg_color, show)
    return fig, ax


def plot_bout_timeline(
    df,
    duration_s,
    freq_min_khz=20,
    freq_max_khz=100,
    gap_thresh_sec=DEFAULT_GAP_THRESH_SEC,
    title="USV and Harmonic Bouts Timeline",
    xlabel=DEFAULT_TIME_LABEL,
    ylabel=DEFAULT_MEAN_FREQ_LABEL,
    usv_color=USV_COLOR,
    harmonic_color=HARMONIC_COLOR,
    bg_color=BG_COLOR,
    fg_color=FG_COLOR,
    figsize=DEFAULT_TIMELINE_FIGSIZE,
    line_width=6,
    alpha=1.0,
    show_intervals=True,
    show_legend=True,
    drop_amount=18,
    text_offset=1.5,
    interval_fontsize=9,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
    label_fontsize=DEFAULT_LABEL_FONTSIZE,
    tick_fontsize=DEFAULT_TICK_FONTSIZE,
    legend_fontsize=DEFAULT_LEGEND_FONTSIZE,
    y_pad_top=5,
    verbose=True,
    save_path=None,
    dpi=SAVE_DPI,
    show=True,
):
    """Draw one horizontal line per bout of calls.

    Calls closer together than gap_thresh_sec are merged into one bout. When
    show_intervals is on, the silent gap between consecutive USV bouts is
    annotated below the lines.

    Args:
        df: detection DataFrame from usvdetr.analysis.
        duration_s: length of the recording.
        gap_thresh_sec: silence that separates two bouts.
        drop_amount: how far below the lines the interval brackets sit, in
            kHz. The y axis is extended by the same amount.
        verbose: print the bout counts.
        save_path, dpi, show: see show_prediction.

    Returns:
        (fig, ax). Call usvdetr.analysis.merge_into_bouts directly if the
        bout list itself is needed.
    """
    usv_bouts = merge_into_bouts(df[df["label"] == LABEL_USV], gap_thresh_sec)
    harmonic_bouts = merge_into_bouts(
        df[df["label"] == LABEL_HARMONIC], gap_thresh_sec
    )

    if verbose:
        print("USV bouts: %d, Harmonic bouts: %d"
              % (len(usv_bouts), len(harmonic_bouts)))

    y_pad_bottom = drop_amount + 5 if show_intervals else 5

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)

    _setup_timeline_ax(
        ax, duration_s, freq_min_khz, freq_max_khz,
        usv_color, harmonic_color, bg_color, fg_color,
        xlabel, ylabel, show_legend, legend_fontsize,
        label_fontsize, tick_fontsize, y_pad_bottom, y_pad_top,
    )

    for bouts, color in ((usv_bouts, usv_color), (harmonic_bouts, harmonic_color)):
        for bout in bouts:
            ax.plot(
                [bout["onset"], bout["offset"]], [bout["freq"], bout["freq"]],
                color=color, linewidth=line_width, alpha=alpha,
                solid_capstyle="butt",
            )

    if show_intervals:
        _annotate_intervals(
            ax, usv_bouts, fg_color, drop_amount, text_offset, interval_fontsize
        )

    if title:
        ax.set_title(title, color=fg_color, fontsize=title_fontsize,
                     fontweight="bold")

    _finish(fig, save_path, dpi, bg_color, show)
    return fig, ax


def _annotate_intervals(ax, bouts, fg_color, drop_amount, text_offset, fontsize):
    """Draw a bracket and a duration label in each gap between bouts."""
    dash_style = dict(color=fg_color, linewidth=1.5, linestyle="--", alpha=0.9)

    for i in range(len(bouts) - 1):
        current, following = bouts[i], bouts[i + 1]
        gap = following["onset"] - current["offset"]
        if gap <= 0:
            continue

        y_bottom = min(current["freq"], following["freq"]) - drop_amount

        ax.plot([current["offset"], current["offset"]],
                [current["freq"], y_bottom], **dash_style)
        ax.plot([following["onset"], following["onset"]],
                [following["freq"], y_bottom], **dash_style)
        ax.plot([current["offset"], following["onset"]],
                [y_bottom, y_bottom], **dash_style)

        ax.text(
            (current["offset"] + following["onset"]) / 2,
            y_bottom - text_offset,
            "%.1fs" % gap,
            color=fg_color, fontsize=fontsize, fontweight="bold",
            ha="center", va="top",
        )
