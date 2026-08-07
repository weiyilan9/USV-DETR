# -*- coding: utf-8 -*-
"""Command line interface for USV-DETR.

Run detection on one wav file or a whole folder and write a table of calls.

    python detect.py recording.wav --config configs/USV-DETR.yml \
        --checkpoint USV-DETR.pth

The config file must sit inside the upstream RT-DETRv4 configs directory,
because its __include__ lines are resolved relative to its own location.

Every tunable parameter of the pipeline is exposed as a flag. Defaults come
from the modules themselves, so this file never repeats a number.
"""

import argparse
import os
import sys

import matplotlib

from usvdetr import __version__
from usvdetr.analysis import (
    DEFAULT_CONTAINMENT_THRESH,
    DEFAULT_GAP_THRESH_SEC,
    DEFAULT_RATIO_MAX,
    DEFAULT_RATIO_MIN,
    DEFAULT_TIME_OVERLAP_THRESH,
    detect_folder,
    detect_wav,
    save_records,
)
from usvdetr.model import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_INPUT_SIZE,
    DEFAULT_NMS_IOU,
    DEFAULT_SWAP_TO_BGR,
    load_model,
)
from usvdetr.spectrogram import (
    DEFAULT_GAMMA,
    DEFAULT_HIGH_FREQ_HZ,
    DEFAULT_LOW_FREQ_HZ,
    DEFAULT_NFFT,
    DEFAULT_NOVERLAP,
    DEFAULT_NPERSEG,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_STEP_SEC,
    DEFAULT_VMAX_PCT,
    DEFAULT_VMIN_PCT,
    DEFAULT_WINDOW_SEC,
)


DEFAULT_OUTPUT_NAME = "detections.xlsx"


def build_parser():
    """Define every command line flag."""
    parser = argparse.ArgumentParser(
        prog="usvdetr-detect",
        description="Detect ultrasonic vocalizations in wav files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)

    parser.add_argument(
        "input",
        help="wav file, or a folder of wav files",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--config", required=True,
        help="USV-DETR yml, inside the upstream configs directory",
    )
    model_group.add_argument(
        "--checkpoint", required=True,
        help="trained .pth checkpoint",
    )
    model_group.add_argument(
        "--device", default=None,
        help="cuda or cpu, default picks cuda when available",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--output", default=None,
        help="table to write, .xlsx or .csv, default is next to the input",
    )
    output_group.add_argument(
        "--plot-dir", default=None,
        help="write per-file timeline figures into this folder",
    )
    output_group.add_argument(
        "--dpi", type=int, default=None,
        help="figure resolution, default is the package setting",
    )
    output_group.add_argument(
        "--quiet", action="store_true",
        help="suppress per-file progress lines",
    )

    audio_group = parser.add_argument_group("audio and spectrogram")
    audio_group.add_argument(
        "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
        help="expected wav sample rate, 0 to accept any",
    )
    audio_group.add_argument("--window", type=float, default=DEFAULT_WINDOW_SEC,
                             help="window length in seconds")
    audio_group.add_argument("--step", type=float, default=DEFAULT_STEP_SEC,
                             help="hop between windows in seconds")
    audio_group.add_argument("--low-freq", type=int, default=DEFAULT_LOW_FREQ_HZ,
                             help="lower edge of the frequency band in Hz")
    audio_group.add_argument("--high-freq", type=int, default=DEFAULT_HIGH_FREQ_HZ,
                             help="upper edge of the frequency band in Hz")
    audio_group.add_argument("--nperseg", type=int, default=DEFAULT_NPERSEG,
                             help="STFT window length in samples")
    audio_group.add_argument("--noverlap", type=int, default=DEFAULT_NOVERLAP,
                             help="STFT overlap in samples")
    audio_group.add_argument("--nfft", type=int, default=DEFAULT_NFFT,
                             help="FFT length")
    audio_group.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                             help="spectrogram contrast exponent")
    audio_group.add_argument("--vmin-pct", type=float, default=DEFAULT_VMIN_PCT,
                             help="percentile mapped to black")
    audio_group.add_argument("--vmax-pct", type=float, default=DEFAULT_VMAX_PCT,
                             help="percentile mapped to full brightness")

    detect_group = parser.add_argument_group("detection")
    detect_group.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD,
                              help="confidence threshold")
    detect_group.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU,
                              help="NMS IoU threshold")
    detect_group.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE,
                              help="square side fed to the model")
    detect_group.add_argument(
        "--no-swap-bgr", dest="swap_to_bgr",
        action="store_false", default=DEFAULT_SWAP_TO_BGR,
        help="feed the model RGB instead of BGR",
    )

    post_group = parser.add_argument_group("post-processing")
    post_group.add_argument("--containment", type=float,
                            default=DEFAULT_CONTAINMENT_THRESH,
                            help="drop a box this covered by a larger one")
    post_group.add_argument("--time-overlap", type=float,
                            default=DEFAULT_TIME_OVERLAP_THRESH,
                            help="time overlap required to call a harmonic")
    post_group.add_argument("--ratio-min", type=float, default=DEFAULT_RATIO_MIN,
                            help="lowest accepted harmonic frequency ratio")
    post_group.add_argument("--ratio-max", type=float, default=DEFAULT_RATIO_MAX,
                            help="highest accepted harmonic frequency ratio")
    post_group.add_argument("--gap", type=float, default=DEFAULT_GAP_THRESH_SEC,
                            help="silence that separates two bouts, seconds")

    parser.add_argument(
        "--recursive", action="store_true",
        help="when the input is a folder, also walk sub-folders",
    )

    return parser


def resolve_output_path(input_path, output):
    """Pick where the table goes when the user did not say."""
    if output:
        return output

    if os.path.isdir(input_path):
        return os.path.join(input_path, DEFAULT_OUTPUT_NAME)

    stem = os.path.splitext(input_path)[0]
    return stem + "_detections.xlsx"


def run_detection(args, model, device):
    """Run the detector over the input path and return the table."""
    shared = dict(
        window_sec=args.window,
        step_sec=args.step,
        low_freq_hz=args.low_freq,
        high_freq_hz=args.high_freq,
        conf_threshold=args.conf,
        nms_iou=args.nms_iou,
        input_size=args.input_size,
        swap_to_bgr=args.swap_to_bgr,
        containment_thresh=args.containment,
        time_overlap_thresh=args.time_overlap,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        spectrogram_kwargs=dict(
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            nfft=args.nfft,
            gamma=args.gamma,
            vmin_pct=args.vmin_pct,
            vmax_pct=args.vmax_pct,
        ),
    )

    expected_sample_rate = args.sample_rate or None
    verbose = not args.quiet

    if os.path.isdir(args.input):
        return detect_folder(
            model, device, args.input, expected_sample_rate,
            recursive=args.recursive, verbose=verbose, **shared
        )

    return detect_wav(
        model, device, args.input, expected_sample_rate,
        verbose=verbose, **shared
    )


def save_plots(df, plot_dir, freq_min_khz, freq_max_khz, gap_thresh_sec, dpi):
    """Write a detection timeline and a bout timeline for every file."""
    # Import here so the backend is fixed before pyplot loads.
    from usvdetr.plotting import (
        SAVE_DPI, plot_bout_timeline, plot_detection_timeline,
    )

    dpi = dpi or SAVE_DPI
    os.makedirs(plot_dir, exist_ok=True)

    for filename, group in df.groupby("filename"):
        duration_s = float(group["audio_duration_s"].iloc[0])
        stem = os.path.splitext(filename)[0]

        plot_detection_timeline(
            group, duration_s, freq_min_khz, freq_max_khz,
            title="%s detections" % stem,
            save_path=os.path.join(plot_dir, stem + "_timeline.png"),
            dpi=dpi, show=False,
        )
        plot_bout_timeline(
            group, duration_s, freq_min_khz, freq_max_khz,
            gap_thresh_sec=gap_thresh_sec,
            title="%s bouts" % stem,
            save_path=os.path.join(plot_dir, stem + "_bouts.png"),
            dpi=dpi, show=False, verbose=False,
        )


def main(argv=None):
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.input):
        print("input not found: %s" % args.input, file=sys.stderr)
        return 1

    # No interactive display in a command line run.
    matplotlib.use("Agg")

    model, device = load_model(
        args.config, args.checkpoint, args.device, verbose=not args.quiet
    )

    df = run_detection(args, model, device)

    if len(df) == 0:
        print("no detections")
        return 0

    output_path = resolve_output_path(args.input, args.output)
    save_records(df, output_path, preview_rows=0 if args.quiet else 5)

    if args.plot_dir:
        save_plots(
            df, args.plot_dir,
            args.low_freq / 1000.0, args.high_freq / 1000.0,
            args.gap, args.dpi,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
