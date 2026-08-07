# -*- coding: utf-8 -*-
"""Post-processing and batch analysis for USV-DETR.

This module is the interpretation stage of the pipeline. It cleans up raw
detections, decides which boxes are harmonics of a lower fundamental,
converts pixel boxes into seconds and kHz, and runs whole files or whole
folders into a single table.

Pipeline position:
    spectrogram.py -> model.py -> analysis.py -> plotting.py
"""

import os

import numpy as np
import pandas as pd

from usvdetr.spectrogram import (
    DEFAULT_HIGH_FREQ_HZ,
    DEFAULT_LOW_FREQ_HZ,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_STEP_SEC,
    DEFAULT_WINDOW_SEC,
    iter_segments,
    load_audio,
    make_spectrogram,
    spectrogram_to_rgb,
)
from usvdetr.model import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_INPUT_SIZE,
    DEFAULT_NMS_IOU,
    DEFAULT_SWAP_TO_BGR,
    predict_on_rgb_image,
)


# ---------------------------------------------------------------------------
# Default parameters. Every function below takes these as keyword arguments,
# so a demo notebook can override any of them without editing this file.
# ---------------------------------------------------------------------------

LABEL_USV = "USV"
LABEL_HARMONIC = "Harmonic"

DEFAULT_CONTAINMENT_THRESH = 0.8    # drop a box this covered by a larger one
DEFAULT_TIME_OVERLAP_THRESH = 0.7   # time overlap required to call a harmonic
DEFAULT_RATIO_MIN = 1.80            # accepted frequency ratio, lower bound
DEFAULT_RATIO_MAX = 2.20            # accepted frequency ratio, upper bound
DEFAULT_GAP_THRESH_SEC = 1.0        # silence that separates two bouts

# Columns of the detection table, in output order. The first eight are
# enough to reload the exact spectrogram patch of any single call.
RECORD_COLUMNS = [
    "filename",
    "audio_duration_s",
    "seg_start_s",
    "seg_end_s",
    "onset_time_s",
    "offset_time_s",
    "duration_ms",
    "freq_min_khz",
    "freq_max_khz",
    "freq_mean_khz",
    "freq_bandwidth_khz",
    "label",
    "pair_id",
    "confidence",
]


# ---------------------------------------------------------------------------
# Box cleanup
# ---------------------------------------------------------------------------

def merge_contained_boxes(boxes, scores, containment_thresh=DEFAULT_CONTAINMENT_THRESH):
    """Drop boxes that are mostly swallowed by a larger box.

    NMS only removes boxes with a high mutual IoU. A small box sitting
    entirely inside a long one has a low IoU and survives, so it is removed
    here instead.

    Args:
        boxes: Nx4 array of x1, y1, x2, y2.
        scores: length-N array of confidences.
        containment_thresh: drop a box when this fraction of its own area
            lies inside another surviving box.

    Returns:
        (boxes, scores) with contained boxes removed.
    """
    if len(boxes) == 0:
        return boxes, scores

    n_boxes = len(boxes)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = np.ones(n_boxes, dtype=bool)

    for i in range(n_boxes):
        if not keep[i]:
            continue
        for j in range(n_boxes):
            if i == j or not keep[j]:
                continue
            inter_x1 = max(boxes[i, 0], boxes[j, 0])
            inter_y1 = max(boxes[i, 1], boxes[j, 1])
            inter_x2 = min(boxes[i, 2], boxes[j, 2])
            inter_y2 = min(boxes[i, 3], boxes[j, 3])
            intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
            if areas[i] > 0 and intersection / areas[i] >= containment_thresh:
                keep[i] = False
                break

    return boxes[keep], scores[keep]


# ---------------------------------------------------------------------------
# Pixel to physical units
# ---------------------------------------------------------------------------

def boxes_to_freq_khz(boxes, image_height, freq_min_khz, freq_max_khz):
    """Convert box row coordinates into frequencies.

    Row 0 of the image is freq_min_khz, so the mapping is affine rather than
    proportional. A box halfway up a 20 to 100 kHz image is at 60 kHz, not
    at 50 kHz.

    Returns:
        (low_khz, high_khz), each a length-N array.
    """
    if len(boxes) == 0:
        return np.empty((0,)), np.empty((0,))

    span = freq_max_khz - freq_min_khz
    edge_a = freq_min_khz + (boxes[:, 1] / image_height) * span
    edge_b = freq_min_khz + (boxes[:, 3] / image_height) * span
    return np.minimum(edge_a, edge_b), np.maximum(edge_a, edge_b)


def boxes_to_time_sec(boxes, image_width, start_time_s, end_time_s):
    """Convert box column coordinates into absolute times.

    Returns:
        (onset_s, offset_s), each a length-N array.
    """
    if len(boxes) == 0:
        return np.empty((0,)), np.empty((0,))

    span = end_time_s - start_time_s
    onset = start_time_s + (boxes[:, 0] / image_width) * span
    offset = start_time_s + (boxes[:, 2] / image_width) * span
    return onset, offset


# ---------------------------------------------------------------------------
# Harmonic classification
# ---------------------------------------------------------------------------

def classify_harmonics(
    boxes,
    freq_mean_khz,
    time_overlap_thresh=DEFAULT_TIME_OVERLAP_THRESH,
    ratio_min=DEFAULT_RATIO_MIN,
    ratio_max=DEFAULT_RATIO_MAX,
):
    """Label each box as a call or as the harmonic of a lower call.

    A box is only called a harmonic when a partner exists that overlaps it in
    time and sits at roughly half its frequency. Boxes are visited from high
    to low frequency, and each fundamental can be claimed once, so a stack of
    three overlapping calls does not collapse into one chain.

    Args:
        boxes: Nx4 array of x1, y1, x2, y2 in pixels.
        freq_mean_khz: length-N array of mean frequencies.
        time_overlap_thresh: required overlap, as a fraction of the shorter
            of the two boxes.
        ratio_min, ratio_max: accepted range for the frequency ratio between
            the candidate harmonic and its fundamental.

    Returns:
        (labels, partner_index) where labels holds LABEL_USV or
        LABEL_HARMONIC, and partner_index holds the paired index or None.
    """
    n_boxes = len(boxes)
    labels = [LABEL_USV] * n_boxes
    partner_index = [None] * n_boxes

    if n_boxes == 0:
        return labels, partner_index

    claimed = set()
    high_to_low = np.argsort(-freq_mean_khz)

    for i in high_to_low:
        best_j = None
        best_error = None

        for j in range(n_boxes):
            if j == i or j in claimed or labels[j] == LABEL_HARMONIC:
                continue
            if freq_mean_khz[j] >= freq_mean_khz[i]:
                continue

            overlap = min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0])
            shorter = min(boxes[i, 2] - boxes[i, 0], boxes[j, 2] - boxes[j, 0])
            if shorter <= 0 or overlap / shorter < time_overlap_thresh:
                continue

            ratio = freq_mean_khz[i] / freq_mean_khz[j]
            if ratio < ratio_min or ratio > ratio_max:
                continue

            error = abs(ratio - 2.0)
            if best_error is None or error < best_error:
                best_error = error
                best_j = j

        if best_j is not None:
            labels[i] = LABEL_HARMONIC
            partner_index[i] = best_j
            partner_index[best_j] = i
            claimed.add(best_j)

    return labels, partner_index


# ---------------------------------------------------------------------------
# Single image analysis
# ---------------------------------------------------------------------------

def analyze_result(
    result,
    containment_thresh=DEFAULT_CONTAINMENT_THRESH,
    time_overlap_thresh=DEFAULT_TIME_OVERLAP_THRESH,
    ratio_min=DEFAULT_RATIO_MIN,
    ratio_max=DEFAULT_RATIO_MAX,
):
    """Run the full post-processing chain on one detection result.

    Args:
        result: the dict returned by usvdetr.model.predict_on_rgb_image.
        containment_thresh: see merge_contained_boxes.
        time_overlap_thresh, ratio_min, ratio_max: see classify_harmonics.

    Returns:
        A dict with cleaned boxes, scores, labels, partner indices, and the
        onset, offset and frequency edges of every box in physical units.
    """
    boxes, scores = merge_contained_boxes(
        np.array(result["boxes"], dtype=float),
        np.array(result["scores"], dtype=float),
        containment_thresh,
    )

    freq_min_khz, freq_max_khz = boxes_to_freq_khz(
        boxes, result["img_h"], result["fmin_khz"], result["fmax_khz"]
    )
    onset_s, offset_s = boxes_to_time_sec(
        boxes, result["img_w"], result["start_time"], result["end_time"]
    )
    freq_mean_khz = (freq_min_khz + freq_max_khz) / 2

    labels, partner_index = classify_harmonics(
        boxes, freq_mean_khz, time_overlap_thresh, ratio_min, ratio_max
    )

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "partner_index": partner_index,
        "onset_s": onset_s,
        "offset_s": offset_s,
        "freq_min_khz": freq_min_khz,
        "freq_max_khz": freq_max_khz,
        "freq_mean_khz": freq_mean_khz,
    }


def result_to_records(
    result,
    containment_thresh=DEFAULT_CONTAINMENT_THRESH,
    time_overlap_thresh=DEFAULT_TIME_OVERLAP_THRESH,
    ratio_min=DEFAULT_RATIO_MIN,
    ratio_max=DEFAULT_RATIO_MAX,
):
    """Turn one detection result into a list of table rows.

    pair_id is local to this image. detect_audio offsets it so that ids stay
    unique across the whole file.

    Returns:
        A list of dicts, one per detection.
    """
    analysis = analyze_result(
        result, containment_thresh, time_overlap_thresh, ratio_min, ratio_max
    )

    labels = analysis["labels"]
    partner_index = analysis["partner_index"]

    pair_id = [None] * len(labels)
    next_pair_id = 0
    for i, label in enumerate(labels):
        if label == LABEL_HARMONIC and partner_index[i] is not None:
            pair_id[i] = next_pair_id
            pair_id[partner_index[i]] = next_pair_id
            next_pair_id += 1

    records = []
    for i in range(len(labels)):
        onset = float(analysis["onset_s"][i])
        offset = float(analysis["offset_s"][i])
        low = float(analysis["freq_min_khz"][i])
        high = float(analysis["freq_max_khz"][i])

        records.append({
            "seg_start_s": round(float(result["start_time"]), 4),
            "seg_end_s": round(float(result["end_time"]), 4),
            "onset_time_s": round(onset, 4),
            "offset_time_s": round(offset, 4),
            "duration_ms": round((offset - onset) * 1000, 2),
            "freq_min_khz": round(low, 2),
            "freq_max_khz": round(high, 2),
            "freq_mean_khz": round((low + high) / 2, 2),
            "freq_bandwidth_khz": round(high - low, 2),
            "label": labels[i],
            "pair_id": pair_id[i],
            "confidence": round(float(analysis["scores"][i]), 4),
        })

    return records


# ---------------------------------------------------------------------------
# Whole file and whole folder analysis
# ---------------------------------------------------------------------------

def detect_audio(
    model,
    device,
    audio,
    sample_rate,
    window_sec=DEFAULT_WINDOW_SEC,
    step_sec=DEFAULT_STEP_SEC,
    low_freq_hz=DEFAULT_LOW_FREQ_HZ,
    high_freq_hz=DEFAULT_HIGH_FREQ_HZ,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    nms_iou=DEFAULT_NMS_IOU,
    input_size=DEFAULT_INPUT_SIZE,
    swap_to_bgr=DEFAULT_SWAP_TO_BGR,
    containment_thresh=DEFAULT_CONTAINMENT_THRESH,
    time_overlap_thresh=DEFAULT_TIME_OVERLAP_THRESH,
    ratio_min=DEFAULT_RATIO_MIN,
    ratio_max=DEFAULT_RATIO_MAX,
    time_offset_sec=0.0,
    spectrogram_kwargs=None,
    progress_every=0,
):
    """Run detection window by window over an audio array.

    Note that step_sec below window_sec makes windows overlap, and a call
    landing in the overlap is reported once per window. Deduplication is not
    performed here.

    Args:
        model, device: as returned by usvdetr.model.load_model.
        audio, sample_rate: mono audio and its sample rate.
        window_sec, step_sec: windowing, see spectrogram.iter_segments.
        low_freq_hz, high_freq_hz: displayed frequency band.
        conf_threshold, nms_iou, input_size, swap_to_bgr: detector settings.
        containment_thresh, time_overlap_thresh, ratio_min, ratio_max:
            post-processing settings.
        time_offset_sec: added to every timestamp, for analysing a slice of
            a longer recording.
        spectrogram_kwargs: dict of extra arguments for make_spectrogram,
            such as nperseg, nfft or gamma.
        progress_every: print progress every N windows, 0 to stay silent.

    Returns:
        A DataFrame with one row per detection. Empty input still returns a
        DataFrame carrying the full set of columns.
    """
    spectrogram_kwargs = spectrogram_kwargs or {}

    all_records = []
    pair_id_offset = 0

    segments = iter_segments(
        audio, sample_rate, window_sec, step_sec, time_offset_sec=time_offset_sec
    )

    for index, (start_time_s, end_time_s, segment) in enumerate(segments, 1):
        freqs_hz, _, power_01 = make_spectrogram(
            segment,
            sample_rate,
            low_freq_hz=low_freq_hz,
            high_freq_hz=high_freq_hz,
            **spectrogram_kwargs
        )
        rgb_image = spectrogram_to_rgb(power_01)

        result = predict_on_rgb_image(
            model,
            device,
            rgb_image,
            start_time_s,
            end_time_s,
            freqs_hz[0] / 1000.0,
            freqs_hz[-1] / 1000.0,
            conf_threshold=conf_threshold,
            nms_iou=nms_iou,
            input_size=input_size,
            swap_to_bgr=swap_to_bgr,
        )

        records = result_to_records(
            result, containment_thresh, time_overlap_thresh, ratio_min, ratio_max
        )

        # Shift local pair ids so they never collide across windows.
        highest_id = -1
        for record in records:
            if record["pair_id"] is not None:
                record["pair_id"] += pair_id_offset
                highest_id = max(highest_id, record["pair_id"])
        if highest_id >= 0:
            pair_id_offset = highest_id + 1

        all_records.extend(records)

        if progress_every and index % progress_every == 0:
            print("  processed %d windows" % index)

    return pd.DataFrame(all_records, columns=RECORD_COLUMNS)


def detect_wav(
    model,
    device,
    wav_path,
    expected_sample_rate=DEFAULT_SAMPLE_RATE,
    verbose=True,
    **kwargs
):
    """Detect USVs in one wav file.

    Args:
        model, device: as returned by usvdetr.model.load_model.
        wav_path: path to the wav file.
        expected_sample_rate: raise if the file does not match. Pass None to
            accept any sample rate.
        verbose: print a one-line summary per file.
        kwargs: forwarded to detect_audio.

    Returns:
        A DataFrame with filename and audio_duration_s filled in.
    """
    audio, sample_rate = load_audio(wav_path, expected_sample_rate)
    duration_s = len(audio) / sample_rate
    filename = os.path.basename(wav_path)

    df = detect_audio(model, device, audio, sample_rate, **kwargs)
    df["filename"] = filename
    df["audio_duration_s"] = round(duration_s, 2)

    if verbose:
        n_usv = int((df["label"] == LABEL_USV).sum())
        n_harmonic = int((df["label"] == LABEL_HARMONIC).sum())
        print("%s | %.1fs | USV %d | Harmonic %d"
              % (filename, duration_s, n_usv, n_harmonic))

    return df


def detect_folder(
    model,
    device,
    wav_dir,
    expected_sample_rate=DEFAULT_SAMPLE_RATE,
    recursive=False,
    verbose=True,
    **kwargs
):
    """Detect USVs in every wav file of a folder.

    Files that cannot be read, for example because of a sample rate
    mismatch, are reported and skipped rather than aborting the run.

    Args:
        wav_dir: folder to scan.
        recursive: also walk sub-folders.
        kwargs: forwarded to detect_wav.

    Returns:
        One DataFrame holding the detections of all files.
    """
    wav_paths = _list_wav_files(wav_dir, recursive)
    if verbose:
        print("found %d wav files" % len(wav_paths))

    frames = []
    for wav_path in wav_paths:
        try:
            df = detect_wav(
                model, device, wav_path, expected_sample_rate,
                verbose=verbose, **kwargs
            )
        except (ValueError, RuntimeError) as error:
            print("skipped %s: %s" % (os.path.basename(wav_path), error))
            continue
        if len(df) > 0:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=RECORD_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def _list_wav_files(wav_dir, recursive=False):
    """Return sorted paths of the wav files in a folder."""
    if not recursive:
        names = sorted(f for f in os.listdir(wav_dir) if f.lower().endswith(".wav"))
        return [os.path.join(wav_dir, name) for name in names]

    paths = []
    for root, _, names in os.walk(wav_dir):
        for name in names:
            if name.lower().endswith(".wav"):
                paths.append(os.path.join(root, name))
    return sorted(paths)


# ---------------------------------------------------------------------------
# Summaries over a detection table
# ---------------------------------------------------------------------------

def harmonic_pairs(df):
    """Pair every harmonic with its fundamental and report the ratio.

    Useful for checking that the ratio window in classify_harmonics is set
    sensibly for a given recording setup.

    Returns:
        A DataFrame with pair_id, both mean frequencies and their ratio.
    """
    paired = df[df["pair_id"].notna()]
    fundamentals = paired[paired["label"] == LABEL_USV][["pair_id", "freq_mean_khz"]]
    harmonics = paired[paired["label"] == LABEL_HARMONIC][["pair_id", "freq_mean_khz"]]

    pairs = fundamentals.merge(
        harmonics, on="pair_id", suffixes=("_usv", "_harmonic")
    )
    pairs["ratio"] = pairs["freq_mean_khz_harmonic"] / pairs["freq_mean_khz_usv"]
    return pairs


def merge_into_bouts(df, gap_thresh_sec=DEFAULT_GAP_THRESH_SEC):
    """Group calls separated by less than gap_thresh_sec into bouts.

    Args:
        df: detection rows, typically already filtered to one label.
        gap_thresh_sec: silence longer than this starts a new bout.

    Returns:
        A list of dicts with onset, offset, mean frequency and call count.
    """
    if len(df) == 0:
        return []

    df = df.sort_values("onset_time_s").reset_index(drop=True)

    bouts = []
    bout_onset = df.iloc[0]["onset_time_s"]
    bout_offset = df.iloc[0]["offset_time_s"]
    freq_values = [df.iloc[0]["freq_mean_khz"]]

    for i in range(1, len(df)):
        row = df.iloc[i]
        if row["onset_time_s"] - bout_offset <= gap_thresh_sec:
            bout_offset = max(bout_offset, row["offset_time_s"])
            freq_values.append(row["freq_mean_khz"])
        else:
            bouts.append(_make_bout(bout_onset, bout_offset, freq_values))
            bout_onset = row["onset_time_s"]
            bout_offset = row["offset_time_s"]
            freq_values = [row["freq_mean_khz"]]

    bouts.append(_make_bout(bout_onset, bout_offset, freq_values))
    return bouts


def _make_bout(onset, offset, freq_values):
    return {
        "onset": float(onset),
        "offset": float(offset),
        "freq": float(np.mean(freq_values)),
        "n_calls": len(freq_values),
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def save_records(df, output_path, columns=None, preview_rows=5):
    """Write the detection table to xlsx or csv.

    The format follows the file extension of output_path.

    Args:
        df: detection DataFrame.
        output_path: destination file, ending in .xlsx or .csv.
        columns: column order, defaults to RECORD_COLUMNS.
        preview_rows: print this many rows after saving, 0 to stay silent.

    Returns:
        The DataFrame as written, sorted by filename and onset time.
    """
    if len(df) == 0:
        print("nothing to save")
        return df

    columns = columns or RECORD_COLUMNS
    columns = [name for name in columns if name in df.columns]

    sort_keys = [k for k in ("filename", "onset_time_s") if k in df.columns]
    df_out = df[columns].sort_values(sort_keys).reset_index(drop=True)

    if output_path.lower().endswith(".csv"):
        df_out.to_csv(output_path, index=False)
    else:
        df_out.to_excel(output_path, index=False)

    print("saved: %s (%d rows)" % (output_path, len(df_out)))

    if preview_rows:
        print(df_out.head(preview_rows).to_string(index=False))

    return df_out
