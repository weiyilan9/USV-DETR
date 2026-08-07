# -*- coding: utf-8 -*-
"""USV-DETR: high-resolution detection of rodent ultrasonic vocalizations.

Typical use:

    from usvdetr import load_model, detect_wav, save_records

    model, device = load_model("configs/USV-DETR.yml", "USV-DETR.pth")
    df = detect_wav(model, device, "recording.wav")
    save_records(df, "detections.xlsx")

The pipeline runs in four stages, one module each:

    spectrogram  audio in, RGB spectrogram images out
    model        images in, boxes in pixel coordinates out
    analysis     boxes in, a table in seconds and kHz out
    plotting     the table in, figures out

Everything listed in __all__ is re-exported here, so a demo notebook can
import from usvdetr directly. Import from the submodule instead when a name
would be ambiguous out of context, for example usvdetr.model.load_model.

Tunable defaults live at the top of each module as DEFAULT_ constants and
are also exposed as keyword arguments, so nothing needs editing in place.
"""

__version__ = "0.1.0"

from usvdetr.spectrogram import (
    AUDITION_CMAP,
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
    count_segments,
    iter_segments,
    load_audio,
    make_spectrogram,
    segment_to_rgb,
    slice_audio,
    spectrogram_to_rgb,
)

from usvdetr.model import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_INPUT_SIZE,
    DEFAULT_NMS_IOU,
    DEFAULT_SWAP_TO_BGR,
    denormalize_boxes,
    filter_detections,
    letterbox_resize,
    load_model,
    predict_on_rgb_image,
)

from usvdetr.analysis import (
    DEFAULT_CONTAINMENT_THRESH,
    DEFAULT_GAP_THRESH_SEC,
    DEFAULT_RATIO_MAX,
    DEFAULT_RATIO_MIN,
    DEFAULT_TIME_OVERLAP_THRESH,
    LABEL_HARMONIC,
    LABEL_USV,
    RECORD_COLUMNS,
    analyze_result,
    boxes_to_freq_khz,
    boxes_to_time_sec,
    classify_harmonics,
    detect_audio,
    detect_folder,
    detect_wav,
    harmonic_pairs,
    merge_contained_boxes,
    merge_into_bouts,
    result_to_records,
    save_records,
)

from usvdetr.plotting import (
    HARMONIC_COLOR,
    SAVE_DPI,
    USV_COLOR,
    plot_bout_timeline,
    plot_detection_timeline,
    show_prediction,
)


__all__ = [
    "__version__",

    # Stage 1: audio to spectrogram images
    "load_audio",
    "slice_audio",
    "count_segments",
    "iter_segments",
    "make_spectrogram",
    "spectrogram_to_rgb",
    "segment_to_rgb",
    "AUDITION_CMAP",

    # Stage 2: detection
    "load_model",
    "predict_on_rgb_image",
    "letterbox_resize",
    "denormalize_boxes",
    "filter_detections",

    # Stage 3: post-processing and batch runs
    "analyze_result",
    "result_to_records",
    "merge_contained_boxes",
    "classify_harmonics",
    "boxes_to_freq_khz",
    "boxes_to_time_sec",
    "detect_audio",
    "detect_wav",
    "detect_folder",
    "harmonic_pairs",
    "merge_into_bouts",
    "save_records",

    # Stage 4: figures
    "show_prediction",
    "plot_detection_timeline",
    "plot_bout_timeline",

    # Labels and table schema
    "LABEL_USV",
    "LABEL_HARMONIC",
    "RECORD_COLUMNS",

    # Defaults, for demos that want to show or override them
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_LOW_FREQ_HZ",
    "DEFAULT_HIGH_FREQ_HZ",
    "DEFAULT_WINDOW_SEC",
    "DEFAULT_STEP_SEC",
    "DEFAULT_NPERSEG",
    "DEFAULT_NOVERLAP",
    "DEFAULT_NFFT",
    "DEFAULT_GAMMA",
    "DEFAULT_VMIN_PCT",
    "DEFAULT_VMAX_PCT",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_CONF_THRESHOLD",
    "DEFAULT_NMS_IOU",
    "DEFAULT_SWAP_TO_BGR",
    "DEFAULT_CONTAINMENT_THRESH",
    "DEFAULT_TIME_OVERLAP_THRESH",
    "DEFAULT_RATIO_MIN",
    "DEFAULT_RATIO_MAX",
    "DEFAULT_GAP_THRESH_SEC",
    "USV_COLOR",
    "HARMONIC_COLOR",
    "SAVE_DPI",
]
