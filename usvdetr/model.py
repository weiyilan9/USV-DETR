# -*- coding: utf-8 -*-
"""Model loading and single-image inference for USV-DETR.

This module is the detection stage of the pipeline. It takes an RGB
spectrogram image produced by usvdetr.spectrogram and returns bounding
boxes in pixel coordinates, together with the time and frequency span the
image covers so that later stages can convert pixels to seconds and kHz.

The model definition itself lives in the upstream RT-DETRv4 codebase, which
is not a pip package. It is imported lazily inside load_model, so the rest
of this package works even when the upstream repository is not on sys.path.
"""

import numpy as np
import torch
import cv2
from torchvision.ops import nms


# ---------------------------------------------------------------------------
# Default parameters. Every function below takes these as keyword arguments,
# so a demo notebook can override any of them without editing this file.
# ---------------------------------------------------------------------------

DEFAULT_INPUT_SIZE = 640          # square side the model expects, pixels
DEFAULT_CONF_THRESHOLD = 0.6      # keep detections at or above this score
DEFAULT_NMS_IOU = 0.4             # IoU above which overlapping boxes merge

DEFAULT_SWAP_TO_BGR = True        # see the note in predict_on_rgb_image
LETTERBOX_PAD_VALUE = 0           # pixel value used for letterbox padding
RESIZE_INTERPOLATION = cv2.INTER_LINEAR


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config_file, checkpoint_path, device=None, verbose=True):
    """Build the model from a config file and load trained weights.

    Args:
        config_file: path to the USV-DETR yml. It must sit inside the
            upstream configs directory so that its __include__ paths resolve.
        checkpoint_path: path to the .pth checkpoint.
        device: "cuda", "cpu", or None to pick automatically.
        verbose: print device and parameter count.

    Returns:
        (model, device) with the model in eval mode on the chosen device.
    """
    # Upstream package is named engine in RT-DETRv4 and DEIM, src in
    # RT-DETRv2 and D-FINE. Try both so one config works with either.
    try:
        from engine.core import YAMLConfig
    except ImportError:
        from src.core import YAMLConfig

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = YAMLConfig(config_file)

    # The backbone would otherwise try to download ImageNet weights that the
    # checkpoint immediately overwrites.
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    model = cfg.model

    # torch 2.6 defaults to weights_only=True, which refuses training
    # checkpoints because they contain more than plain tensors.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Strip the prefix left by DistributedDataParallel training.
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model = model.to(device).eval()

    # Some tensors in HybridEncoder, such as the positional embedding, are
    # plain attributes rather than registered buffers, so .to(device) leaves
    # them on the CPU and inference fails with a device mismatch.
    _move_loose_tensors(model, device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print("missing keys: %d, unexpected keys: %d" % (len(missing), len(unexpected)))
        print("model loaded on %s, %.1f M parameters" % (device, n_params))

    return model, device


def _move_loose_tensors(model, device):
    """Move tensors held as plain attributes onto the target device."""
    for module in model.modules():
        for name, value in list(vars(module).items()):
            if torch.is_tensor(value):
                setattr(module, name, value.to(device))


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def letterbox_resize(image, target_size=DEFAULT_INPUT_SIZE):
    """Resize an image to a square canvas without distorting it.

    The image is scaled so its longer side matches target_size and then
    centred on a padded square canvas.

    Args:
        image: HxWx3 uint8 array.
        target_size: side length of the output canvas.

    Returns:
        (canvas, scale, pad_left, pad_top). The last three are needed to map
        predictions back to original pixel coordinates.
    """
    height, width = image.shape[:2]
    scale = target_size / max(height, width)

    new_height = max(1, int(height * scale))
    new_width = max(1, int(width * scale))
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=RESIZE_INTERPOLATION
    )

    canvas = np.full(
        (target_size, target_size, 3), LETTERBOX_PAD_VALUE, dtype=np.uint8
    )
    pad_top = (target_size - new_height) // 2
    pad_left = (target_size - new_width) // 2
    canvas[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = resized

    return canvas, scale, pad_left, pad_top


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def denormalize_boxes(
    boxes_cxcywh,
    image_width,
    image_height,
    scale,
    pad_left,
    pad_top,
    target_size=DEFAULT_INPUT_SIZE,
):
    """Map normalised model outputs back to original pixel coordinates.

    Args:
        boxes_cxcywh: Nx4 array of centre-x, centre-y, width, height, each
            normalised to [0, 1] against the letterboxed canvas.
        image_width, image_height: size of the original image.
        scale, pad_left, pad_top: values returned by letterbox_resize.
        target_size: canvas side length used during letterboxing.

    Returns:
        Nx4 array of x1, y1, x2, y2 in original pixel coordinates, clipped
        to the image bounds.
    """
    if len(boxes_cxcywh) == 0:
        return np.empty((0, 4), dtype=np.float32)

    center_x = boxes_cxcywh[:, 0] * target_size
    center_y = boxes_cxcywh[:, 1] * target_size
    width = boxes_cxcywh[:, 2] * target_size
    height = boxes_cxcywh[:, 3] * target_size

    x1 = (center_x - width / 2 - pad_left) / scale
    y1 = (center_y - height / 2 - pad_top) / scale
    x2 = (center_x + width / 2 - pad_left) / scale
    y2 = (center_y + height / 2 - pad_top) / scale

    x1 = np.clip(x1, 0, image_width)
    y1 = np.clip(y1, 0, image_height)
    x2 = np.clip(x2, 0, image_width)
    y2 = np.clip(y2, 0, image_height)

    return np.stack([x1, y1, x2, y2], axis=1)


def filter_detections(
    boxes,
    scores,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    nms_iou=DEFAULT_NMS_IOU,
):
    """Drop low-confidence boxes, then suppress overlapping duplicates.

    Args:
        boxes: Nx4 array of x1, y1, x2, y2.
        scores: length-N array of confidences.
        conf_threshold: minimum score to keep. Set to 0 to keep everything.
        nms_iou: IoU threshold for non-maximum suppression.

    Returns:
        (boxes, scores) sorted by the order NMS returns, highest score first.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    keep_mask = scores >= conf_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]

    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    keep_index = nms(
        torch.from_numpy(boxes),
        torch.from_numpy(scores),
        nms_iou,
    ).numpy()

    return boxes[keep_index], scores[keep_index]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_on_rgb_image(
    model,
    device,
    rgb_image,
    start_time_s,
    end_time_s,
    freq_min_khz,
    freq_max_khz,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    nms_iou=DEFAULT_NMS_IOU,
    input_size=DEFAULT_INPUT_SIZE,
    swap_to_bgr=DEFAULT_SWAP_TO_BGR,
):
    """Detect USVs in one RGB spectrogram image.

    Args:
        model, device: as returned by load_model.
        rgb_image: HxWx3 uint8 array from usvdetr.spectrogram.
        start_time_s, end_time_s: absolute time span the image covers.
        freq_min_khz, freq_max_khz: frequency span the image covers.
        conf_threshold: minimum detection score to keep.
        nms_iou: IoU threshold for non-maximum suppression.
        input_size: square side fed to the model.
        swap_to_bgr: reverse the channel order before inference. The default
            reproduces the original Colab pipeline, which passed a BGR array
            to the model. Whether this is correct depends on the channel
            order used when the training images were written, so it is worth
            comparing both settings on annotated data.

    Returns:
        A dict holding the image, the detections in pixel coordinates, the
        image size, and the time and frequency span. usvdetr.analysis
        consumes this dict directly.
    """
    image = rgb_image[..., ::-1] if swap_to_bgr else rgb_image
    image = np.ascontiguousarray(image)

    image_height, image_width = image.shape[:2]
    canvas, scale, pad_left, pad_top = letterbox_resize(image, input_size)

    image_tensor = torch.from_numpy(canvas).float().permute(2, 0, 1).unsqueeze(0)
    image_tensor = (image_tensor / 255.0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    logits = outputs["pred_logits"][0]
    boxes_cxcywh = outputs["pred_boxes"][0]

    # One class in the paper config, so the class dimension is size 1. Taking
    # the maximum keeps this correct if more classes are added later.
    class_scores = logits.sigmoid()
    if class_scores.shape[-1] == 1:
        scores = class_scores.squeeze(-1)
    else:
        scores = class_scores.max(dim=-1).values

    boxes = denormalize_boxes(
        boxes_cxcywh.cpu().numpy(),
        image_width,
        image_height,
        scale,
        pad_left,
        pad_top,
        input_size,
    )
    boxes, scores = filter_detections(
        boxes, scores.cpu().numpy(), conf_threshold, nms_iou
    )

    return {
        "image_rgb": rgb_image,
        "boxes": boxes,
        "scores": scores,
        "img_h": image_height,
        "img_w": image_width,
        "start_time": start_time_s,
        "end_time": end_time_s,
        "fmin_khz": freq_min_khz,
        "fmax_khz": freq_max_khz,
    }
