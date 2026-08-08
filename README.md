# USV-DETR

USV-DETR is a tool for detecting rodent ultrasonic vocalizations. It locates
every call in a spectrogram and reports its timing and frequency precisely
enough to feed downstream acoustic analysis. The model is built on RT-DETR with
a high-resolution P2 feature layer and the DEIM training framework, and we have
tested it on both rat and mouse spectrograms.

<p align="center">
  <img src="configs/USV-DETR_model_structure.png"
       alt="USV-DETR model structure" width="80%">
</p>

It is open source and free to use. The quickest way to try it is the notebook in
[`demo/`](demo), which runs end to end in Google Colab without any local setup.
Beyond that, adapt it to your own setup, or take the checkpoint as a starting
point for fine-tuning on your own data. We hope it is useful in your research.

## Installation

A CUDA-capable GPU is required. CPU inference works, but is slow on long
recordings.

The model definition lives in the upstream RT-DETRv4 codebase, so both
repositories are needed.

```bash
git clone https://github.com/weiyilan9/USV-DETR.git
git clone --depth 1 https://github.com/RT-DETRs/RT-DETRv4.git

cd USV-DETR
pip install -e .
export PYTHONPATH=/path/to/RT-DETRv4:$PYTHONPATH
```

## Model weights

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download("yilanwei/USV-DETR", "USV-DETR.pth")
```

The config resolves its `__include__` lines relative to its own location, so a
copy outside `RT-DETRv4/configs/` cannot load. Use the one versioned here:

```bash
mkdir -p RT-DETRv4/configs/usvdetr
cp USV-DETR/configs/USV-DETR.yml RT-DETRv4/configs/usvdetr/
```

## Usage

```bash
usvdetr-detect /data/recordings --recursive \
    --config RT-DETRv4/configs/usvdetr/USV-DETR.yml \
    --checkpoint USV-DETR.pth \
    --output detections.xlsx \
    --plot-dir figures
```

The input may be a single wav file or a folder. From a clone without
installing, `python detect.py` takes the same arguments. Run
`usvdetr-detect --help` for every option, grouped by pipeline stage.

```python
from usvdetr import load_model, detect_wav, save_records

model, device = load_model(
    "RT-DETRv4/configs/usvdetr/USV-DETR.yml", "USV-DETR.pth"
)
df = detect_wav(model, device, "recording.wav")
save_records(df, "detections.xlsx")
```

## Output

An XLSX or CSV table with one row per detected call: source filename, audio
duration, the analysis window in which the call was detected, onset, offset and
duration, frequency range, mean frequency and bandwidth, a USV or Harmonic label
with an ID linking each harmonic to its fundamental, and the detector confidence
score.
