# Prototyping Notebooks

Jupyter notebooks for the Boggle Vision ML training pipeline. Run from the `prototyping/` directory with the uv-managed `.venv`.

## Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01-bootstrap-and-synthesize.ipynb` | Bootstraps YOLO-seg labels from the legacy OpenCV board detection pipeline, generates synthetic composites (boards pasted onto varied backgrounds with augmentation), and assembles a complete YOLO-seg dataset at `data/yolo-seg-board/`. Includes visual QA grids for reviewing detections and an exclusion list for filtering bad labels. |
| 02 | `02-train-and-evaluate.ipynb` | Fine-tunes YOLOv8n-seg on the board dataset from notebook 01. Trains with reduced augmentation (synthetic data already augmented), validates with mask mAP and per-image IoU analysis, performs qualitative eval on real vs synthetic images, and exports to ONNX with a smoke-test. |
