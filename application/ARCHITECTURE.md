# Architecture — Infrastructure Defect Segmentation System

## 1. Overview

This system accepts construction-site and infrastructure images and returns
pixel-level segmentation masks identifying defects — **cracks, spalling,
corrosion, and structural deformation**. It is designed around a single guiding
principle: *keep the moving parts few and the boundaries between them sharp.*

The system has two distinct phases:

**Offline (model production)** — a data pipeline and training notebook that
convert raw annotated images into a trained `best.pt` checkpoint. This runs
once (or whenever the dataset grows) in Google Colab and produces the weight
file the runtime services consume.

**Online (serving)** — two runtime services that accept images from a browser
and return segmentation results:

1. **Frontend (Streamlit)** — handles image upload and result presentation.
2. **Inference API (FastAPI + YOLOv8)** — handles image quality analysis and
   segmentation.

A user uploads an image in the browser; Streamlit forwards the raw bytes to the
API; the API analyzes quality, runs the YOLOv8 segmentation model, and returns
an annotated image plus structured detections; Streamlit renders the result.

```
── OFFLINE (Colab) ────────────────────────────────────────────────────────────

  raw/images/ + raw/labels/ 
        │
        ▼
  data_pipeline.ipynb
  ┌─────────────────────────────────────────────────────┐
  │  1. Quality filter  (brightness / blur / contrast)  │
  │  2. Data → YOLO polygon format                   │
  │  3. Stratified split  70 / 15 / 15                  │
  │  4. Write dataset_yolo/ + data.yaml                 │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  train_yolov8_seg.ipynb
  ┌─────────────────────────────────────────────────────┐
  │  Fine-tune YOLOv8n-seg (frozen backbone)            │
  │  Evaluate on held-out test split                    │
  │  Save  best.pt  →  Google Drive                     │
  └─────────────────────────────────────────────────────┘
        │
        ▼  weights/best.pt

── ONLINE (Docker) ────────────────────────────────────────────────────────────

  ┌──────────────┐    HTTP multipart     ┌────────────────────────────┐
  │   Browser    │  ──── image bytes ──► │      FastAPI service        │
  │              │                       │                             │
  │  Streamlit   │                       │  1. Quality analysis        │
  │  frontend    │                       │     (brightness + blur)     │
  │  (port 8501) │                       │  2. YOLOv8-seg inference    │
  │              │  ◄── JSON response ── │  3. Annotate + serialize    │
  └──────────────┘   annotated image,    │         (port 8000)         │
                     detections,         └────────────────────────────┘
                     quality report
```

## 2. Components

### 2.1 Frontend (`app/streamlit_app.py`)

A thin client. It contains **no model logic** — its only responsibilities are:

- Present a file uploader restricted to JPEG/PNG.
- POST the uploaded bytes to the API's `/predict` endpoint.
- Render the three parts of the response: the quality report (pass/warn), the
  annotated segmentation image, and a detection summary (class counts +
  per-instance confidence table).
- Offer the annotated image as a download.

Because the frontend holds no state and no model, it starts instantly and can be
restarted, scaled, or replaced without touching inference.

### 2.2 Inference API (`api/main.py`)

A FastAPI application exposing two endpoints:

- `GET /health` — liveness probe returning status and the model's class names.
  Used by Docker's healthcheck and any load balancer.
- `POST /predict` — the core endpoint. It performs three sequential steps:

  **Step 1 — Quality analysis (advisory, never blocking).**
  The image is converted to grayscale and two cheap metrics are computed:
  - *Brightness* = mean pixel intensity (0–255). Flags images that are too dark
    or overexposed.
  - *Blur / focus* = variance of the Laplacian. Low variance means few sharp
    edges, i.e. a blurry image where hairline cracks may be missed.

  If either metric falls outside its threshold, a human-readable warning is
  attached to the response — **but inference still runs regardless.** This was a
  deliberate design choice (see DECISIONS.md): a warning informs the user
  without denying them a result.

  **Step 2 — Segmentation inference.**
  The YOLOv8 segmentation model runs on the image with a configurable confidence
  threshold. Ultralytics renders masks, boxes, and labels onto the image.

  **Step 3 — Serialization.**
  The annotated image is encoded as a base64 PNG (so it travels cleanly inside
  JSON), and each detected instance is collected as `{class_name, confidence}`.

The model is loaded **once at process startup**, so the cold-start cost is paid a
single time rather than per request.

### 2.3 Data Pipeline (`data_pipeline.ipynb`)

The pipeline transforms raw annotated images into a clean, YOLO-formatted
dataset. Its output — `dataset_yolo/` and `data.yaml` — is the only artefact
the training notebook consumes.

**Inputs**

| Path | Contents |
|---|---|
| `raw/images/*.jpg` | Source photographs |
| `raw/labels/*.json` | LabelMe polygon annotations (one JSON per image) |

**Stage 1 — Image quality filter**

Every image is measured against three metrics before it enters the dataset.
Images that fail are excluded from training but are always written to
`quality_report.csv` so that nothing is dropped silently.

| Metric | How computed | Default threshold |
|---|---|---|
| Brightness | Mean of the HSV V-channel | 40 – 220 |
| Blur / focus | Variance of the Laplacian (grayscale) | ≥ 80 |
| Contrast | Standard deviation of grayscale pixels | ≥ 15 |

The thresholds mirror those used by the runtime quality check in the inference
API, so an image rejected here would have produced a warning at inference time
anyway.

**Stage 2 — Annotation conversion**

Data stores polygons as absolute pixel coordinates. The pipeline normalises
each point to the 0–1 range expected by YOLO segmentation:

```
x_norm = x_pixel / image_width
y_norm = y_pixel / image_height
```

Each output label file is one line per instance: `<class_id> x1 y1 x2 y2 …`

Class names are discovered automatically by scanning every JSON file, so no
class list needs to be maintained by hand. The resulting `class_map` is
serialised into `data.yaml`.

**Stage 3 — Stratified split**

Records are split 70 / 15 / 15 (train / val / test) using stratified sampling
on the dominant class per image, preserving class proportions across all three
splits. The random seed is fixed at 42 for reproducibility.

**Outputs**

```
dataset_yolo/
├── images/
│   ├── train/   (70 %)
│   ├── val/     (15 %)
│   └── test/    (15 %)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
quality_report.csv
```

### 2.4 Model Training (`train_yolov8_seg.ipynb`)

Fine-tunes a pretrained **YOLOv8n-seg** checkpoint on the prepared dataset,
then evaluates the result on the held-out test split and saves `best.pt` to
Google Drive. The notebook runs in Colab on a T4 GPU.

**Key training decisions**

| Setting | Value | Rationale |
|---|---|---|
| Base model | `yolov8n-seg.pt` | Smallest seg variant; fast to fine-tune on limited data |
| Frozen layers | First 10 (backbone) | Preserves pretrained edge/texture features; reduces overfitting on small datasets |
| Epochs | 100 (+ early stopping, patience 20) | Trains until validation plateaus |
| Augmentation | HSV jitter (H 0.015, S 0.7, V 0.4) | Defects vary in colour/lighting more than in geometry |
| Image size | 640 × 640 | YOLO standard; matches inference resolution |
| Batch | 8 | Fits within T4 VRAM with seg head |



**Evaluation (post-training, fresh runtime)**

After training the Colab runtime is assumed to have disconnected. The
evaluation notebook section re-mounts Drive, reloads `best.pt`, and runs a
full evaluation pass:

- **Aggregate metrics** — mask mAP50 and mAP50-95 on the test split.
- **Per-class metrics table** — Precision, Recall, F1, AP@50, AP@50:95 for
  each defect class
- **Qualitative analysis** — at least 5 annotated examples covering good
  predictions, failure cases (missed hairline cracks, fragmented masks), and
  edge cases (overlapping instances, low-light images).
- **Confusion matrix and PR curve**


**Metric selection rationale**

Instance-level mAP (not pixel IoU) is the primary metric because infrastructure
images typically contain multiple defect instances at different severities.
mAP50-95 is reported alongside mAP50 because it penalises loose mask boundaries
that mAP50 alone would accept — important when defect extent determines repair
scope.

**Model artefact handoff**

The trained checkpoint is saved to `defect_models/best.pt` in Google Drive.
Deploying a new model to the runtime services requires only replacing
`weights/best.pt` — no code change is needed. A helper script

## 3. Data Flow & Contracts

### 3.1 Offline pipeline contract

The data pipeline produces `dataset_yolo/` and `data.yaml`. The training
notebook consumes them by path. The only shared assumption is the YOLO
segmentation label format: `<class_id> x1 y1 x2 y2 …` (normalised 0–1).

### 3.2 Runtime API contract

The contract between frontend and API is a single JSON schema:

```jsonc
{
  "annotated_image": "<base64 PNG>",
  "detections": [
    { "class_name": "crack", "confidence": 0.91 }
  ],
  "quality": {
    "brightness": 129.5,
    "blur_score": 14587.6,
    "passed": true,
    "warnings": []
  }
}
```


### 3.3 Quality threshold alignment

The brightness and blur thresholds are defined in two places: the data pipeline
(`BRIGHTNESS_MIN/MAX`, `BLUR_MIN`, `CONTRAST_MIN`) and the runtime API
(`BRIGHTNESS_MIN/MAX`, `BLUR_MIN` env vars). They should be kept in sync.
Images excluded by the pipeline would have triggered warnings at inference time
anyway; keeping the thresholds consistent prevents training on data the
production system would flag as unreliable.

## 4. Deployment

The system ships as **two Docker images orchestrated by docker-compose**:

- `api` — the FastAPI service, exposed only on the internal compose network.
- `app` — the Streamlit frontend, the single public entry point on port 8501.


`docker compose up` builds and starts both services; the frontend waits for the
API's healthcheck to pass before accepting traffic.

## 5. Configuration

### 5.1 Runtime (environment variables)

All tunable behavior is environment-variable driven, so deployment can adjust it
without code edits:

| Variable          | Purpose                                  | Default          |
|-------------------|------------------------------------------|------------------|
| `MODEL_PATH`      | Path to the YOLOv8 weights               | `weights/best.pt`|
| `CONF_THRESHOLD`  | Minimum detection confidence             | `0.25`           |
| `BRIGHTNESS_MIN`  | Lower brightness bound before warning    | `40`             |
| `BRIGHTNESS_MAX`  | Upper brightness bound before warning    | `220`            |
| `BLUR_MIN`        | Minimum focus score before warning       | `100`            |
| `API_URL`         | Backend URL the frontend calls           | `http://api:8000`|

### 5.2 Data pipeline (top of `data_pipeline.ipynb`)

| Variable          | Purpose                                        | Default              |
|-------------------|------------------------------------------------|----------------------|
| `RAW_IMAGES`      | Source images folder                           | `raw/images`         |
| `RAW_LABELS`      | Source LabelMe JSON folder                     | `raw/labels`         |
| `BASE`            | Output dataset root                            | `dataset_yolo`       |
| `QUALITY_CSV`     | Path for the quality audit log                 | `quality_report.csv` |
| `BRIGHTNESS_MIN`  | Minimum mean brightness (HSV V)                | `40`                 |
| `BRIGHTNESS_MAX`  | Maximum mean brightness                        | `220`                |
| `BLUR_MIN`        | Minimum Laplacian variance                     | `80`                 |
| `CONTRAST_MIN`    | Minimum grayscale std deviation                | `15`                 |
| `EXCLUDE_FLAGGED` | Drop quality-failed images from training split | `True`               |

### 5.3 Training (top of `train_yolov8_seg.ipynb`)

| Variable   | Purpose                                          | Default               |
|------------|--------------------------------------------------|-----------------------|
| `DATA`     | Path to `data.yaml`                              | `data.yaml`           |
| `MODEL`    | Base checkpoint to fine-tune                     | `yolov8n-seg.pt`      |
| `EPOCHS`   | Maximum training epochs                          | `100`                 |
| `IMGSZ`    | Input resolution                                 | `640`                 |
| `BATCH`    | Batch size                                       | `8`                   |
| `FREEZE`   | Number of backbone layers to freeze              | `10`                  |
| `LR0`      | Initial learning rate                            | `0.001`               |
| `PATIENCE` | Early-stopping patience (epochs)                 | `20`                  |
| `HSV_H/S/V`| Colour jitter augmentation                      | `0.015 / 0.7 / 0.4`  |
| `PROJECT`  | Ultralytics output folder name                   | `defect_segmentation` |
| `SEED`     | Random seed for reproducibility                  | `42`                  |

## 6. Why This Design

The architecture is intentionally minimal but cleanly separated:

- **Two services, one contract.** The split between *presentation* (Streamlit)
  and *inference* (FastAPI) is the only structural boundary, and it is the one
  that matters: it lets the UI and the model evolve, scale, and fail
  independently.
- **Configuration over code.** Thresholds and paths are env vars, so the same
  images run in dev and prod with different tuning.
- **Advisory quality gating.** The system informs rather than blocks, which fits
  real inspection workflows where a flawed photo is often the only photo
  available.
- **Offline / online separation.** The data pipeline and training notebooks are
  entirely offline artefacts — they run in Jupyter Notebook and Colab  , produce a checkpoint, and have
  no coupling to the serving stack beyond the weight file path. Retraining never
  requires touching the Docker images.
- **Frozen backbone.** Fine-tuning with the first 10 layers frozen keeps the
  pretrained edge and texture detectors intact and reduces overfitting when the
  defect dataset is small. The freeze count is a single variable, so it is easy
  to update.
- **Consistent quality thresholds.** The brightness and blur bounds are defined
  in both the pipeline and the API. Keeping them aligned means the model is
  never trained on images it would warn about at inference time.
- **Automatic class discovery.** The pipeline scans every annotation file to
  build `class_map`, so adding a new defect class requires only labelling new
  images — no code change anywhere in the stack.

The result is a system a single engineer can understand end-to-end in an
afternoon, yet one that exhibits the right seams to grow.
