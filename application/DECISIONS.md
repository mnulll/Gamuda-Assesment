# Technical Decisions & Trade-offs

This log records the key technical decisions made while building the
Infrastructure Defect Segmentation System, along with the alternatives
considered and the trade-offs accepted.

---

## Decision 1 — Run inference even when quality checks fail (advisory, not blocking)

**Decision:** Brightness and blur checks produce *warnings* attached to the
response, but the model always runs and always returns a result.

**Alternatives considered:**
- Reject low-quality images with an error and refuse to infer.

**Trade-off:** Blocking would guarantee that every returned result came from a
"good" image, but in real infrastructure inspection the flawed photo is often
the *only* photo — taken in a dark stairwell, at arm's length over a ledge, or
in haste. Refusing it serves no one. By always inferring and surfacing a clear
warning, the user keeps agency: they see the result and decide
for themselves whether to trust it or re-shoot. The cost is that users must read
and heed warnings rather than being protected by a hard gate.

---

## Decision 2 — Variance-of-Laplacian for blur, mean intensity for brightness

**Decision:** Use the variance of the Laplacian as the focus/blur metric and the
grayscale mean as the brightness metric.

**Alternatives considered:**
- A small CNN trained to score image quality.

**Trade-off:** Variance-of-Laplacian and mean intensity are textbook, dependency-free,
and run in milliseconds on a single grayscale pass — no extra model, no GPU, no
training data. They are not perfectly robust (a legitimately smooth surface can
read as "blurry"). A learned quality model would be more accurate but
adds a second model to train, version, and serve.

---

## Decision 3 — Transport the annotated image as base64 inside JSON

**Decision:** The API returns the annotated image base64-encoded inside the JSON
response, alongside the detections and quality report.

**Alternatives considered:**
- Write the image to shared storage and return a URL.

**Trade-off:** Base64-in-JSON inflates the payload by ~33% and is not ideal for
very large images. But it keeps the entire result — image, detections, quality —
in **one atomic response with one schema**, which the frontend consumes in a
single request with no second round-trip and no shared filesystem or object
store to provision. For interactive single-image inspection, the simplicity wins
decisively.

---

## Decision 4 — Load the model once at startup, single worker

**Decision:** Instantiate the YOLOv8 model at module load time (process startup),
and run the API with a single worker.

**Alternatives considered:**
- Run multiple workers, each with its own model copy.

**Trade-off:** Startup loading means the first request is fast (no cold-start
penalty mid-traffic) at the cost of a slightly longer boot before the service is
ready — which the Docker healthcheck already accounts for. A single worker keeps
exactly one model in memory, avoiding the multiplied RAM/VRAM cost of one copy
per worker. To scale, we add **replicas** (more containers) rather than workers
within a container, which keeps each process's memory footprint predictable.


---

## Decision 5 — Streamlit instead of React for the frontend

**Decision:** Used Streamlit instead of React + JavaScript for the frontend.

**Why:** My background is in computer vision, not frontend development. Given the
time constraint I chose to prioritise the parts I know best — the model pipeline,
data quality checks, and the inference API — and used Streamlit to get a working
UI up quickly rather than spending the majority of my time on a React app.

In a real team setting this is exactly the kind of decision I'd make in
collaboration with a more experienced frontend/software engineer. I'd own the
CV and backend side, they'd own the React frontend, and the handoff point would
be the `/predict` API — which is already designed to be frontend-agnostic so
either side can be swapped out independently.

**What I would have done with more time (or with a frontend engineer):**
- Replace `app/streamlit_app.py` with a React app calling the same `/predict` endpoint
- UI improvements: drag-and-drop upload, side-by-side image comparison, per-class confidence bars

---

## Decision 6 — YOLOv8n-seg as the segmentation model

**Decision:** Used the nano variant of YOLOv8 segmentation (`yolov8n-seg`) as the base model.

**Alternatives considered:**
- Larger YOLOv8 variants (s, m, l, x) for better accuracy.

**Trade-off:** The nano model is the smallest and fastest in the YOLOv8 family — it fits comfortably within Render's free tier limit. The cost is lower accuracy compared to the larger variants, especially on small or thin defects like hairline cracks. For a production system with a GPU, I would move up to `yolov8s-seg` or `yolov8m-seg`. For this assessment the priority was a working deployable system, so the nano model was the right call.

---

## Decision 7 — One shared requirements.txt for both services

**Decision:** Used a single `requirements.txt` at the project root covering both the API and the frontend dependencies.

**Alternatives considered:**
- Separate `requirements.txt` per service (`api/requirements.txt` and `app/requirements.txt`).

**Trade-off:** One file is simpler to maintain — one place to update a version, no duplication. The downside is that both Docker images install all packages even though the frontend only needs three of them (`streamlit`, `requests`, `pillow`) and has no use for `ultralytics` or `torch`. This makes the frontend image unnecessarily large . For this assessment that is acceptable. In production I would split them so each service only installs what it actually needs.

## Future Enhancements

Things I would add with more time, grouped by priority.

---

### 1. Export model to TensorRT for faster inference

Right now the model runs in PyTorch on CPU. Exporting to TensorRT would give significant speedups on a GPU-enabled server — typically 3–5x faster inference with lower memory usage. Ultralytics supports this in one line:

The API would then load `best.engine` instead of `best.pt`.

---

### 2. Request validation with Pydantic response schemas

The `/predict` endpoint currently accepts any file upload with no validation — a non-image file, a corrupted upload, or a zero-byte file would cause an unhandled crash. The fix is to add proper input validation and typed response models using Pydantic:


This gives three things: FastAPI auto-validates the response shape before it goes out. It also validates the uploaded file is actually an image (JPEG/PNG) and rejects anything else before it reaches the model.

---


### 3. Batch inference endpoint

The current `/predict` endpoint handles one image per request. For a real inspection workflow — uploading a set of photos from a site visit — a `/predict/batch` endpoint that accepts multiple images and returns results for all of them would save significant round-trip time. Ultralytics supports batched inference natively so the model side is straightforward; the main work is handling partial failures (one bad image in a batch of ten should not fail the whole request).

---

### 5. Confidence threshold as a per-request parameter

Right now `CONF_THRESHOLD` is a fixed environment variable. Different use cases want different thresholds — a safety-critical inspection might want 0.5 to reduce false positives, a routine survey might want 0.15 to catch everything.


