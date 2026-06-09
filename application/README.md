# Infrastructure Defect Segmentation System

Pixel-level segmentation of infrastructure defects — **cracks, spalling,
corrosion, and structural deformation** — from construction-site images.

Upload an image in the browser → the system analyzes its quality, runs a
YOLOv8 segmentation model, and returns an annotated image with per-defect class
labels and confidence scores.

```
Browser (Streamlit, :8501)  ──►  FastAPI + YOLOv8 (:8000)  ──►  annotated result
```

- **Frontend:** Streamlit — image upload + result presentation.
- **Backend:** FastAPI + YOLOv8 segmentation — quality analysis + inference.
- **Quality pre-check:** brightness + blur analysis. If an image fails, the
  system **still runs inference** but attaches a clear warning.
- **Deployment:** two Docker containers via docker-compose.

---

## Project structure

```
defect-seg/
├── api/
│   └── main.py                  # FastAPI inference service
├── app/
│   └── streamlit_app.py         # Streamlit frontend
├── weights/
│   └── best.pt                  # YOLOv8 segmentation model (you provide this)
├── requirements.txt
├── Dockerfile.api
├── Dockerfile.app
├── docker-compose.yml
├── ARCHITECTURE.md
├── DECISIONS.md
└── README.md
```

---

## Prerequisites

- **Docker** and **Docker Compose** (for the recommended path), or
- **Python 3.11** (for local non-Docker runs).
- A trained **YOLOv8 segmentation** model at `weights/best.pt`.

---

## Quick start (Docker — recommended)

1. **Add a model.** Put your fine-tuned YOLOv8 segmentation weights at
   `weights/best.pt`.

   ```bash
   pip install ultralytics
   ```

2. **Build and run both services:**

   ```bash
   docker compose up --build
   ```

3. **Open the app:** http://localhost:8501

The frontend waits for the API healthcheck to pass, then becomes available.
Only the frontend port (8501) is exposed; the inference API stays on the
internal Docker network.

---

## Local development (without Docker)

Run the two services in two terminals.

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Terminal 1 — API:**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — frontend:**

```bash
export API_URL=http://localhost:8000
streamlit run app/streamlit_app.py
```

Open http://localhost:8501.

---

## Using the app

1. Upload a JPEG or PNG of the structure or site.
2. Click **RUN INFERENCE**.
3. Review the output:
   - **Quality report** — pass, or a warning (dark / overexposed / blurry). A
     warning does **not** stop inference; it informs you the result may be less
     reliable.
   - **Segmented output** — the annotated image with defect masks and labels.
   - **Detections** — class counts and a per-instance confidence table.
4. Download the annotated image if needed.

---

## Configuration

All settings are environment variables (set them in `docker-compose.yml`, your
shell, or your deployment platform):

| Variable          | Purpose                                  | Default          |
|-------------------|------------------------------------------|------------------|
| `MODEL_PATH`      | Path to the YOLOv8 weights               | `weights/best.pt`|
| `CONF_THRESHOLD`  | Minimum detection confidence             | `0.25`           |
| `BRIGHTNESS_MIN`  | Below this, warn "too dark"              | `40`             |
| `BRIGHTNESS_MAX`  | Above this, warn "overexposed"           | `220`            |
| `BLUR_MIN`        | Below this focus score, warn "blurry"    | `100`            |
| `API_URL`         | Backend URL the frontend calls           | `http://api:8000`|

Example — stricter blur gating and a custom model:

```bash
docker compose run -e BLUR_MIN=150 -e MODEL_PATH=weights/site_b.pt app
```

---

## API reference

The backend is a standard FastAPI service with interactive docs at
`http://localhost:8000/docs`.

### `GET /health`
Liveness probe. Returns status and the model's class names.

### `POST /predict`
Multipart form upload with a single `file` field (JPEG/PNG). Returns:

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

Call it directly with curl:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/image.jpg"
```

---

## Deploying to a public URL (free)

I deployed this using **Render** for the API and **Streamlit Community Cloud** for the frontend — both are free, no credit card needed.

The code lives at `https://github.com/mnulll/Gamuda-Assesment` with everything inside the `application/` folder.

---

### 1. Deploy the API on Render

1. Go to [render.com](https://render.com) → New → Web Service
2. Connect GitHub and pick the `Gamuda-Assesment` repo
3. Set these three fields:

   | Field | Value |
   |---|---|
   | **Root Directory** | `application` |
   | **Runtime** | Docker |
   | **Dockerfile Path** | `./Dockerfile.api` |

   Setting Root Directory to `application` tells Render to treat that folder as
   the project root, so all the `COPY` paths in the Dockerfile resolve correctly.

4. Instance Type: **Free**
5. Add these environment variables:

   | Variable | Value |
   |---|---|
   | `MODEL_PATH` | `weights/best.pt` |
   | `CONF_THRESHOLD` | `0.25` |
   | `BRIGHTNESS_MIN` | `40` |
   | `BRIGHTNESS_MAX` | `220` |
   | `BLUR_MIN` | `100` |

6. Hit **Create Web Service** and wait for the build (~5–8 min). Once it's live, test it:

   ```bash
   curl https://YOUR-SERVICE.onrender.com/health
   # {"status":"ok"}
   ```

   Copy the URL — you'll need it for the next step.

---

### 2. Deploy the frontend on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

2. Click **New app** and fill in:

   | Field | Value |
   |---|---|
   | Repository | `mnulll/Gamuda-Assesment` |
   | Branch | `master` |
   | Main file path | `application/app/streamlit_app.py` |

3. Click **Advanced settings** and add the secret:

   ```toml
   API_URL = "https://YOUR-SERVICE.onrender.com"
   ```

4. Hit **Deploy** — done in ~2 min. Your app will be live at
   `https://mnulll-gamuda-assesment-....streamlit.app`

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — system design and data flow.
- **[DECISIONS.md](DECISIONS.md)** — key technical decisions and trade-offs.
