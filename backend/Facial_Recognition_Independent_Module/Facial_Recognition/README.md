# Facial Recognition Prototype

A modular Python-based facial recognition system that processes two video files sequentially:

1. **Video 1 (Enrollment)** — Detects faces, generates ArcFace embeddings, stores them in ChromaDB as the reference database, and logs metadata in SQLite.
2. **Video 2 (Recognition)** — Detects faces frame-by-frame, compares each against the reference embeddings, and produces an annotated output video with **MATCH FOUND** / **NO MATCH** labels.

---

## Architecture

```
┌──────────────────────┐
│   video_1.mp4        │     PHASE 1: ENROLLMENT
│   (Entrance Camera)  │
└──────────┬───────────┘
           │
           ▼
  Face Detection (InsightFace SCRFD)
           │
           ▼
  Quality Filter (size + confidence)
           │
           ▼
  Alignment + Embedding (ArcFace 512-d)
           │  ← handled internally by InsightFace
           ▼
  Deduplication Check (ChromaDB search)
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
  New Person    Duplicate
    │             │
    ▼             ▼
  Store in      Skip
  ChromaDB      
    │
    ▼
  Log in SQLite

─────── video_1 processing COMPLETE ───────

┌──────────────────────┐
│   video_2.mp4        │     PHASE 2: RECOGNITION
│   (Aisle Camera)     │
└──────────┬───────────┘
           │
           ▼
  Face Detection (InsightFace SCRFD)
           │
           ▼
  Quality Filter (size + confidence)
           │
           ▼
  Embedding Generation (ArcFace 512-d)
           │
           ▼
  ChromaDB Similarity Search
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
  MATCH         NO MATCH
  (sim ≥ 0.55) (sim < 0.55)
    │             │
    ▼             ▼
  Green Box     Red Box
  + Person ID   + "NO MATCH"
    │             │
    ▼             ▼
  Log in SQLite → Output Video
```

---

## Folder Structure

```
Facial_Recognition/
├── config.py                  # All configurable thresholds and paths
├── main.py                    # Entry point (orchestrates both phases)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── modules/
│   ├── __init__.py
│   ├── face_detector.py       # InsightFace detection + quality filter
│   ├── embedding_engine.py    # Embedding extraction + L2 normalization
│   ├── chroma_store.py        # ChromaDB persistent storage interface
│   ├── sqlite_logger.py       # SQLite metadata + recognition logs
│   ├── video_processor.py     # OpenCV video I/O + frame annotation
│   └── id_manager.py          # Temp ID assignment with deduplication
│
├── data/
│   ├── videos/                # Place video_1.mp4 and video_2.mp4 here
│   └── output/                # Annotated output video saved here
│
└── storage/
    ├── chroma_db/             # ChromaDB persistent storage (auto-created)
    └── metadata.db            # SQLite database (auto-created)
```

---

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Navigate to the project directory
cd Facial_Recognition

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

> **GPU Acceleration (Optional):** If you have an NVIDIA GPU with CUDA:
> ```bash
> pip install onnxruntime-gpu
> ```
> This will dramatically speed up face detection and embedding generation (~10x).

> **Note:** On first run, InsightFace will automatically download the `buffalo_l` model pack (~300MB). This requires an internet connection.

---

## How to Run

### 1. Place your videos

Copy your enrollment and recognition videos into the `data/videos/` folder:

```
data/videos/video_1.mp4   ← Enrollment (reference faces)
data/videos/video_2.mp4   ← Recognition (faces to match)
```

### 2. Run the pipeline

```bash
python main.py
```

### Custom video paths

```bash
python main.py --video1 path/to/enrollment.mp4 --video2 path/to/recognition.mp4 --output path/to/output.mp4
```

### Fresh run (clear previous data)

```bash
python main.py --reset
```

### Output

The annotated video is saved to `data/output/recognized_output.mp4`.

---

## Configuration

All thresholds are in **`config.py`**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FACE_QUALITY_MIN_SIZE` | `40` | Minimum face size (px). Smaller faces are filtered out. |
| `FACE_QUALITY_MIN_CONFIDENCE` | `0.6` | Minimum detection confidence (0–1). |
| `SIMILARITY_THRESHOLD` | `0.55` | Match threshold. Higher = stricter. |
| `DEDUP_THRESHOLD` | `0.55` | Deduplication threshold during enrollment. |
| `FRAME_SKIP_ENROLLMENT` | `5` | Process every Nth frame in video_1. |
| `FRAME_SKIP_RECOGNITION` | `1` | Process every Nth frame in video_2. |
| `DET_SIZE` | `(640, 640)` | Detection input size. Larger = better for small faces. |

---

## How Matching Works

### The Similarity Score

1. Each face is converted to a **512-dimensional embedding** by ArcFace.
2. Embeddings are **L2-normalized** so that cosine similarity = dot product.
3. ChromaDB stores embeddings and uses an **HNSW index with cosine distance**.
4. When searching, ChromaDB returns the nearest embedding and its **cosine distance**.

### The Decision

One formula, used everywhere:

```
similarity = 1 - cosine_distance
```

- `similarity >= 0.55` → **MATCH FOUND** ← same person (green box)
- `similarity < 0.55` → **NO MATCH** ← different person (red box)

### During Enrollment (Deduplication)

The same person may appear in many frames of video_1. Before creating a new person ID, the system searches ChromaDB. If an existing embedding has similarity ≥ 0.55 to the new face, it's treated as the same person and skipped.

### No Face Found

Frames where the detector finds no usable face (too small, too blurry, or no face present) are handled gracefully:
- **During enrollment:** The frame is skipped silently.
- **During recognition:** The frame is written to the output video without annotation.

---

## Output Annotations

The output video includes:

| Element | Description |
|---------|-------------|
| **Green bounding box** | Match found — person recognized from video_1 |
| **Red bounding box** | No match — person not in the reference database |
| **Label text** | `MATCH: Person_01 (sim: 0.85)` or `NO MATCH (sim: 0.32)` |
| **Info overlay** | Frame number, face count, enrolled persons count |

---

## SQLite Database

The SQLite database (`storage/metadata.db`) contains two tables:

### `enrollments` — Records from video_1 processing
- `person_temp_id`, `embedding_id`, `frame_number`, `timestamp_sec`
- `video_name`, bounding box coordinates, `detection_confidence`

### `recognition_logs` — Records from video_2 processing
- `frame_number`, `timestamp_sec`, `video_name`
- `matched_person_id`, `similarity_score`, `match_result` (MATCH/NO_MATCH)
- Bounding box coordinates

Query examples:
```sql
-- View all enrolled persons
SELECT DISTINCT person_temp_id FROM enrollments;

-- View all matches from recognition
SELECT * FROM recognition_logs WHERE match_result = 'MATCH' ORDER BY similarity_score DESC;

-- Count matches vs no-matches
SELECT match_result, COUNT(*), AVG(similarity_score) FROM recognition_logs GROUP BY match_result;
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: insightface` | Run `pip install -r requirements.txt` |
| Model download fails | Ensure internet access. Models download to `~/.insightface/` |
| No faces detected | Lower `FACE_QUALITY_MIN_SIZE` or `FACE_QUALITY_MIN_CONFIDENCE` in `config.py` |
| Too many false matches | Increase `SIMILARITY_THRESHOLD` (e.g., 0.60–0.65) |
| Too many false negatives | Decrease `SIMILARITY_THRESHOLD` (e.g., 0.45–0.50) |
| Slow processing | Install `onnxruntime-gpu`, or increase `FRAME_SKIP_*` values |
| Output video won't play | Try VLC player, or change codec in `video_processor.py` |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Face Detection | InsightFace (SCRFD) | Detect + align faces |
| Face Embedding | InsightFace (ArcFace) | 512-d feature vectors |
| Vector Database | ChromaDB | Store + search embeddings |
| Metadata Storage | SQLite | Logs, metadata, audit trail |
| Video Processing | OpenCV | Read/write/annotate video |
| Runtime | ONNX Runtime | Model inference (CPU/GPU) |
