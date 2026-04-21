# Facial Recognition Prototype — Implementation Plan

## Goal

Build a complete, standalone Python facial recognition prototype operating in two phases:

1. **Enrollment Phase (video_1.mp4)** — Detect faces, generate embeddings, store in ChromaDB as the reference database, log metadata in SQLite.
2. **Recognition Phase (video_2.mp4)** — Detect faces, generate embeddings, search ChromaDB for matches, annotate the output video with bounding boxes and MATCH FOUND / NO MATCH labels.

## User Review Required

> [!IMPORTANT]
> **Video files**: Place `video_1.mp4` and `video_2.mp4` in `Facial_Recognition/data/videos/` before running. These are not included in the repo.

> [!IMPORTANT]
> **GPU vs CPU**: The plan uses InsightFace with `onnxruntime`. If you have an NVIDIA GPU, install `onnxruntime-gpu` instead of `onnxruntime` for ~10x faster processing. The code defaults to CPU fallback.

> [!WARNING]
> **InsightFace model licensing**: The pre-trained ArcFace models bundled with InsightFace are licensed for **non-commercial research only**. This is fine for an FYP prototype but should be noted.

---

## Proposed Folder Structure

```
Facial_Recognition/
├── config.py                  # All configurable thresholds and paths
├── main.py                    # Entry point — orchestrates both phases
├── requirements.txt           # Python dependencies
├── README.md                  # Full project documentation
│
├── modules/
│   ├── __init__.py
│   ├── face_detector.py       # Face detection + quality check + alignment
│   ├── embedding_engine.py    # Embedding generation (InsightFace ArcFace)
│   ├── chroma_store.py        # ChromaDB persistent storage interface
│   ├── sqlite_logger.py       # SQLite metadata/log storage
│   ├── video_processor.py     # OpenCV video reading/writing + annotation
│   └── id_manager.py          # Temporary ID assignment logic
│
├── data/
│   ├── videos/                # Input videos (video_1.mp4, video_2.mp4)
│   └── output/                # Output annotated videos
│
└── storage/
    ├── chroma_db/             # ChromaDB persistent storage directory
    └── metadata.db            # SQLite database file
```

---

## Proposed Changes

### Configuration

#### [NEW] [config.py](file:///c:/Users/Lenovo/Documents/fyp-2/Facial_Recognition/config.py)

Centralized configuration file with all tunable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VIDEO_1_PATH` | `data/videos/video_1.mp4` | Enrollment video |
| `VIDEO_2_PATH` | `data/videos/video_2.mp4` | Recognition video |
| `OUTPUT_VIDEO_PATH` | `data/output/recognized_output.mp4` | Annotated output |
| `CHROMA_PERSIST_DIR` | `storage/chroma_db` | ChromaDB storage |
| `SQLITE_DB_PATH` | `storage/metadata.db` | SQLite database |
| `FACE_QUALITY_MIN_SIZE` | `40` | Minimum face size in pixels (width & height) |
| `FACE_QUALITY_MIN_CONFIDENCE` | `0.6` | Minimum detection confidence |
| `SIMILARITY_THRESHOLD` | `0.55` | Cosine similarity threshold for match (ChromaDB uses cosine distance, so threshold = 1 - similarity) |
| `FRAME_SKIP_ENROLLMENT` | `5` | Process every Nth frame during enrollment (speed) |
| `FRAME_SKIP_RECOGNITION` | `1` | Process every Nth frame during recognition |
| `DET_SIZE` | `(640, 640)` | InsightFace detection input size |

---

### Core Modules

#### [NEW] [face_detector.py](file:///c:/Users/Lenovo/Documents/fyp-2/Facial_Recognition/modules/face_detector.py)

Wraps InsightFace's `FaceAnalysis` API. Responsibilities:
- Initialize the `buffalo_l` model pack (SCRFD detector + ArcFace recognizer)
- Detect faces in a frame → returns list of face objects
- **Quality check**: Filter faces by minimum size (pixels) and detection confidence
- Faces that pass quality check are already aligned by InsightFace internally (landmark-based alignment)
- Returns bounding boxes, landmarks, and detection scores for each valid face

**Key decisions:**
- InsightFace handles detection + alignment + embedding in one unified `app.get()` call. We still modularize the code, but leverage this convenience internally.
- Quality filtering happens post-detection: we discard faces that are too small or low-confidence.

#### [NEW] [embedding_engine.py](file:///c:/Users/Lenovo/Documents/fyp-2/Facial_Recognition/modules/embedding_engine.py)

Extracts the 512-dimensional ArcFace embedding from detected face objects:
- Input: face object from InsightFace (already contains `.embedding`)
- Output: normalized 512-d float vector
- Normalization: L2-normalize embeddings before storage (required for cosine similarity)

#### [NEW] [chroma_store.py](file:///c:/Users/Lenovo/Documents/fyp-2/Facial_Recognition/modules/chroma_store.py)

ChromaDB interface for embedding storage and search:
- Uses `PersistentClient` with path `storage/chroma_db/`
- Collection: `face_embeddings` with `metadata={"hnsw:space": "cosine"}`
- **Enrollment API**: `add_embedding(embedding_id, embedding_vector, metadata_dict)` — stores a reference face
- **Search API**: `search_embedding(query_embedding, n_results=1)` → returns best match ID + cosine distance
- **Match decision**: If `cosine_distance < (1 - SIMILARITY_THRESHOLD)`, it's a match. ChromaDB returns distance (0 = identical), so lower is better.
- Includes `reset()` method to clear the collection for fresh runs
- Includes `get_count()` to check how many embeddings are stored

#### [NEW] [sqlite_logger.py](file:///c:/Users/Lenovo/Documents/fyp-2/Facial_Recognition/modules/sqlite_logger.py)

SQLite metadata/log storage with two tables:

**Table: `enrollments`**
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `person_temp_id` | TEXT | e.g., "Person_01" |
| `embedding_id` | TEXT | Unique ID stored in ChromaDB |
| `frame_number` | INTEGER | Frame index |
| `timestamp_sec` | REAL | Frame timestamp in seconds |
| `video_name` | TEXT | Source video filename |
| `bbox_x1, bbox_y1, bbox_x2, bbox_y2` | INTEGER | Face bounding box |
| `detection_confidence` | REAL | Face detection confidence |
| `created_at` | TEXT | ISO timestamp |

**Table: `recognition_logs`**
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `frame_number` | INTEGER | Frame index |
| `timestamp_sec` | REAL | Frame timestamp in seconds |
| `video_name` | TEXT | Source video filename |
| `matched_person_id` | TEXT | Matched person ID or "NEW" |
| `similarity_score` | REAL | Cosine similarity (1 - distance) |
| `match_result` | TEXT | "MATCH" or "NO_MATCH" |
| `bbox_x1, bbox_y1, bbox_x2, bbox_y2` | INTEGER | Face bounding box |
| `created_at` | TEXT | ISO timestamp |

#### [NEW] [video_processor.py](file:///c:/Users/Lenovo/Documents/fyp-2/Facial_Recognition/modules/video_processor.py)

OpenCV video I/O and annotation:
- `VideoReader`: Opens video, provides frame-by-frame iteration with frame index and timestamp
- `VideoWriter`: Creates output video with same codec/fps/resolution
- `annotate_frame()`: Draws bounding boxes + text labels on frames
  - **MATCH FOUND**: Green bounding box + "MATCH: Person_XX (sim: 0.85)"
  - **NO MATCH**: Red bounding box + "NO MATCH (sim: 0.32)"
- Adds an info overlay (frame number, enrollment count, etc.)

#### [NEW] [id_manager.py](file:///c:/Users/Lenovo/Documents/fyp-2/Facial_Recognition/modules/id_manager.py)

Temporary ID assignment during enrollment:
- Maintains an internal counter: `Person_01`, `Person_02`, ...
- Deduplication: Before assigning a new ID, checks if the current face embedding is similar to any already-enrolled face (using ChromaDB search). If similar → reuse existing ID, don't create duplicate.
- This prevents the same person appearing in multiple frames from getting multiple IDs.
- Returns the assigned `person_temp_id` and `embedding_id`

---

### Entry Point

#### [NEW] [main.py](file:///c:/Users/Lenovo/Documents/fyp-2/Facial_Recognition/main.py)

Orchestrates the full pipeline:

```
Phase 1: Enrollment (video_1.mp4)
  1. Initialize all modules
  2. Reset ChromaDB collection (fresh start)
  3. Read video_1 frame-by-frame (respecting FRAME_SKIP_ENROLLMENT)
  4. For each frame:
     a. Detect faces → quality filter
     b. For each quality face:
        - Generate embedding
        - Check ChromaDB for duplicates (same person across frames)
        - If new person → assign temp ID, store in ChromaDB, log in SQLite
        - If duplicate → skip (or update metadata)
  5. Print enrollment summary (total unique persons enrolled)

Phase 2: Recognition (video_2.mp4)  ← starts only after Phase 1 completes
  1. Open video_2 and output video writer
  2. Read frame-by-frame (respecting FRAME_SKIP_RECOGNITION)
  3. For each frame:
     a. Detect faces → quality filter
     b. For each quality face:
        - Generate embedding
        - Search ChromaDB for nearest match
        - Compute similarity = 1 - cosine_distance
        - If similarity >= SIMILARITY_THRESHOLD → MATCH FOUND
        - Else → NO MATCH
        - Log result to SQLite
     c. Annotate frame with bounding boxes + labels
     d. Write annotated frame to output video
  4. Release resources
  5. Print recognition summary
```

---

### Documentation

#### [NEW] [requirements.txt](file:///c:/Users/Lenovo/Documents/fyp-2/Facial_Recognition/requirements.txt)

```
opencv-python>=4.8.0
insightface>=0.7.3
onnxruntime>=1.16.0
chromadb>=0.4.0
numpy>=1.24.0
```

#### [NEW] [README.md](file:///c:/Users/Lenovo/Documents/fyp-2/Facial_Recognition/README.md)

Comprehensive documentation including:
- Project overview and architecture diagram
- Installation instructions (pip + model download)
- Folder structure explanation
- How to run the project
- Configuration guide (all thresholds)
- Matching decision explanation
- Example output screenshots
- Troubleshooting

---

## How Matching Works (to include in README)

1. **Enrollment**: Each unique face from video_1 gets a 512-dimensional ArcFace embedding stored in ChromaDB.
2. **Recognition**: Each face from video_2 gets an embedding, which is compared against ALL stored embeddings using **cosine similarity** (ChromaDB HNSW index).
3. **Decision**: ChromaDB returns the nearest embedding and its **cosine distance** (0 = identical, 2 = opposite).
   - We compute `similarity = 1 - distance`
   - If `similarity >= 0.55` (configurable) → **MATCH FOUND** → annotate with the matched person's ID
   - If `similarity < 0.55` → **NO MATCH** → annotate as unknown
4. **Deduplication during enrollment**: When processing video_1, before creating a new person, we check if the face is already enrolled (similarity > 0.55). This prevents the same person from getting multiple IDs across different frames.

---

## Open Questions

> [!IMPORTANT]
> **Q1**: Do you have `video_1.mp4` and `video_2.mp4` ready? If not, I can configure the code to work with any video filenames passed as command-line arguments.

> [!IMPORTANT]
> **Q2**: Should I also generate a **face gallery** — saving cropped face images for each enrolled person to `data/faces/Person_XX.jpg`? This is useful for visual verification but not strictly required.

---

## Verification Plan

### Automated Tests
1. Run `main.py` with test videos and verify:
   - ChromaDB collection is populated after Phase 1
   - SQLite tables contain correct enrollment records
   - Output video is created with correct annotations
   - No crashes or unhandled exceptions

2. Verify the output video has:
   - Green boxes for matched faces
   - Red boxes for unmatched faces
   - Correct similarity scores displayed

### Manual Verification
- Review the output video visually to confirm bounding boxes and labels are correct
- Check SQLite database contents using a query
- Verify ChromaDB has the expected number of unique persons
