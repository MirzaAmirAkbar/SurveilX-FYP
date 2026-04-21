"""
config.py — Centralized configuration for the Facial Recognition Prototype.

All tunable thresholds, file paths, and model parameters are defined here.
Modify these values to adapt the system to different videos and environments.
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# Base directory (project root)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────────────────────
# Video Paths
# ──────────────────────────────────────────────────────────────────────────────
VIDEO_1_PATH = os.path.join(BASE_DIR, "data", "videos", "video_1.mp4")
VIDEO_2_PATH = os.path.join(BASE_DIR, "data", "videos", "video_2.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "output", "recognized_output.mp4")

# ──────────────────────────────────────────────────────────────────────────────
# Storage Paths
# ──────────────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "storage", "chroma_db")
SQLITE_DB_PATH = os.path.join(BASE_DIR, "storage", "metadata.db")

# ──────────────────────────────────────────────────────────────────────────────
# Face Detection & Quality
# ──────────────────────────────────────────────────────────────────────────────
# Minimum face bounding box size in pixels (width AND height must exceed this)
FACE_QUALITY_MIN_SIZE = 40

# Minimum detection confidence from InsightFace (0.0 to 1.0)
FACE_QUALITY_MIN_CONFIDENCE = 0.6

# InsightFace detection input size — larger = better detection of small faces,
# but slower. (640, 640) is the recommended default.
DET_SIZE = (640, 640)

# ──────────────────────────────────────────────────────────────────────────────
# Similarity Matching
# ──────────────────────────────────────────────────────────────────────────────
# Cosine similarity threshold for deciding MATCH vs NO_MATCH.
# We always compute:  similarity = 1 - cosine_distance
# Then compare:       similarity >= SIMILARITY_THRESHOLD
#
# Typical good values: 0.45–0.60
#   - Higher = stricter matching (fewer false positives, more false negatives)
#   - Lower  = looser matching  (more false positives, fewer false negatives)
SIMILARITY_THRESHOLD = 0.55

# During enrollment, if a new face's similarity to an existing enrolled face
# exceeds this threshold, it is treated as a duplicate (same person).
# Should be >= SIMILARITY_THRESHOLD to avoid conflicts.
DEDUP_THRESHOLD = 0.55

# ──────────────────────────────────────────────────────────────────────────────
# Frame Processing
# ──────────────────────────────────────────────────────────────────────────────
# Process every Nth frame during enrollment (video_1).
# CAUTION: Large values may miss brief or small faces.
# Use 1–3 for short clips, 5–10 for long videos.
FRAME_SKIP_ENROLLMENT = 5

# Process every Nth frame during recognition (video_2).
# Set to 1 for full accuracy; higher values trade accuracy for speed.
FRAME_SKIP_RECOGNITION = 1

# ──────────────────────────────────────────────────────────────────────────────
# Annotation Colors (BGR format for OpenCV)
# ──────────────────────────────────────────────────────────────────────────────
COLOR_MATCH = (0, 200, 0)       # Green — match found
COLOR_NO_MATCH = (0, 0, 220)    # Red — no match
COLOR_TEXT_BG = (0, 0, 0)       # Black — text background
COLOR_TEXT_FG = (255, 255, 255) # White — text foreground
COLOR_INFO_BG = (40, 40, 40)    # Dark gray — info overlay background

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
