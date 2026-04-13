"""
Shoplifting Detection - Configuration
Centralized settings for paths, model parameters, and detection thresholds.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Best available weight file (75% accuracy, RGB-only Gate_Flow_SlowFast)
WEIGHT_FILE = os.path.join(
    WEIGHTS_DIR, "weights_at_epoch_8_75___good.h5"
)

# ─── Model Parameters ───────────────────────────────────────────────────────
INPUT_FRAMES = 64           # Number of frames the model expects
INPUT_HEIGHT = 224          # Frame height
INPUT_WIDTH = 224           # Frame width
INPUT_CHANNELS = 3          # RGB channels
SLOW_PATH_STRIDE = 16      # Tau: stride for slow pathway (64/16 = 4 frames)

# ─── Detection Thresholds ───────────────────────────────────────────────────
# Classes: [0] Bag concealment, [1] Clothes concealment, [2] Normal
CLASS_NAMES = ["Bag", "Clothes", "Normal"]
# A detection is flagged as theft if Normal < both Bag and Clothes
THEFT_SCORE_THRESHOLD = 0.5  # Minimum theft class probability to flag

# ─── Video Processing ───────────────────────────────────────────────────────
FRAME_WINDOW_SIZE = 149     # Number of frames to collect before processing
SLIDING_WINDOW_STEP = 32   # Overlap step for splitting into 64-frame chunks
OUTPUT_FPS = 15             # FPS for saved output videos
OUTPUT_CODEC = "MJPG"       # Video codec for output (MJPG for .avi)

# ─── Email Alerts (disabled by default) ──────────────────────────────────────
EMAIL_ALERTS_ENABLED = False
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = ""
SENDER_PASSWORD = ""
ALERT_RECIPIENTS = []

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
