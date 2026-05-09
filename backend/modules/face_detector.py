"""
face_detector.py — Face detection, alignment, and quality filtering.

Uses InsightFace's FaceAnalysis API which handles detection, landmark
extraction, alignment, and embedding generation as one atomic pipeline
stage via app.get(). This module wraps that call and applies post-detection
quality filters (minimum face size, minimum confidence).

Usage:
    detector = FaceDetector()
    faces = detector.detect_faces(frame)
    # Each face dict contains: bbox, confidence, embedding, landmarks
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Wraps InsightFace FaceAnalysis for face detection + quality filtering.

    InsightFace internally handles:
      - Face detection (SCRFD)
      - Landmark extraction (5-point)
      - Face alignment (affine transform based on landmarks)
      - Embedding generation (ArcFace, 512-d)

    This class adds a quality gate on top: faces that are too small or
    have low detection confidence are filtered out.
    """

    def __init__(self, det_size=(640, 640), min_face_size=40, min_confidence=0.6):
        """
        Initialize the face detector.

        Args:
            det_size: Input size for the detection model (width, height).
            min_face_size: Minimum face bbox dimension in pixels.
            min_confidence: Minimum detection confidence score (0–1).
        """
        self.det_size = det_size
        self.min_face_size = min_face_size
        self.min_confidence = min_confidence
        self._app = None  # Lazy initialization

    def _initialize(self):
        """Lazy-load InsightFace models (downloads on first run)."""
        from insightface.app import FaceAnalysis

        logger.info("Initializing InsightFace FaceAnalysis (buffalo_l)...")
        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._app.prepare(ctx_id=0, det_size=self.det_size)
        logger.info("InsightFace initialized successfully.")

    def detect_faces(self, frame):
        """
        Detect faces in a BGR frame and return quality-filtered results.

        InsightFace's app.get() performs detection → alignment → embedding
        as one pipeline stage. We then filter by size and confidence.

        Args:
            frame: BGR image (numpy array, HWC format).

        Returns:
            List of dicts, each containing:
              - 'bbox': [x1, y1, x2, y2] as integers
              - 'confidence': float detection score
              - 'embedding': 512-d numpy array (raw from ArcFace)
              - 'landmarks': 5-point facial landmarks
              - 'face_obj': original InsightFace face object
        """
        if self._app is None:
            self._initialize()

        # InsightFace pipeline: detect → align → embed (all in one call)
        faces = self._app.get(frame)

        if not faces:
            return []

        quality_faces = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            confidence = float(face.det_score)

            # Quality gate: minimum size check
            if width < self.min_face_size or height < self.min_face_size:
                logger.debug(
                    "Face rejected (too small): %dx%d < %d",
                    width, height, self.min_face_size
                )
                continue

            # Quality gate: minimum confidence check
            if confidence < self.min_confidence:
                logger.debug(
                    "Face rejected (low confidence): %.3f < %.3f",
                    confidence, self.min_confidence
                )
                continue

            quality_faces.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": confidence,
                "embedding": face.embedding,   # 512-d vector from ArcFace
                "landmarks": face.kps if hasattr(face, 'kps') else None,
                "face_obj": face,
            })

        return quality_faces
