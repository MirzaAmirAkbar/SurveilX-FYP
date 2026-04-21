"""
embedding_engine.py — Embedding extraction and normalization.

Provides a clean interface for extracting and normalizing face embeddings
from InsightFace face objects. The actual embedding computation is done
by InsightFace internally (ArcFace model); this module handles:
  - L2 normalization (required for cosine similarity)
  - Conversion to Python list (for ChromaDB storage)
  - Dimensionality validation

Usage:
    engine = EmbeddingEngine()
    vector = engine.get_normalized_embedding(face_dict)
    vector_list = engine.to_list(vector)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# ArcFace produces 512-dimensional embeddings
EXPECTED_DIM = 512


class EmbeddingEngine:
    """
    Extracts and normalizes face embeddings from InsightFace face objects.

    InsightFace computes the raw embedding inside app.get().
    This class takes that raw embedding, validates it, and L2-normalizes it
    so that cosine similarity can be computed correctly.
    """

    def get_normalized_embedding(self, face_dict):
        """
        Extract and L2-normalize the embedding from a face dict.

        Args:
            face_dict: Dict from FaceDetector.detect_faces() containing
                       an 'embedding' key with a raw numpy vector.

        Returns:
            L2-normalized 512-d numpy array, or None if invalid.
        """
        raw_embedding = face_dict.get("embedding")

        if raw_embedding is None:
            logger.warning("Face dict has no embedding.")
            return None

        embedding = np.array(raw_embedding, dtype=np.float32)

        # Validate dimensionality
        if embedding.shape[0] != EXPECTED_DIM:
            logger.warning(
                "Unexpected embedding dimension: %d (expected %d)",
                embedding.shape[0], EXPECTED_DIM
            )
            return None

        # L2 normalization — essential for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            logger.warning("Zero-norm embedding detected, returning None.")
            return None

        normalized = embedding / norm
        return normalized

    @staticmethod
    def to_list(embedding):
        """
        Convert a numpy embedding to a Python list (for ChromaDB storage).

        Args:
            embedding: numpy array.

        Returns:
            List of floats.
        """
        if embedding is None:
            return None
        return embedding.tolist()

    @staticmethod
    def compute_similarity(embedding_a, embedding_b):
        """
        Compute cosine similarity between two L2-normalized embeddings.

        Since both vectors are L2-normalized, cosine similarity = dot product.

        Args:
            embedding_a: numpy array (normalized).
            embedding_b: numpy array (normalized).

        Returns:
            Float in [-1, 1], typically [0, 1] for face embeddings.
        """
        return float(np.dot(embedding_a, embedding_b))
