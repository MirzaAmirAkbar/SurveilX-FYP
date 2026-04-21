"""
sqlite_logger.py — SQLite metadata and log storage.

Manages two tables:
  1. enrollments  — Records of faces enrolled during video_1 processing.
  2. recognition_logs — Records of face match attempts during video_2 processing.

This module handles only structured metadata; the actual face embeddings
are stored in ChromaDB (see chroma_store.py).

Usage:
    db = SQLiteLogger(db_path="storage/metadata.db")
    db.log_enrollment(person_temp_id="Person_01", ...)
    db.log_recognition(frame_number=100, ...)
    db.close()
"""

import sqlite3
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SQLiteLogger:
    """
    SQLite database for enrollment metadata and recognition event logs.
    """

    def __init__(self, db_path):
        """
        Initialize SQLite connection and create tables if needed.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        logger.info("Opening SQLite database: %s", db_path)

        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        self._create_tables()

    def _create_tables(self):
        """Create the enrollments and recognition_logs tables if they don't exist."""
        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enrollments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                person_temp_id  TEXT NOT NULL,
                embedding_id    TEXT NOT NULL UNIQUE,
                frame_number    INTEGER NOT NULL,
                timestamp_sec   REAL NOT NULL,
                video_name      TEXT NOT NULL,
                bbox_x1         INTEGER,
                bbox_y1         INTEGER,
                bbox_x2         INTEGER,
                bbox_y2         INTEGER,
                detection_confidence REAL,
                created_at      TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_number    INTEGER NOT NULL,
                timestamp_sec   REAL NOT NULL,
                video_name      TEXT NOT NULL,
                matched_person_id TEXT,
                similarity_score REAL,
                match_result    TEXT NOT NULL,
                bbox_x1         INTEGER,
                bbox_y1         INTEGER,
                bbox_x2         INTEGER,
                bbox_y2         INTEGER,
                created_at      TEXT NOT NULL
            )
        """)

        self._conn.commit()
        logger.info("SQLite tables verified/created.")

    def log_enrollment(self, person_temp_id, embedding_id, frame_number,
                       timestamp_sec, video_name, bbox, detection_confidence):
        """
        Log a face enrollment event.

        Args:
            person_temp_id: Assigned temporary ID (e.g., "Person_01").
            embedding_id: Unique embedding ID stored in ChromaDB.
            frame_number: Frame index in the source video.
            timestamp_sec: Frame timestamp in seconds.
            video_name: Source video filename.
            bbox: [x1, y1, x2, y2] bounding box coordinates.
            detection_confidence: Face detection confidence score.
        """
        now = datetime.now(timezone.utc).isoformat()
        x1, y1, x2, y2 = bbox

        self._conn.execute(
            """
            INSERT INTO enrollments
                (person_temp_id, embedding_id, frame_number, timestamp_sec,
                 video_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                 detection_confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (person_temp_id, embedding_id, frame_number, timestamp_sec,
             video_name, x1, y1, x2, y2, detection_confidence, now)
        )
        self._conn.commit()

        logger.debug(
            "Enrolled %s (frame %d, confidence %.3f)",
            person_temp_id, frame_number, detection_confidence
        )

    def log_recognition(self, frame_number, timestamp_sec, video_name,
                        matched_person_id, similarity_score, match_result, bbox):
        """
        Log a face recognition event.

        Args:
            frame_number: Frame index in the source video.
            timestamp_sec: Frame timestamp in seconds.
            video_name: Source video filename.
            matched_person_id: Matched person ID or "NEW" if no match.
            similarity_score: Cosine similarity (1 - distance).
            match_result: "MATCH" or "NO_MATCH".
            bbox: [x1, y1, x2, y2] bounding box coordinates.
        """
        now = datetime.now(timezone.utc).isoformat()
        x1, y1, x2, y2 = bbox

        self._conn.execute(
            """
            INSERT INTO recognition_logs
                (frame_number, timestamp_sec, video_name, matched_person_id,
                 similarity_score, match_result, bbox_x1, bbox_y1,
                 bbox_x2, bbox_y2, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (frame_number, timestamp_sec, video_name, matched_person_id,
             similarity_score, match_result, x1, y1, x2, y2, now)
        )
        self._conn.commit()

    def get_enrollment_count(self):
        """Return the number of unique persons enrolled."""
        cursor = self._conn.execute(
            "SELECT COUNT(DISTINCT person_temp_id) FROM enrollments"
        )
        return cursor.fetchone()[0]

    def get_recognition_summary(self):
        """Return a summary of recognition results."""
        cursor = self._conn.execute("""
            SELECT
                match_result,
                COUNT(*) as count,
                AVG(similarity_score) as avg_similarity
            FROM recognition_logs
            GROUP BY match_result
        """)
        return cursor.fetchall()

    def clear_all(self):
        """Delete all records from both tables (for fresh runs)."""
        self._conn.execute("DELETE FROM enrollments")
        self._conn.execute("DELETE FROM recognition_logs")
        self._conn.commit()
        logger.info("SQLite tables cleared.")

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            logger.info("SQLite connection closed.")
