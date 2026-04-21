"""
main.py — Entry point for the Facial Recognition Prototype.

Orchestrates two sequential phases:
  Phase 1 (Enrollment): Process video_1.mp4 → detect faces → store embeddings
                         in ChromaDB → log metadata in SQLite.
  Phase 2 (Recognition): Process video_2.mp4 → detect faces → compare against
                          ChromaDB → annotate output video → log results.

video_2 processing does NOT start until video_1 processing is fully complete.

Usage:
    python main.py
    python main.py --video1 path/to/v1.mp4 --video2 path/to/v2.mp4
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# ── Setup logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("main")

# ── Ensure project root is on the path ───────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Import project modules ───────────────────────────────────────────────────
import config
from modules.face_detector import FaceDetector
from modules.embedding_engine import EmbeddingEngine
from modules.chroma_store import ChromaStore
from modules.sqlite_logger import SQLiteLogger
from modules.video_processor import VideoReader, VideoWriter, annotate_frame
from modules.id_manager import IDManager


def parse_args():
    """Parse optional command-line arguments for video paths."""
    parser = argparse.ArgumentParser(description="Facial Recognition Prototype")
    parser.add_argument("--video1", type=str, default=config.VIDEO_1_PATH,
                        help="Path to enrollment video (default: config.VIDEO_1_PATH)")
    parser.add_argument("--video2", type=str, default=config.VIDEO_2_PATH,
                        help="Path to recognition video (default: config.VIDEO_2_PATH)")
    parser.add_argument("--output", type=str, default=config.OUTPUT_VIDEO_PATH,
                        help="Path for annotated output video")
    parser.add_argument("--reset", action="store_true",
                        help="Reset ChromaDB and SQLite before enrollment")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: ENROLLMENT (video_1)
# ══════════════════════════════════════════════════════════════════════════════

def phase_enrollment(video_path, detector, engine, store, db, id_manager):
    """
    Process video_1 for face enrollment.

    For each frame (respecting FRAME_SKIP_ENROLLMENT):
      1. Detect faces and apply quality filter.
      2. If no usable face → skip silently.
      3. For each quality face:
         a. Generate L2-normalized embedding.
         b. Check ChromaDB for duplicate (same person, different frame).
         c. If new person → assign temp ID, store in ChromaDB, log in SQLite.
         d. If duplicate → skip (same person already enrolled).

    Args:
        video_path: Path to the enrollment video.
        detector: FaceDetector instance.
        engine: EmbeddingEngine instance.
        store: ChromaStore instance.
        db: SQLiteLogger instance.
        id_manager: IDManager instance.
    """
    video_name = os.path.basename(video_path)
    logger.info("=" * 60)
    logger.info("PHASE 1: ENROLLMENT — %s", video_name)
    logger.info("=" * 60)

    reader = VideoReader(video_path)
    total_faces_detected = 0
    total_faces_enrolled = 0
    frames_processed = 0

    try:
        for frame, frame_idx, timestamp_sec in reader.frames(skip=config.FRAME_SKIP_ENROLLMENT):
            # Detect faces with quality filtering
            faces = detector.detect_faces(frame)

            # No usable face in this frame → skip silently
            if not faces:
                frames_processed += 1
                continue

            for face_dict in faces:
                total_faces_detected += 1

                # Get normalized embedding
                embedding = engine.get_normalized_embedding(face_dict)
                if embedding is None:
                    continue

                embedding_list = engine.to_list(embedding)

                # Assign ID with deduplication
                person_id, embedding_id, is_new = id_manager.assign_id(embedding_list)

                if is_new:
                    # Store new person's embedding in ChromaDB
                    store.add_embedding(
                        embedding_id=embedding_id,
                        embedding_vector=embedding_list,
                        metadata={"person_temp_id": person_id}
                    )

                    # Log enrollment in SQLite
                    db.log_enrollment(
                        person_temp_id=person_id,
                        embedding_id=embedding_id,
                        frame_number=frame_idx,
                        timestamp_sec=timestamp_sec,
                        video_name=video_name,
                        bbox=face_dict["bbox"],
                        detection_confidence=face_dict["confidence"]
                    )

                    total_faces_enrolled += 1
                    logger.info(
                        "  ✅ Enrolled %s at frame %d (confidence: %.3f)",
                        person_id, frame_idx, face_dict["confidence"]
                    )

            frames_processed += 1

            # Progress report every 100 frames
            if frames_processed % 100 == 0:
                logger.info(
                    "  Progress: %d frames processed, %d faces detected, "
                    "%d persons enrolled",
                    frames_processed, total_faces_detected, total_faces_enrolled
                )

    finally:
        reader.release()

    # ── Enrollment Summary ────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("ENROLLMENT COMPLETE")
    logger.info("  Frames processed:     %d", frames_processed)
    logger.info("  Total faces detected: %d", total_faces_detected)
    logger.info("  Unique persons:       %d", id_manager.total_persons)
    logger.info("  ChromaDB embeddings:  %d", store.get_count())
    logger.info("  SQLite enrollments:   %d", db.get_enrollment_count())
    logger.info("-" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: RECOGNITION (video_2)
# ══════════════════════════════════════════════════════════════════════════════

def phase_recognition(video_path, output_path, detector, engine, store, db,
                      enrollment_count):
    """
    Process video_2 for face recognition against enrolled embeddings.

    For each frame (respecting FRAME_SKIP_RECOGNITION):
      1. Detect faces and apply quality filter.
      2. If no usable face → write unannotated frame, continue.
      3. For each quality face:
         a. Generate L2-normalized embedding.
         b. Search ChromaDB for nearest match.
         c. Compute similarity = 1 - cosine_distance.
         d. If similarity >= threshold → MATCH FOUND.
         e. Else → NO MATCH.
         f. Log result to SQLite.
      4. Annotate frame with bounding boxes + labels.
      5. Write to output video.

    Args:
        video_path: Path to the recognition video.
        output_path: Path for the annotated output video.
        detector: FaceDetector instance.
        engine: EmbeddingEngine instance.
        store: ChromaStore instance.
        db: SQLiteLogger instance.
        enrollment_count: Number of unique persons enrolled.
    """
    video_name = os.path.basename(video_path)
    logger.info("=" * 60)
    logger.info("PHASE 2: RECOGNITION — %s", video_name)
    logger.info("=" * 60)
    logger.info("  Similarity threshold: %.2f", config.SIMILARITY_THRESHOLD)
    logger.info("  Reference embeddings: %d", store.get_count())

    reader = VideoReader(video_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = VideoWriter(output_path, reader.fps, reader.width, reader.height)

    total_matches = 0
    total_no_matches = 0
    frames_processed = 0

    try:
        for frame, frame_idx, timestamp_sec in reader.frames(skip=config.FRAME_SKIP_RECOGNITION):
            # Detect faces with quality filtering
            faces = detector.detect_faces(frame)

            if not faces:
                # No usable face → write unannotated frame and continue
                annotate_frame(frame, [], frame_idx, enrollment_count)
                writer.write(frame)
                frames_processed += 1
                continue

            # Collect detection results for annotation
            detection_results = []

            for face_dict in faces:
                # Get normalized embedding
                embedding = engine.get_normalized_embedding(face_dict)
                if embedding is None:
                    continue

                embedding_list = engine.to_list(embedding)

                # Search ChromaDB for nearest match
                match_emb_id, similarity, metadata = store.search_best(embedding_list)

                # Match decision: similarity = 1 - cosine_distance
                if match_emb_id is not None and similarity >= config.SIMILARITY_THRESHOLD:
                    match_result = "MATCH"
                    matched_person_id = metadata.get("person_temp_id", "Unknown")
                    total_matches += 1
                else:
                    match_result = "NO_MATCH"
                    matched_person_id = "NEW"
                    total_no_matches += 1

                # Log to SQLite
                db.log_recognition(
                    frame_number=frame_idx,
                    timestamp_sec=timestamp_sec,
                    video_name=video_name,
                    matched_person_id=matched_person_id,
                    similarity_score=similarity,
                    match_result=match_result,
                    bbox=face_dict["bbox"]
                )

                # Collect for annotation
                detection_results.append({
                    "bbox": face_dict["bbox"],
                    "match_result": match_result,
                    "matched_person_id": matched_person_id,
                    "similarity": similarity,
                })

                if match_result == "MATCH":
                    logger.debug(
                        "  🟢 MATCH: %s (sim: %.3f) at frame %d",
                        matched_person_id, similarity, frame_idx
                    )
                else:
                    logger.debug(
                        "  🔴 NO MATCH (sim: %.3f) at frame %d",
                        similarity, frame_idx
                    )

            # Annotate frame and write to output
            annotate_frame(frame, detection_results, frame_idx, enrollment_count)
            writer.write(frame)
            frames_processed += 1

            # Progress report every 100 frames
            if frames_processed % 100 == 0:
                logger.info(
                    "  Progress: %d frames, %d matches, %d no-matches",
                    frames_processed, total_matches, total_no_matches
                )

    finally:
        reader.release()
        writer.release()

    # ── Recognition Summary ───────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("RECOGNITION COMPLETE")
    logger.info("  Frames processed: %d", frames_processed)
    logger.info("  Total matches:    %d", total_matches)
    logger.info("  Total no-matches: %d", total_no_matches)
    logger.info("  Output saved to:  %s", output_path)

    # Print SQLite summary
    summary = db.get_recognition_summary()
    if summary:
        logger.info("  SQLite Summary:")
        for result, count, avg_sim in summary:
            logger.info(
                "    %s: %d events (avg similarity: %.3f)",
                result, count, avg_sim
            )
    logger.info("-" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Validate input videos exist
    if not os.path.isfile(args.video1):
        logger.error("Enrollment video not found: %s", args.video1)
        logger.error("Place your enrollment video at: %s", config.VIDEO_1_PATH)
        sys.exit(1)

    if not os.path.isfile(args.video2):
        logger.error("Recognition video not found: %s", args.video2)
        logger.error("Place your recognition video at: %s", config.VIDEO_2_PATH)
        sys.exit(1)

    # ── Initialize modules ───────────────────────────────────────────────
    logger.info("Initializing modules...")

    detector = FaceDetector(
        det_size=config.DET_SIZE,
        min_face_size=config.FACE_QUALITY_MIN_SIZE,
        min_confidence=config.FACE_QUALITY_MIN_CONFIDENCE
    )

    engine = EmbeddingEngine()

    store = ChromaStore(persist_dir=config.CHROMA_PERSIST_DIR)

    db = SQLiteLogger(db_path=config.SQLITE_DB_PATH)

    # Reset for a fresh run if requested
    if args.reset:
        logger.info("Resetting ChromaDB and SQLite for fresh run...")
        store.reset()
        db.clear_all()
    elif store.get_count() > 0:
        logger.info(
            "ChromaDB already has %d embeddings. Use --reset for a fresh run.",
            store.get_count()
        )
        store.reset()
        db.clear_all()
        logger.info("Auto-reset performed for clean enrollment.")

    id_manager = IDManager(
        chroma_store=store,
        dedup_threshold=config.DEDUP_THRESHOLD
    )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Enrollment (video_1) — must complete before Phase 2 starts
    # ══════════════════════════════════════════════════════════════════════
    phase_enrollment(args.video1, detector, engine, store, db, id_manager)

    enrollment_count = id_manager.total_persons

    if enrollment_count == 0:
        logger.warning(
            "⚠️  No faces were enrolled from video_1. "
            "Recognition will have nothing to match against."
        )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Recognition (video_2) — starts only after Phase 1 completes
    # ══════════════════════════════════════════════════════════════════════
    phase_recognition(
        args.video2, args.output,
        detector, engine, store, db, enrollment_count
    )

    # ── Cleanup ──────────────────────────────────────────────────────────
    db.close()
    logger.info("✅ Pipeline complete. Output: %s", args.output)


if __name__ == "__main__":
    main()
