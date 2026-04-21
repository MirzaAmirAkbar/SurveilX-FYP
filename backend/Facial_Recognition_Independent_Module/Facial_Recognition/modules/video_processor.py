"""
video_processor.py — OpenCV video I/O and frame annotation.

Provides:
  - VideoReader: Iterates frames from a video file with frame index and timestamp.
  - VideoWriter: Writes annotated frames to an output video file.
  - annotate_frame(): Draws bounding boxes and match/no-match labels.

Usage:
    reader = VideoReader("data/videos/video_1.mp4")
    writer = VideoWriter("data/output/output.mp4", reader.fps, reader.width, reader.height)

    for frame, frame_idx, timestamp in reader:
        annotated = annotate_frame(frame, faces_info)
        writer.write(annotated)

    reader.release()
    writer.release()
"""

import logging
import cv2

logger = logging.getLogger(__name__)


class VideoReader:
    """
    Reads frames from a video file using OpenCV.

    Yields (frame, frame_index, timestamp_sec) tuples.
    Supports frame skipping for faster processing.
    """

    def __init__(self, video_path):
        """
        Open a video file for reading.

        Args:
            video_path: Path to the video file.

        Raises:
            FileNotFoundError: If the video file doesn't exist.
            RuntimeError: If the video cannot be opened.
        """
        self.video_path = video_path
        self._cap = cv2.VideoCapture(video_path)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Opened video: %s (%dx%d, %.1f fps, %d frames)",
            video_path, self.width, self.height, self.fps, self.total_frames
        )

    def frames(self, skip=1):
        """
        Yield frames with index and timestamp.

        Args:
            skip: Process every Nth frame (1 = every frame).

        Yields:
            Tuple of (frame_bgr, frame_index, timestamp_seconds).
        """
        frame_index = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            if frame_index % skip == 0:
                timestamp_sec = frame_index / self.fps
                yield frame, frame_index, timestamp_sec

            frame_index += 1

    def release(self):
        """Release the video capture resource."""
        if self._cap:
            self._cap.release()
            logger.info("Video reader released: %s", self.video_path)


class VideoWriter:
    """
    Writes frames to an output video file using OpenCV.
    """

    def __init__(self, output_path, fps, width, height):
        """
        Open an output video file for writing.

        Args:
            output_path: Path for the output video.
            fps: Frames per second.
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {output_path}")

        logger.info("Video writer opened: %s (%dx%d, %.1f fps)", output_path, width, height, fps)

    def write(self, frame):
        """Write a single frame to the output video."""
        self._writer.write(frame)

    def release(self):
        """Release the video writer resource."""
        if self._writer:
            self._writer.release()
            logger.info("Video writer released: %s", self.output_path)


def annotate_frame(frame, detections, frame_index, enrollment_count=0):
    """
    Draw bounding boxes and labels on a frame for each detection result.

    Args:
        frame: BGR image (numpy array). Will be modified in-place.
        detections: List of dicts with keys:
            - 'bbox': [x1, y1, x2, y2]
            - 'match_result': "MATCH" or "NO_MATCH"
            - 'matched_person_id': str (e.g., "Person_01") or None
            - 'similarity': float
        frame_index: Current frame index (for the overlay).
        enrollment_count: Total enrolled persons (for the overlay).

    Returns:
        The annotated frame (same reference as input).
    """
    # Import colors from config
    from config import COLOR_MATCH, COLOR_NO_MATCH, COLOR_TEXT_BG, COLOR_TEXT_FG

    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        match_result = det.get("match_result", "NO_MATCH")
        similarity = det.get("similarity", 0.0)
        person_id = det.get("matched_person_id", "Unknown")

        if match_result == "MATCH":
            color = COLOR_MATCH
            label = f"MATCH: {person_id} (sim: {similarity:.2f})"
        else:
            color = COLOR_NO_MATCH
            label = f"NO MATCH (sim: {similarity:.2f})"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_y = max(text_h + 10, y1 - 8)
        cv2.rectangle(
            frame,
            (x1, label_y - text_h - 6),
            (x1 + text_w + 6, label_y + baseline),
            COLOR_TEXT_BG,
            -1
        )

        # Draw label text
        cv2.putText(
            frame, label, (x1 + 3, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_FG, 2, cv2.LINE_AA
        )

    # Info overlay (top-left)
    info_lines = [
        f"Frame: {frame_index}",
        f"Faces: {len(detections)}",
        f"Enrolled: {enrollment_count}",
    ]

    # Draw semi-transparent background for info
    overlay_h = 25 * len(info_lines) + 15
    overlay = frame[5:5 + overlay_h, 5:250].copy()
    cv2.rectangle(frame, (5, 5), (250, 5 + overlay_h), (30, 30, 30), -1)
    # Blend for transparency effect
    cv2.addWeighted(overlay, 0.3, frame[5:5 + overlay_h, 5:250], 0.7, 0,
                    frame[5:5 + overlay_h, 5:250])

    for i, line in enumerate(info_lines):
        cv2.putText(
            frame, line, (12, 28 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA
        )

    return frame
