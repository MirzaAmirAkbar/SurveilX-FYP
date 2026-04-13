"""
Shoplifting Detection - Data Pipeline & Inference
Handles frame preprocessing, uniform sampling, normalization,
and running predictions through the Gate_Flow_SlowFast model.
"""
import os
import cv2
import numpy as np
from datetime import datetime

from shoplifting_net import ShopliftingNet
from config import (
    INPUT_FRAMES, INPUT_HEIGHT, INPUT_WIDTH,
    CLASS_NAMES, OUTPUT_DIR, OUTPUT_FPS, OUTPUT_CODEC,
    SLIDING_WINDOW_STEP
)


class ShopliftingDetector:
    """
    Complete shoplifting detection pipeline:
    model loading → frame processing → prediction → result saving.
    """

    def __init__(self, weights_path):
        self.net = ShopliftingNet(weights_path)
        self.model = None

    def load_model(self):
        """Build and load the pre-trained model."""
        print("[*] Building Gate_Flow_SlowFast model...")
        self.model = self.net.load_model_and_weight()
        print("[+] Model ready for inference.")

    # ─── Frame Processing ────────────────────────────────────────────────

    @staticmethod
    def uniform_sampling(frames, target_frames=INPUT_FRAMES):
        """
        Uniformly sample `target_frames` from the input frame list.
        Pads with trailing frames if fewer than target.
        """
        n = len(frames)
        interval = int(np.ceil(n / target_frames))

        sampled = []
        for i in range(0, n, interval):
            sampled.append(frames[i])

        # Pad if needed
        num_pad = target_frames - len(sampled)
        if num_pad > 0:
            for i in range(-num_pad, 0):
                try:
                    sampled.append(frames[i])
                except IndexError:
                    sampled.append(frames[0])

        return np.array(sampled[:target_frames], dtype=np.float32)

    @staticmethod
    def normalize(data):
        """Z-score normalization."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return data - mean
        return (data - mean) / std

    @staticmethod
    def resize_and_convert(frame, size=(INPUT_WIDTH, INPUT_HEIGHT)):
        """Resize frame to model input size and convert BGR→RGB."""
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.reshape((INPUT_HEIGHT, INPUT_WIDTH, 3))

    def make_frame_set(self, frames):
        """Convert a list of raw BGR frames to model-ready format."""
        processed = [self.resize_and_convert(f) for f in frames]
        return np.array(processed)

    def frame_preprocessing(self, frames):
        """
        Full preprocessing pipeline:
        1. Uniform sample to 64 frames
        2. Normalize pixel values
        3. Reshape for model input: (1, 64, 224, 224, 3)
        """
        result = self.uniform_sampling(frames, target_frames=INPUT_FRAMES)
        result[..., :3] = self.normalize(result)
        return result.reshape((-1, INPUT_FRAMES, INPUT_HEIGHT, INPUT_WIDTH, 3))

    # ─── Prediction ──────────────────────────────────────────────────────

    def predict_single(self, preprocessed_frames):
        """
        Run model prediction on preprocessed frames.
        Returns: (bag_prob, clothes_prob, normal_prob)
        """
        predictions = self.model.predict(preprocessed_frames, verbose=0)
        preds = predictions[0]
        return (round(preds[0].item(), 3),
                round(preds[1].item(), 3),
                round(preds[2].item(), 3))

    def classify(self, bag, clothes, normal):
        """
        Determine if the prediction indicates theft.
        Returns: (is_theft: bool, event_type: str, event_index: int)
        """
        if normal < bag and normal < clothes:
            if bag > clothes:
                return True, "Bag", 0
            else:
                return True, "Clothes", 1
        return False, "Normal", 2

    def split_into_windows(self, frames, window_size=INPUT_FRAMES,
                           step=SLIDING_WINDOW_STEP):
        """
        Split frame set into overlapping windows of `window_size`.
        Uses sliding window with `step` overlap.
        """
        windows = []
        n = len(frames)
        start = 0
        while start + window_size <= n:
            windows.append(frames[start:start + window_size])
            start += step
        # Handle remaining frames
        if start < n and n >= window_size:
            windows.append(frames[n - window_size:n])
        elif not windows and n > 0:
            # If video is shorter than window_size, use what we have
            windows.append(frames)
        return windows

    def predict_frame_set(self, frame_set):
        """
        Process a set of frames (can be > 64 frames).
        Splits into windows and aggregates predictions.
        Returns: (bag, clothes, normal, is_theft, event_type)
        """
        processed = self.make_frame_set(frame_set)
        windows = self.split_into_windows(processed)

        bag_scores, clothes_scores, normal_scores = [], [], []
        theft_count = 0

        for window in windows:
            preprocessed = self.frame_preprocessing(window)
            bag, clothes, normal = self.predict_single(preprocessed)
            bag_scores.append(bag)
            clothes_scores.append(clothes)
            normal_scores.append(normal)

            is_theft, _, _ = self.classify(bag, clothes, normal)
            if is_theft:
                theft_count += 1

        # Aggregate: use max theft scores
        avg_bag = round(max(bag_scores), 3)
        avg_clothes = round(max(clothes_scores), 3)
        avg_normal = round(min(normal_scores), 3)

        is_theft, event_type, event_idx = self.classify(
            avg_bag, avg_clothes, avg_normal
        )

        return avg_bag, avg_clothes, avg_normal, is_theft, event_type

    # ─── Video Output ────────────────────────────────────────────────────

    @staticmethod
    def save_annotated_video(output_path, frames, prediction, w, h):
        """
        Save frames as video with detection annotations overlaid.
        """
        bag, clothes, normal, is_theft, event_type = prediction
        fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
        out = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (w, h))

        for frame in frames:
            if is_theft:
                # Red banner for theft
                cv2.putText(
                    frame,
                    f"THEFT ALERT - {event_type}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 255), 3
                )
                cv2.putText(
                    frame,
                    f"Bag: {bag * 100:.1f}%",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"Clothes: {clothes * 100:.1f}%",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"Normal: {normal * 100:.1f}%",
                    (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2
                )
            else:
                cv2.putText(
                    frame,
                    "Normal Activity",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 255, 0), 3
                )

            out.write(frame)

        out.release()
        return output_path
