from collections import deque
import numpy as np
import time

class RunningDetector:
    def __init__(self, history_len=5, run_threshold=0.05, fps=30):
        # Configuration
        self.history_len = history_len
        self.run_threshold = run_threshold # Heights per frame
        self.fps = fps
        
        # State: {track_id: deque([(x, y), ...])}
        self.track_histories = {}
        
        # Cooldown State: {track_id: last_alert_timestamp}
        self.alert_cooldowns = {}
        self.COOLDOWN_SECONDS = 5  # Don't alert same person running for 5s

    def update(self, track_id, center_x, center_y, box_height):
        """
        Returns: (is_running: bool, speed_score: float)
        """
        # Initialize history for new track
        if track_id not in self.track_histories:
            self.track_histories[track_id] = deque(maxlen=self.history_len)
        
        # Update history
        self.track_histories[track_id].append((center_x, center_y))

        # Need full history buffer to calculate speed accurately
        if len(self.track_histories[track_id]) < self.history_len:
            return False, 0.0

        # Calculate Distance (Euclidean)
        start_pos = self.track_histories[track_id][0]
        end_pos = self.track_histories[track_id][-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        dist_pixels = np.sqrt(dx**2 + dy**2)
        
        # Average pixels moved per frame
        # (len - 1) because 5 points create 4 intervals
        avg_dist_per_frame = dist_pixels / (self.history_len - 1)
        
        # Normalize by height (Speed relative to body size)
        speed_score = avg_dist_per_frame / box_height
        
        is_running = speed_score > self.run_threshold
        return is_running, speed_score

    def should_alert(self, track_id):
        """
        Checks if we can send an alert for this user (Cooldown logic).
        """
        current_time = time.time()
        last_alert = self.alert_cooldowns.get(track_id, 0)
        
        if current_time - last_alert > self.COOLDOWN_SECONDS:
            self.alert_cooldowns[track_id] = current_time
            return True
        return False

    def cleanup(self, active_track_ids):
        """Remove data for IDs that are no longer tracked to save memory"""
        stored_ids = list(self.track_histories.keys())
        for tid in stored_ids:
            if tid not in active_track_ids:
                del self.track_histories[tid]
                if tid in self.alert_cooldowns:
                    del self.alert_cooldowns[tid]