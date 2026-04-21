from __future__ import annotations

import json
import time
from pathlib import Path
from dotenv import load_dotenv

from talkdown.manager import TalkdownManager
from talkdown.types import PersonAttributes, TalkdownEvent

# Load environment variables from .env file if it exists
load_dotenv()

def load_detections(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("detections", [])


def main():
    base_dir = Path(__file__).parent
    detections_path = base_dir / "detections.json"
    detections = load_detections(detections_path)

    manager = TalkdownManager()

    print("Starting simulated talkdown for detections.json")
    start_ts = time.time()

    # For demo: trigger restricted_area_breach for each person with staggered times
    for idx, det in enumerate(detections):
        person_id = det["person_id"]
        attrs_raw = det["attributes"]
        attrs = PersonAttributes(
            upper_clothing=attrs_raw.get("upper_clothing", ""),
            lower_clothing=attrs_raw.get("lower_clothing", ""),
            accessories=attrs_raw.get("accessories", []),
            footwear=attrs_raw.get("footwear", ""),
            gender=attrs_raw.get("gender"),
        )
        now = time.time()
        event = TalkdownEvent(
            person_id=person_id,
            event_type="restricted_area_breach",
            attributes=attrs,
            first_seen_ts=now,
            last_seen_ts=now,
            camera_id="cam-1",
            zone_id="restricted-zone-1",
        )
        manager.handle_event(event)
        # delay between people
        time.sleep(3)

    # Run tick loop for ~15 seconds to observe escalation
    end_time = time.time() + 15
    while time.time() < end_time:
        manager.tick()
        time.sleep(3)

    print("Simulation complete.")


if __name__ == "__main__":
    main()

