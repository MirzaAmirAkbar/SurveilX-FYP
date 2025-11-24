import cv2
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pd import RealTimePersonDetector  # assuming your file is named pd.py
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

VIDEO_PATH = "vid1_v2.mp4"
stop_stream = False
rectangles = [
    {"id": 1, "x": 484, "y": 204, "width": 137, "height": 138, "name": "area 1"},
    {"id": 2, "x": 8, "y": 185, "width": 105, "height": 168, "name": "area2"},
]
_next_area_id = 3

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize YOLO detector once (outside generator)
print("🚀 Initializing YOLO model...")
detector = RealTimePersonDetector(performance_mode='balanced')
print("✅ YOLO model ready!")


async def generate_frames(request: Request):
    global stop_stream
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1 / fps

    # rectangles comes from module-level, can be updated via API

    print("🎥 Starting YOLO inference stream...")

    try:
        while not stop_stream:
            if await request.is_disconnected():
                print("🚪 Client disconnected from stream.")
                break

            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Run YOLO detection on frame
            enhanced_frame = detector.enhance_low_light(frame)
            detections = detector.detect_persons(enhanced_frame)
            persons = detector.track_persons(detections, frame)
            frame = detector.draw_detections(frame, persons)

            # Draw zones, info text, and breach alerts
            breaches = detector.detect_breaches_with_ids(persons, rectangles)
            for pid, zone_name in breaches:
                pdata = persons.get(pid)
                if pdata:
                    detector.create_alert_if_new(frame, pid, pdata['bbox'], zone_name)

            # Do not draw zone boxes on backend; frontend overlays will display them

            # Info text
            info_text = (
                f"Persons: {len(persons)}"
            )
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Breach text
            zones_now = {zone for (_, zone) in breaches}
            breach_text = (" | ".join([f"{zone} breached" for zone in zones_now])
                if zones_now else "All zones clear")
            cv2.putText(frame, breach_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if zones_now else (0, 255, 0), 2)

            # Encode frame for streaming
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            await asyncio.sleep(frame_delay)

    except asyncio.CancelledError:
        print("⚠️ Stream task cancelled.")
    finally:
        cap.release()
        print("✅ Stream stopped cleanly.")


@app.get("/video_feed")
async def video_feed(request: Request):
    return StreamingResponse(
        generate_frames(request),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/alerts")
async def get_alerts():
    # Return alerts list (most recent first)
    alerts = list(reversed(detector.alerts))
    return JSONResponse(content={"alerts": alerts})


class RestrictedArea(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    coords: dict


@app.post("/restricted-areas/")
async def add_restricted_area(area: RestrictedArea):
    global rectangles, _next_area_id
    c = area.coords
    name = area.name or f"area {len(rectangles)+1}"
    rect = {"id": _next_area_id,
            "x": int(c.get("x", 0)), "y": int(c.get("y", 0)),
            "width": int(c.get("width", 0)), "height": int(c.get("height", 0)),
            "name": name}
    rectangles.append(rect)
    _next_area_id += 1
    return {"status": "ok", "area": rect, "count": len(rectangles)}


@app.get("/restricted-areas/")
async def list_restricted_areas():
    return {"areas": rectangles}


@app.delete("/restricted-areas/{area_id}")
async def delete_restricted_area(area_id: int):
    global rectangles
    before = len(rectangles)
    rectangles = [r for r in rectangles if int(r.get("id", -1)) != area_id]
    deleted = before - len(rectangles)
    return {"status": "ok", "deleted": deleted}


@app.on_event("shutdown")
async def shutdown_event():
    global stop_stream
    stop_stream = True
    print("⚠️ Backend shutting down... stopping video feed.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
