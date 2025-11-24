import cv2
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pd import RealTimePersonDetector
from pydantic import BaseModel
from typing import Optional, List, Union

app = FastAPI()

VIDEO_PATH = "vid1_v2.mp4"
stop_stream = False

# Areas storage
restricted_areas = [] 
_next_area_id = 1

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Initializing YOLO model...")
detector = RealTimePersonDetector(performance_mode='balanced')
print("✅ YOLO model ready!")

async def generate_frames(request: Request):
    global stop_stream
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1 / fps

    print("🎥 Starting stream...")
    try:
        while not stop_stream:
            if await request.is_disconnected():
                print("🚪 Client disconnected.")
                break

            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            enhanced = detector.enhance_low_light(frame)
            detections = detector.detect_persons(enhanced)
            persons = detector.track_persons(detections, frame)
            frame = detector.draw_detections(frame, persons)

            # Check breaches
            breaches = detector.detect_breaches_with_ids(persons, restricted_areas)
            for pid, zone_name in breaches:
                pdata = persons.get(pid)
                if pdata:
                    detector.create_alert_if_new(frame, pid, pdata['bbox'], zone_name)

            # Draw areas
            for area in restricted_areas:
                frame = detector.draw_area_overlay(frame, area)

            zones_now = {zone for (_, zone) in breaches}
            breach_text = (" | ".join([f"{zone} breached" for zone in zones_now])
                           if zones_now else "All zones clear")
            cv2.putText(frame, breach_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if zones_now else (0, 255, 0), 2)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            await asyncio.sleep(frame_delay)

    except asyncio.CancelledError:
        print("⚠️ Stream cancelled.")
    finally:
        cap.release()

@app.get("/video_feed")
async def video_feed(request: Request):
    return StreamingResponse(generate_frames(request), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/alerts")
async def get_alerts():
    return JSONResponse(content={"alerts": list(reversed(detector.alerts))})

# Generalized Area Model
class RestrictedAreaModel(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    type: str = "polygon" # 'polygon' or 'ellipse'
    
    # For Polygon
    points: Optional[List[List[int]]] = [] 
    
    # For Ellipse
    center: Optional[List[int]] = None # [x, y]
    radii: Optional[List[int]] = None  # [rx, ry]

@app.post("/restricted-areas/")
async def add_restricted_area(area: RestrictedAreaModel):
    global restricted_areas, _next_area_id
    name = area.name or f"Zone {_next_area_id}"
    
    new_area = {
        "id": _next_area_id,
        "name": name,
        "type": area.type
    }
    
    if area.type == "polygon":
        new_area["points"] = area.points
    elif area.type == "ellipse":
        new_area["center"] = area.center
        new_area["radii"] = area.radii
        
    restricted_areas.append(new_area)
    _next_area_id += 1
    return {"status": "ok", "area": new_area}

@app.get("/restricted-areas/")
async def list_restricted_areas():
    return {"areas": restricted_areas}

@app.delete("/restricted-areas/{area_id}")
async def delete_restricted_area(area_id: int):
    global restricted_areas
    restricted_areas = [r for r in restricted_areas if r["id"] != area_id]
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)