import cv2
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pd import RealTimePersonDetector  # assuming your file is named pd.py

app = FastAPI()

VIDEO_PATH = "vid1_v2.mp4"
stop_stream = False

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

    # Define static rectangles (same as in pd.py)
    rectangles = [
        {"x": 484, "y": 204, "width": 137, "height": 138, "name": "area 1"},
        {"x": 8, "y": 185, "width": 105, "height": 168, "name": "area2"},
    ]

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

            # 🧩 ADD THIS — Draw zones, info text, and breach alerts
            breached_zones = detector.check_zone_breach(persons, rectangles)

            # Draw translucent rectangles for zones
            for rect in rectangles:
                frame = detector.draw_translucent_rectangle(
                    frame,
                    rect["x"], rect["y"],
                    rect["width"], rect["height"],
                    name=rect.get("name", None)
                )

            # Info text
            info_text = (
                f"Persons: {len(persons)} | "
                f"Known: {sum(1 for p in persons.values() if p['type'] == 'Known')} | "
                f"Unknown: {sum(1 for p in persons.values() if p['type'] == 'Unknown')}"
            )
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Breach text
            breach_text = (
                " | ".join([f"{zone} breached" for zone in breached_zones])
                if breached_zones else "All zones clear"
            )
            cv2.putText(frame, breach_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if breached_zones else (0, 255, 0), 2)

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


@app.on_event("shutdown")
async def shutdown_event():
    global stop_stream
    stop_stream = True
    print("⚠️ Backend shutting down... stopping video feed.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
