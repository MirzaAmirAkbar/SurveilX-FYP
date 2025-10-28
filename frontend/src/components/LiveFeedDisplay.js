import React, { useRef, useState, useEffect } from "react";
import "./LiveFeedDisplay.css";

function LiveFeedDisplay({ selectedCamera, isDrawingMode, visibleAreas, restrictedAreas, setRestrictedAreas }) {

  const videoRef = useRef(null);
  const overlayRef = useRef(null);

  const [rectangles, setRectangles] = useState([]);
  const [drawing, setDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentRect, setCurrentRect] = useState(null);

  const videoSrc = "http://127.0.0.1:8000/video_feed";

  // ✅ Make sure image is loaded before using its dimensions
  const [imgLoaded, setImgLoaded] = useState(false);
  useEffect(() => {
    const img = videoRef.current;
    if (img) {
      img.onload = () => {
        setImgLoaded(true);
        console.log("✅ Image loaded:", img.naturalWidth, img.naturalHeight);
      };
    }
  }, [videoSrc]);

  const handleMouseDown = (e) => {
    if (!isDrawingMode) return;
    const rect = overlayRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setStartPos({ x, y });
    setDrawing(true);
    setCurrentRect({ x, y, width: 0, height: 0 });
  };

  const handleMouseMove = (e) => {
    if (!drawing || !isDrawingMode) return;
    const rect = overlayRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setCurrentRect({
      x: Math.min(x, startPos.x),
      y: Math.min(y, startPos.y),
      width: Math.abs(x - startPos.x),
      height: Math.abs(y - startPos.y),
    });
  };

  const handleMouseUp = async () => {
    if (!drawing || !isDrawingMode) return;
    setDrawing(false);

    if (currentRect && currentRect.width > 5 && currentRect.height > 5 && imgLoaded) {
      const name = prompt("Enter name for this restricted area:");
      if (name && videoRef.current && overlayRef.current) {
        // ✅ Use image natural dimensions
        const videoWidth = videoRef.current.naturalWidth;
        const videoHeight = videoRef.current.naturalHeight;

        const overlayRect = overlayRef.current.getBoundingClientRect();
        const displayWidth = overlayRect.width;
        const displayHeight = overlayRect.height;

        const scaleX = videoWidth / displayWidth;
        const scaleY = videoHeight / displayHeight;

        const trueCoords = {
          x: Math.round(currentRect.x * scaleX),
          y: Math.round(currentRect.y * scaleY),
          width: Math.round(currentRect.width * scaleX),
          height: Math.round(currentRect.height * scaleY),
        };

        const newArea = {
          id: Date.now(),
          name,
          coords: trueCoords,
        };

        console.log("📦 New restricted area (true coords):", newArea);

        setRectangles((prev) => [...prev, newArea]);
        if (setRestrictedAreas) setRestrictedAreas((prev) => [...prev, newArea]);

        try {
          const response = await fetch("http://127.0.0.1:8000/restricted-areas/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(newArea),
          });

          const data = await response.json();
          console.log("✅ Sent to backend:", data);
        } catch (error) {
          console.error("❌ Error sending to backend:", error);
        }
      }
    }

    setCurrentRect(null);
  };

  return (
    <div className="livefeed-container">
      <div className="detection-info">Detected: 3 persons</div>

      <div className="video-wrapper">
        {/* <video ... /> replaced by <img> */}
        <img
          ref={videoRef}
          src={videoSrc}
          alt="Live feed"
          className="feed-video"
        />

        <div
          className="overlay-layer"
          ref={overlayRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          {restrictedAreas
            .filter((area) => visibleAreas.includes(area.id))
            .map((area) => {
              if (!videoRef.current || !overlayRef.current || !imgLoaded) return null;

              // ✅ Use naturalWidth / naturalHeight here too
              const videoWidth = videoRef.current.naturalWidth;
              const videoHeight = videoRef.current.naturalHeight;
              const overlayRect = overlayRef.current.getBoundingClientRect();
              const displayWidth = overlayRect.width;
              const displayHeight = overlayRect.height;

              const scaleX = displayWidth / videoWidth;
              const scaleY = displayHeight / videoHeight;

              const displayX = area.coords.x * scaleX;
              const displayY = area.coords.y * scaleY;
              const displayWidthScaled = area.coords.width * scaleX;
              const displayHeightScaled = area.coords.height * scaleY;

              return (
                <div
                  key={area.id}
                  className="restricted-box"
                  style={{
                    left: displayX,
                    top: displayY,
                    width: displayWidthScaled,
                    height: displayHeightScaled,
                  }}
                />
              );
            })}

          {currentRect && (
            <div
              className="restricted-box drawing"
              style={{
                left: currentRect.x,
                top: currentRect.y,
                width: currentRect.width,
                height: currentRect.height,
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default LiveFeedDisplay;
