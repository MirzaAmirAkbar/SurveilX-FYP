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

  // Sync with backend restricted areas periodically
  useEffect(() => {
    const fetchAreas = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/restricted-areas/");
        const data = await res.json();
        if (data.areas && Array.isArray(data.areas)) {
          const mapped = data.areas.map((area) => ({
            id: area.id,
            name: area.name,
            coords: { x: area.x, y: area.y, width: area.width, height: area.height },
          }));
          setRectangles(mapped);
          setRestrictedAreas(mapped);
        }
      } catch (e) {
        console.warn("Could not fetch restricted areas from backend");
      }
    };
    const timer = setInterval(fetchAreas, 1000);
    return () => clearInterval(timer);
  }, [setRestrictedAreas]);

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

  if (currentRect && currentRect.width > 5 && currentRect.height > 5) {
    const name = prompt("Enter name for this restricted area:");
    if (name && videoRef.current && overlayRef.current) {
      const videoWidth = videoRef.current.naturalWidth || 0;
      const videoHeight = videoRef.current.naturalHeight || 0;
      
      const overlayRect = overlayRef.current.getBoundingClientRect();
      const displayWidth = overlayRect.width;
      const displayHeight = overlayRect.height;

      if (videoWidth === 0 || videoHeight === 0) {
        console.warn("Could not determine video dimensions");
        setCurrentRect(null);
        return;
      }

      const scaleX = videoWidth / displayWidth;
      const scaleY = videoHeight / displayHeight;

      const trueCoords = {
        x: Math.round(currentRect.x * scaleX),
        y: Math.round(currentRect.y * scaleY),
        width: Math.round(currentRect.width * scaleX),
        height: Math.round(currentRect.height * scaleY),
      };

      console.log("📦 New restricted area (true coords):", { name, ...trueCoords });

      // Send to backend
      try {
        const response = await fetch("http://127.0.0.1:8000/restricted-areas/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            name,
            coords: trueCoords,
          }),
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
        {/*}
        <video
          ref={videoRef}
          className="feed-video"
          src={videoSrc}
          autoPlay
          loop
          muted
          playsInline
        />
        */}

        <img ref={videoRef} src={videoSrc} alt="Live feed" className="feed-video" />
        

        <div
          className="overlay-layer"
          ref={overlayRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          style={{ cursor: isDrawingMode ? "crosshair" : "default" }}
        >
          {restrictedAreas
            .filter(area => visibleAreas.includes(area.id))
            .map((area) => {
              if (!videoRef.current || !overlayRef.current) return null;

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
