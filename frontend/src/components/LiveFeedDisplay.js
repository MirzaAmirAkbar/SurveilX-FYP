import React, { useRef, useState, useEffect } from "react";
import "./LiveFeedDisplay.css";

function LiveFeedDisplay({ drawingTool, setDrawingTool, restrictedAreas }) {
  const videoRef = useRef(null);

  // Polygon / Freehand State
  const [polyPoints, setPolyPoints] = useState([]); // [{x,y}, ...]
  const [isDragging, setIsDragging] = useState(false);

  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });
  const videoSrc = "http://127.0.0.1:8000/video_feed";

  // Reset drawing state when tool changes
  useEffect(() => {
    setPolyPoints([]);
    setIsDragging(false);
  }, [drawingTool]);

  const getScale = () => {
    if (!videoRef.current) return { scaleX: 1, scaleY: 1 };
    const vid = videoRef.current;
    const natW = vid.naturalWidth || 640;
    const natH = vid.naturalHeight || 480;
    const dispW = vid.clientWidth;
    const dispH = vid.clientHeight;
    return { scaleX: natW / dispW, scaleY: natH / dispH };
  };

  const getRelativeCoords = (e) => {
    const rect = videoRef.current.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  // --- MOUSE HANDLERS ---

  const handleMouseDown = (e) => {
    if (!drawingTool) return;
    e.preventDefault();
    const pos = getRelativeCoords(e);

    if (drawingTool === "POLYGON") {
      // Add single point
      setPolyPoints([...polyPoints, pos]);
    } else if (drawingTool === "FREEHAND") {
      // Start freehand stream
      setIsDragging(true);
      setPolyPoints([pos]);
    }
  };

  const handleMouseMove = (e) => {
    if (!drawingTool) return;
    const pos = getRelativeCoords(e);
    setCursorPos(pos);

    if (drawingTool === "FREEHAND" && isDragging) {
      // Add points continuously
      setPolyPoints((prev) => [...prev, pos]);
    }
  };

  const handleMouseUp = async (e) => {
    if (drawingTool === "FREEHAND" && isDragging) {
      setIsDragging(false);
      finishPolygonShape("Freehand Area");
    }
  };

  const handleRightClick = async (e) => {
    e.preventDefault();
    if (drawingTool === "POLYGON") {
      finishPolygonShape("Polygon Area");
    }
  };

  const finishPolygonShape = async (defaultName) => {
    if (polyPoints.length < 3) {
      setPolyPoints([]);
      return;
    }

    // Optional: Simplify points for freehand if needed (sampling every Nth point)
    // For now we send all points.

    const name = prompt(`Enter name for this ${defaultName}:`);
    if (name) {
      const { scaleX, scaleY } = getScale();
      const points = polyPoints.map((p) => [
        Math.round(p.x * scaleX),
        Math.round(p.y * scaleY),
      ]);

      await saveArea({ name, type: "polygon", points });
    }
    setPolyPoints([]);
    setDrawingTool(null);
  };

  const saveArea = async (payload) => {
    try {
      await fetch("http://127.0.0.1:8000/restricted-areas/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (e) {
      console.error(e);
    }
  };

  // --- RENDER HELPERS ---

  const renderSavedAreas = () => {
    const { scaleX, scaleY } = getScale();

    return restrictedAreas
      .filter((area) => area.is_active) // Only draw if active
      .map((area) => {
        if (area.type === "ellipse" && area.center && area.radii) {
          return (
            <ellipse
              key={area.id}
              cx={area.center[0] / scaleX}
              cy={area.center[1] / scaleY}
              rx={area.radii[0] / scaleX}
              ry={area.radii[1] / scaleY}
              fill="rgba(255, 0, 0, 0.2)"
              stroke="red"
              strokeWidth="2"
            />
          );
        } else if (area.points) {
          const pts = area.points
            .map((p) => `${p[0] / scaleX},${p[1] / scaleY}`)
            .join(" ");
          return (
            <polygon
              key={area.id}
              points={pts}
              fill="rgba(255, 0, 0, 0.2)"
              stroke="red"
              strokeWidth="2"
            />
          );
        }
        return null;
      });
  };

  return (
    <div className="livefeed-container">
      <div className="detection-info">
        {drawingTool === "POLYGON"
          ? "Left-click to add points, Right-click to finish"
          : drawingTool === "FREEHAND"
          ? "Hold Left-click and drag to draw"
          : "Monitoring Active"}
      </div>

      <div className="video-wrapper">
        <img
          ref={videoRef}
          src={videoSrc}
          alt="Live feed"
          className="feed-video"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onContextMenu={handleRightClick}
          style={{ cursor: drawingTool ? "crosshair" : "default" }}
        />

        <svg className="overlay-layer" style={{ pointerEvents: "none" }}>
          {renderSavedAreas()}

          {/* Current Drawing */}
          {(drawingTool === "POLYGON" || drawingTool === "FREEHAND") &&
            polyPoints.length > 0 && (
              <>
                <polyline
                  points={polyPoints.map((p) => `${p.x},${p.y}`).join(" ")}
                  fill="none"
                  stroke="#00ff00"
                  strokeWidth="2"
                />
                {drawingTool === "POLYGON" && (
                  <line
                    x1={polyPoints[polyPoints.length - 1].x}
                    y1={polyPoints[polyPoints.length - 1].y}
                    x2={cursorPos.x}
                    y2={cursorPos.y}
                    stroke="#00ff00"
                    strokeWidth="1"
                    strokeDasharray="4"
                  />
                )}
              </>
            )}
        </svg>
      </div>
    </div>
  );
}

export default LiveFeedDisplay;
