import React, { useRef, useState, useEffect } from "react";
import "./LiveFeedDisplay.css";

function LiveFeedDisplay({ 
  drawingTool, 
  setDrawingTool,
  drawingType, // 'RESTRICTED' or 'LOITERING'
  setDrawingType,
  restrictedAreas, 
  loiteringAreas,
  onRefreshAreas
}) {
  const videoRef = useRef(null);

  // Polygon / Freehand State
  const [polyPoints, setPolyPoints] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });
  
  const videoSrc = "http://127.0.0.1:8000/video_feed";

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
      setPolyPoints([...polyPoints, pos]);
    } else if (drawingTool === "FREEHAND") {
      setIsDragging(true);
      setPolyPoints([pos]);
    }
  };

  const handleMouseMove = (e) => {
    if (!drawingTool) return;
    const pos = getRelativeCoords(e);
    setCursorPos(pos);

    if (drawingTool === "FREEHAND" && isDragging) {
      setPolyPoints((prev) => [...prev, pos]);
    }
  };

  const handleMouseUp = async (e) => {
    if (drawingTool === "FREEHAND" && isDragging) {
      setIsDragging(false);
      finishPolygonShape();
    }
  };

  const handleRightClick = async (e) => {
    e.preventDefault();
    if (drawingTool === "POLYGON") {
      finishPolygonShape();
    }
  };

  const finishPolygonShape = async () => {
    if (polyPoints.length < 3) {
      setPolyPoints([]);
      return;
    }

    const label = drawingType === 'RESTRICTED' ? 'Restricted Area' : 'Loitering Area';
    const name = prompt(`Enter name for this ${label}:`);
    
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
    setDrawingType(null);
  };

  const saveArea = async (payload) => {
    // Determine Endpoint based on drawingType
    const endpoint = drawingType === 'RESTRICTED' ? 'restricted-areas' : 'loitering-areas';

    try {
      const response = await fetch(`http://127.0.0.1:8000/${endpoint}/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        alert("Failed to save area.");
      } else {
        if (onRefreshAreas) onRefreshAreas();
      }
    } catch (e) {
      console.error("Network error saving area:", e);
    }
  };

  // --- RENDER HELPERS ---
  
  // Helper to render a list of areas with specific color style
  const renderAreaList = (list, colorStroke, colorFill) => {
    const { scaleX, scaleY } = getScale();
    return list
      .filter((area) => area.is_active)
      .map((area) => {
        if (area.points) {
          const pts = area.points
            .map((p) => `${p[0] / scaleX},${p[1] / scaleY}`)
            .join(" ");
          return (
            <polygon
              key={`${area.id}-${colorStroke}`}
              points={pts}
              fill={colorFill}
              stroke={colorStroke}
              strokeWidth="2"
            />
          );
        }
        return null;
      });
  };

  // Determine drawing line color based on active type
  const drawColor = drawingType === 'RESTRICTED' ? '#ff0000' : '#0099ff';

  return (
    <div className="livefeed-container">
      <div className="detection-info">
        {drawingTool 
          ? `Drawing ${drawingType === 'RESTRICTED' ? 'Restricted' : 'Loitering'} Area...` 
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
          {/* Render Restricted Areas (RED) */}
          {renderAreaList(restrictedAreas, "red", "rgba(255, 0, 0, 0.2)")}
          
          {/* Render Loitering Areas (BLUE) */}
          {renderAreaList(loiteringAreas, "#0099ff", "rgba(0, 153, 255, 0.2)")}

          {/* Render Active Drawing Lines */}
          {(drawingTool === "POLYGON" || drawingTool === "FREEHAND") &&
            polyPoints.length > 0 && (
              <>
                <polyline
                  points={polyPoints.map((p) => `${p.x},${p.y}`).join(" ")}
                  fill="none"
                  stroke={drawColor}
                  strokeWidth="2"
                />
                {drawingTool === "POLYGON" && (
                  <line
                    x1={polyPoints[polyPoints.length - 1].x}
                    y1={polyPoints[polyPoints.length - 1].y}
                    x2={cursorPos.x}
                    y2={cursorPos.y}
                    stroke={drawColor}
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