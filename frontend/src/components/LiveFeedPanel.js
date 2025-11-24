import React, { useState, useEffect } from "react";
import CameraCard from "./CameraCard.js";
import RestrictedAreaCard from "./RestrictedAreaCard.js";
import LiveFeedDisplay from "./LiveFeedDisplay.js";
import "./LiveFeedPanel.css";

function LiveFeedPanel() {
  const [selectedCamera, setSelectedCamera] = useState(1);

  // 'POLYGON', 'FREEHAND', or null
  const [drawingTool, setDrawingTool] = useState(null);

  const cameras = [
    { id: 1, name: "Aisle 1" },
    { id: 2, name: "Aisle 2" },
    { id: 3, name: "Aisle 3" },
  ];

  const [restrictedAreas, setRestrictedAreas] = useState([]);

  // Fetch areas periodically to stay in sync with backend status
  useEffect(() => {
    const fetchAreas = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/restricted-areas/");
        const data = await res.json();
        if (data.areas) {
          setRestrictedAreas(data.areas);
        }
      } catch (e) {
        console.warn("Backend fetch error", e);
      }
    };
    fetchAreas();
    const interval = setInterval(fetchAreas, 1000);
    return () => clearInterval(interval);
  }, []);

  const toggleAreaActive = async (id) => {
    // Optimistic update
    setRestrictedAreas((prev) =>
      prev.map((area) =>
        area.id === id ? { ...area, is_active: !area.is_active } : area
      )
    );

    try {
      await fetch(`http://127.0.0.1:8000/restricted-areas/${id}/toggle`, {
        method: "PATCH",
      });
    } catch (e) {
      console.error("Failed to toggle area", e);
    }
  };

  return (
    <div className="livefeed-panel">
      <div className="camera-list-section">
        <h2 className="section-title">Cameras</h2>
        {cameras.map((cam) => (
          <CameraCard
            key={cam.id}
            name={cam.name}
            isActive={selectedCamera === cam.id}
            onClick={() => setSelectedCamera(cam.id)}
          />
        ))}
      </div>

      <div className="video-section">
        <LiveFeedDisplay
          selectedCamera={selectedCamera}
          drawingTool={drawingTool}
          setDrawingTool={setDrawingTool}
          restrictedAreas={restrictedAreas}
        />
      </div>

      <div className="restricted-section">
        <h3 className="section-title">Restricted Areas</h3>

        <div className="tools-container">
          {!drawingTool ? (
            <>
              <button
                className="tool-btn polygon-btn"
                onClick={() => setDrawingTool("POLYGON")}
              >
                + Polygon
              </button>
              <button
                className="tool-btn ellipse-btn"
                onClick={() => setDrawingTool("FREEHAND")}
              >
                + Freehand
              </button>
            </>
          ) : (
            <button
              className="tool-btn cancel-btn"
              onClick={() => setDrawingTool(null)}
            >
              Cancel Drawing
            </button>
          )}
        </div>

        {restrictedAreas.map((area) => (
          <RestrictedAreaCard
            key={area.id}
            name={area.name}
            isActive={area.is_active} // This now controls color/visibility
            onClick={() => toggleAreaActive(area.id)}
            onDelete={async () => {
              try {
                await fetch(
                  `http://127.0.0.1:8000/restricted-areas/${area.id}`,
                  {
                    method: "DELETE",
                  }
                );
                setRestrictedAreas((prev) =>
                  prev.filter((a) => a.id !== area.id)
                );
              } catch {}
            }}
          />
        ))}
      </div>
    </div>
  );
}

export default LiveFeedPanel;
