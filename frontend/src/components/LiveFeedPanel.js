import React, { useState, useEffect } from "react";
import CameraCard from "./CameraCard.js";
import RestrictedAreaCard from "./RestrictedAreaCard.js";
import LiveFeedDisplay from "./LiveFeedDisplay.js";
import "./LiveFeedPanel.css";

function LiveFeedPanel() {
  const [selectedCamera, setSelectedCamera] = useState(1);
  
  // Drawing State
  const [drawingTool, setDrawingTool] = useState(null); // 'POLYGON', 'FREEHAND', or null
  const [drawingType, setDrawingType] = useState(null); // 'RESTRICTED' or 'LOITERING'

  // Data State
  const [restrictedAreas, setRestrictedAreas] = useState([]);
  const [loiteringAreas, setLoiteringAreas] = useState([]);

  const cameras = [
    { id: 1, name: "Aisle 1" },
    { id: 2, name: "Aisle 2" },
    { id: 3, name: "Aisle 3" },
  ];

  // --- FETCH DATA ---
  const fetchRestrictedAreas = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/restricted-areas/");
      const data = await res.json();
      if (data.areas) {
        setRestrictedAreas(data.areas.map(a => ({ ...a, is_active: a.is_active ?? true })));
      }
    } catch (e) { console.warn("Fetch restricted error", e); }
  };

  const fetchLoiteringAreas = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/loitering-areas/"); // Assumed API
      const data = await res.json();
      if (data.areas) {
        setLoiteringAreas(data.areas.map(a => ({ ...a, is_active: a.is_active ?? true })));
      }
    } catch (e) { console.warn("Fetch loitering error", e); }
  };

  const refreshAll = () => {
    fetchRestrictedAreas();
    fetchLoiteringAreas();
  };

  useEffect(() => {
    refreshAll();
  }, []);

  // --- TOGGLE & DELETE HANDLERS ---
  const toggleArea = (id, type) => {
    if (type === 'RESTRICTED') {
      setRestrictedAreas(prev => prev.map(a => a.id === id ? { ...a, is_active: !a.is_active } : a));
    } else {
      setLoiteringAreas(prev => prev.map(a => a.id === id ? { ...a, is_active: !a.is_active } : a));
    }
  };

  const deleteArea = async (id, type) => {
    const endpoint = type === 'RESTRICTED' ? 'restricted-areas' : 'loitering-areas';
    try {
      await fetch(`http://127.0.0.1:8000/${endpoint}/${id}`, { method: "DELETE" });
      if (type === 'RESTRICTED') {
        setRestrictedAreas(prev => prev.filter(a => a.id !== id));
      } else {
        setLoiteringAreas(prev => prev.filter(a => a.id !== id));
      }
    } catch (e) { console.error("Delete error", e); }
  };

  // --- DRAWING CONTROLS ---
  const startDrawing = (tool, type) => {
    setDrawingTool(tool);
    setDrawingType(type);
  };

  const cancelDrawing = () => {
    setDrawingTool(null);
    setDrawingType(null);
  };

  return (
    <div className="livefeed-panel">
      {/* LEFT: Cameras */}
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

      {/* CENTER: Video */}
      <div className="video-section">
        <LiveFeedDisplay
          selectedCamera={selectedCamera}
          drawingTool={drawingTool}
          setDrawingTool={setDrawingTool} 
          drawingType={drawingType}     // Pass the type down
          setDrawingType={setDrawingType}
          restrictedAreas={restrictedAreas}
          loiteringAreas={loiteringAreas}
          onRefreshAreas={refreshAll}
        />
      </div>

      {/* RIGHT: Split Panel */}
      <div className="sidebar-right">
        
        {/* TOP: Restricted (RED) */}
        <div className="panel-half restricted-half">
          <h3 className="section-title red-title">Restricted Areas</h3>
          
          <div className="tools-container">
            {drawingType === 'RESTRICTED' && drawingTool ? (
              <button className="tool-btn cancel-btn" onClick={cancelDrawing}>Cancel Drawing</button>
            ) : (
              <div className="btn-group">
                <button 
                  className="tool-btn btn-red-outline" 
                  disabled={drawingTool !== null}
                  onClick={() => startDrawing("POLYGON", "RESTRICTED")}
                >Polygon</button>
                <button 
                  className="tool-btn btn-red-outline" 
                  disabled={drawingTool !== null}
                  onClick={() => startDrawing("FREEHAND", "RESTRICTED")}
                >Freehand</button>
              </div>
            )}
          </div>

          <div className="scroll-list">
            {restrictedAreas.map((area) => (
              <RestrictedAreaCard
                key={area.id}
                name={area.name}
                isActive={area.is_active}
                onClick={() => toggleArea(area.id, 'RESTRICTED')}
                onDelete={() => deleteArea(area.id, 'RESTRICTED')}
                // Optional: Pass a "theme" prop to Card if you want the card itself to be red
              />
            ))}
          </div>
        </div>

        {/* BOTTOM: Loitering (BLUE) */}
        <div className="panel-half loitering-half">
          <h3 className="section-title blue-title">Loitering Areas</h3>

          <div className="tools-container">
            {drawingType === 'LOITERING' && drawingTool ? (
              <button className="tool-btn cancel-btn" onClick={cancelDrawing}>Cancel Drawing</button>
            ) : (
              <div className="btn-group">
                <button 
                  className="tool-btn btn-blue-outline" 
                  disabled={drawingTool !== null}
                  onClick={() => startDrawing("POLYGON", "LOITERING")}
                >Polygon</button>
                <button 
                  className="tool-btn btn-blue-outline" 
                  disabled={drawingTool !== null}
                  onClick={() => startDrawing("FREEHAND", "LOITERING")}
                >Freehand</button>
              </div>
            )}
          </div>

          <div className="scroll-list">
            {loiteringAreas.map((area) => (
              <RestrictedAreaCard
                key={area.id}
                name={area.name}
                isActive={area.is_active}
                variant="blue"
                onClick={() => toggleArea(area.id, 'LOITERING')}
                onDelete={() => deleteArea(area.id, 'LOITERING')}
              />
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}

export default LiveFeedPanel;