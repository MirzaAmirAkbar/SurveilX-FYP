import React, {useState} from "react";
import CameraCard from "./CameraCard.js"
import RestrictedAreaCard from "./RestrictedAreaCard.js";
import LiveFeedDisplay from "./LiveFeedDisplay.js";
import "./LiveFeedPanel.css";

function LiveFeedPanel() {
    const [selectedCamera, setSelectedCamera] = useState(1);

    const [isDrawingMode, setIsDrawingMode] = useState(false);

    const cameras = [
        { id: 1, name: "Aisle 1"},
        { id: 2, name: "Aisle 2"},
        { id: 3, name: "Aisle 3"},
    ];

    const [restrictedAreas, setRestrictedAreas] = useState([
      
    ]);


    const [visibleAreas, setVisibleAreas] = useState([]);



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
          isDrawingMode={isDrawingMode}
          visibleAreas={visibleAreas}
          restrictedAreas={restrictedAreas}
          setRestrictedAreas={setRestrictedAreas}
        />

      </div>
      <div className="restricted-section">
          <h3 className="section-title">Restricted Areas</h3>

        

        <button className="add-area-btn" onClick={() => setIsDrawingMode(!isDrawingMode)}>
          {isDrawingMode ? "Cancel" : "Add Restricted Area"}
        </button>


        {restrictedAreas.map((area) => (
          <RestrictedAreaCard
            key={area.id}
            name={area.name}
            isActive={visibleAreas.includes(area.id)} // ✅ highlight if visible
            onClick={() => {
              setVisibleAreas(prev =>
                prev.includes(area.id)
                  ? prev.filter(id => id !== area.id) // toggle off
                  : [...prev, area.id]                // toggle on
              );
            }}
            onDelete={async () => {
              try {
                await fetch(`http://127.0.0.1:8000/restricted-areas/${area.id}`, {
                  method: "DELETE",
                });
              } catch {}
              setRestrictedAreas(prev => prev.filter(a => a.id !== area.id));
              setVisibleAreas(prev => prev.filter(id => id !== area.id));
            }}
          />

        ))}

      </div>

    </div>
  );
}

export default LiveFeedPanel;
