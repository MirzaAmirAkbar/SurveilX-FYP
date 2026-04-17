import React, { useEffect, useState } from "react";
import "./AlertsPanel.css";

function AlertsPanel() {
  const [alerts, setAlerts] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);

  useEffect(() => {
    let timer = null;
    const fetchAlerts = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/alerts");
        const data = await res.json();
        if (Array.isArray(data.alerts)) {
          setAlerts(data.alerts);
        }
      } catch (e) { /* ignore network errors */ }
    };
    fetchAlerts();
    timer = setInterval(fetchAlerts, 2000);
    return () => timer && clearInterval(timer);
  }, []);

  const getAlertTheme = (type) => {
    switch (type?.toLowerCase()) {
      case 'breach': return { border: 'red-border', text: 'text-red' };
      case 'running': return { border: 'orange-border', text: 'text-orange' };
      case 'loitering': return { border: 'yellow-border', text: 'text-yellow' };
      default: return { border: 'red-border', text: 'text-red' };
    }
  };

  // Helper to format timestamp (assumes Unix or ISO string)
  const formatTime = (ts) => {
    const date = new Date(ts);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  return (
    <div className="alerts-panel">
      <h2 className="section-title">Alerts</h2>
      <div className="alerts-list">
        {alerts.length === 0 && <div className="alert-empty">No alerts</div>}
        
        {alerts.map((a, idx) => {
          const theme = getAlertTheme(a.type);
          const imageSrc = `data:image/jpeg;base64,${a.imageB64}`;

          return (
            <div key={`${a.personId}-${a.timestamp}-${idx}`} className={`alert-card ${theme.border}`}>
              <img className="alert-image" src={imageSrc} onClick={() => setSelectedImage(imageSrc)} />
              
              <div className="alert-meta">
                <div className={`alert-heading ${theme.text}`}>
                  {a.type ? a.type.toUpperCase().replace('_', ' ') : "BREACH"}
                </div>
                
                {/* Corrected: Showing Camera Name (Filename) */}
                <div className="alert-detail"><strong>Camera:</strong> {a.camera || "Main Entry"}</div>
                
                {/* Added: Conditional Zone Display for Breach and Loitering */}
                {(a.type === 'breach' || a.type === 'loitering') && (
                  <div className="alert-detail"><strong>Zone:</strong> {a.zone}</div>
                )}

                <div className="alert-detail"><strong>Person ID:</strong> {a.personId}</div>
                <div className="alert-detail"><strong>Time:</strong> {formatTime(a.timestamp)}</div>
              </div>
            </div>
          );
        })}
      </div>
      {/* ... modal logic */}
    </div>
  );
}

export default AlertsPanel;