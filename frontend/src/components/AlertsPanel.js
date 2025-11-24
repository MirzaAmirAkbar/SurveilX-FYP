import React, { useEffect, useState } from "react";
import "./AlertsPanel.css";

function AlertsPanel() {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    let timer = null;
    const fetchAlerts = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/alerts");
        const data = await res.json();
        if (Array.isArray(data.alerts)) {
          setAlerts(data.alerts);
        }
      } catch (e) {
        // ignore network errors for now
      }
    };
    fetchAlerts();
    timer = setInterval(fetchAlerts, 2000);
    return () => timer && clearInterval(timer);
  }, []);

  return (
    <div className="alerts-panel">
      <h2 className="section-title">Alerts</h2>
      <div className="alerts-list">
        {alerts.length === 0 && (
          <div className="alert-empty">No alerts</div>
        )}
        {alerts.map((a, idx) => (
          <div key={`${a.personId}-${a.timestamp}-${idx}`} className="alert-card">
            <img
              className="alert-image"
              src={`data:image/jpeg;base64,${a.imageB64}`}
              alt={`Person ${a.personId}`}
            />
            <div className="alert-meta">
              <div className="alert-title">ID {a.personId}</div>
              <div className="alert-zone">Breach: {a.zone}</div>
              {/* Optional: timestamp formatting */}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default AlertsPanel;
