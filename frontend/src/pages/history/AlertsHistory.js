import React, { useState, useEffect} from "react";
import Navbar from "../../components/navbar.js";
import "./AlertsHistory.css";


function AlertsHistory() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [availableCameras, setAvailableCameras] = useState([]);
  const [alerts, setAlerts] = useState([]);
const [isLoading, setIsLoading] = useState(false);

  // Filter States
  const [filterType, setFilterType] = useState("All");
  const [filterCamera, setFilterCamera] = useState("All");
  const [filterRisk, setFilterRisk] = useState("All");
  const [filterDateFrom, setFilterDateFrom] = useState("");
  const [filterDateTo, setFilterDateTo] = useState("");

  // Fetch cameras from backend on load
  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/videos");
        const data = await res.json();
        if (data.videos && data.videos.length > 0) {
          setAvailableCameras(data.videos);
        }
      } catch (e) { console.warn("Fetch cameras error. Is backend running?", e); }
    };
    fetchCameras();
  }, []);
  
  // Maps types to their risk level
  const getRiskLevel = (type) => {
    switch (type?.toLowerCase()) {
      case 'breach':
      case 'weapon_detected':
      case 'abandoned_object':
      case 'shoplifting': return 'red';
      case 'person_with_bag':
      case 'loitering': return 'yellow';
      default: return 'red';
    }
  };

  const getAlertTheme = (type) => {
    const risk = getRiskLevel(type);
    if (risk === 'red') return { border: 'border-red', text: 'text-red' };
    if (risk === 'yellow') return { border: 'border-yellow', text: 'text-yellow' };
    return { border: 'border-red', text: 'text-red' };
  };

  const formatTime = (ts) => {
    const date = new Date(ts);
    return date.toLocaleString([], { dateStyle: 'short', timeStyle: 'short' });
  };

  // 2. Replace the `filteredAlerts` useMemo with this useEffect
    useEffect(() => {
    const fetchHistory = async () => {
        setIsLoading(true);
        
        try {
            // Build the query string dynamically based on active filters
            const params = new URLSearchParams();
            
            if (filterType !== "All") params.append("type", filterType);
            if (filterCamera !== "All") params.append("camera", filterCamera);
            if (filterRisk !== "All") params.append("risk", filterRisk);
            
            if (filterDateFrom) {
                params.append("start_date", new Date(filterDateFrom).toISOString());
            }
            
            if (filterDateTo) {
                const endDate = new Date(filterDateTo);
                endDate.setHours(23, 59, 59, 999);
                params.append("end_date", endDate.toISOString());
            }

            const res = await fetch(`http://127.0.0.1:8000/api/alerts/history?${params.toString()}`);
            const data = await res.json();
            
            if (data.alerts) {
                // Directly set the alerts since we no longer do client-side Person ID filtering
                setAlerts(data.alerts);
            }
        } catch (e) {
            console.error("Failed to fetch history:", e);
        } finally {
            setIsLoading(false);
        }
    };

    fetchHistory();
  // Removed filterPerson from the dependency array below
  }, [filterType, filterCamera, filterRisk, filterDateFrom, filterDateTo]);

  const clearFilters = () => {
    setFilterType("All");
    setFilterCamera("All");
    setFilterRisk("All");
    setFilterDateFrom("");
    setFilterDateTo("");
  };

  return (
    <div className="history-page">
      <Navbar />
      
      <div className="history-container">
        
        {/* --- TOP: FILTER BAR --- */}
        <div className="filter-bar">
          
          {/* Date Range */}
          <div className="filter-group">
            <label>Date From</label>
            <input 
              type="date" 
              value={filterDateFrom} 
              onChange={(e) => setFilterDateFrom(e.target.value)}
            />
          </div>
          <div className="filter-group">
            <label>Date To</label>
            <input 
              type="date" 
              value={filterDateTo} 
              onChange={(e) => setFilterDateTo(e.target.value)}
            />
          </div>

          <div className="filter-group">
            <label>Risk Level</label>
            <select value={filterRisk} onChange={(e) => setFilterRisk(e.target.value)}>
              <option value="All">All Risks</option>
              <option value="red">High (Red)</option>
              <option value="yellow">Low (Yellow)</option>
            </select>
          </div>

          <div className="filter-group">
            <label>Event Type</label>
            <select value={filterType} onChange={(e) => setFilterType(e.target.value)}>
              <option value="All">All Types</option>
              <option value="breach">Breach</option>
              <option value="weapon_detected">Weapon Detected</option>
              <option value="shoplifting">Shoplifting</option>
              <option value="abandoned_object">Abandoned Object</option>
              <option value="loitering">Loitering</option>
              <option value="person_with_bag">Carrying Bag</option>
            </select>
          </div>

          <div className="filter-group">
            <label>Camera</label>
            <select value={filterCamera} onChange={(e) => setFilterCamera(e.target.value)}>
              <option value="All">All Cameras</option>
              {availableCameras.map(cam => (
                <option key={cam.id} value={cam.id}>{cam.name}</option>
              ))}
            </select>
          </div>

          <div className="filter-actions">
            <button className="clear-btn" onClick={clearFilters}>Clear Filters</button>
          </div>
        </div>

        {/* --- MAIN: GRID VIEW --- */}
        <div className="history-grid">
          {alerts.length === 0 ? (
             <div className="no-results">No alerts match your criteria.</div>
          ) : (
            alerts.map(a => {
              const theme = getAlertTheme(a.type);
              const imageSrc = `data:image/jpeg;base64,${a.imageB64}`;

              return (
                <div key={a.id} className={`grid-card ${theme.border}`}>
                  <div className="card-image-container" onClick={() => setSelectedImage(imageSrc)}>
                    <img src={imageSrc} alt="Alert Thumbnail" className="card-image" />
                    <div className="image-overlay-hint">Click to enlarge</div>
                  </div>
                  
                  <div className="card-details">
                    <div className={`card-type ${theme.text}`}>
                      {a.type.toUpperCase().replace(/_/g, ' ')}
                    </div>
                    <div className="card-row"><strong>Time:</strong> <span>{formatTime(a.timestamp)}</span></div>
                    <div className="card-row"><strong>Camera:</strong> <span>{a.camera}</span></div>
                    <div className="card-row"><strong>Zone:</strong> <span>{a.zone || 'N/A'}</span></div>
                  </div>
                </div>
              );
            })
          )}
        </div>

      </div>

      {/* --- IMAGE MODAL --- */}
      {selectedImage && (
        <div className="image-modal-overlay" onClick={() => setSelectedImage(null)}>
          <img 
            className="full-image" 
            src={selectedImage} 
            alt="Full Alert" 
            onClick={(e) => e.stopPropagation()} 
          />
        </div>
      )}
    </div>
  );
}

export default AlertsHistory;