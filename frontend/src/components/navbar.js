import React from "react";
import { useLocation, Link } from "react-router-dom"; // Added for routing
import "./navbar.css";
import { Shield } from "lucide-react";

function Navbar() {
  const location = useLocation(); // Gets current route to highlight active link

  return (
    <header className="navbar">
      <div className="navbar-left">
        <Shield className="navbar-icon" size={40} />
        <div className="navbar-title">
          <h1>SurveilX</h1>
          <p>Intelligent Surveillance System</p>
        </div>
      </div>

      {/* --- NEW: Center Navigation --- */}
      <div className="navbar-center">
        <Link 
          to="/dashboard" 
          className={`nav-link ${location.pathname === "/dashboard" ? "active" : ""}`}
        >
          Live Dashboard
        </Link>
        <Link 
          to="/history" 
          className={`nav-link ${location.pathname === "/history" ? "active" : ""}`}
        >
          Alerts History
        </Link>
      </div>

      <button className="logout-btn">Logout</button>
    </header>
  );
}

export default Navbar;