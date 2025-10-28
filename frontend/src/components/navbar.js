import React from "react";
import "./navbar.css";
import { Shield } from "lucide-react"; // clean icon library

function Navbar() {
  return (
    <header className="navbar">
      <div className="navbar-left">
        <Shield className="navbar-icon" size={40} />
        <div className="navbar-title">
          <h1>SurveilX</h1>
          <p>Intelligent Surveillance System</p>
        </div>
      </div>

      <button className="logout-btn">Logout</button>
    </header>
  );
}

export default Navbar;
