import React from "react";
import Navbar from "../../components/navbar.js";
import LiveFeedPanel from "../../components/LiveFeedPanel";
import AlertsPanel from "../../components/AlertsPanel";
import "./dashboard.css";

function Dashboard() {
  return (
    <div className="dashboard">
      <Navbar />
      <div className="dashboard-body">
        <LiveFeedPanel />
        <AlertsPanel />
      </div>
    </div>
  );
}

export default Dashboard;
