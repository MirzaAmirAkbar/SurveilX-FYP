import React from "react";
import "./CameraCard.css";

function CameraCard({ name, isActive, onClick }) {
  return (
    <div
      className={`camera-card ${isActive ? "active" : ""}`}
      onClick={onClick}
    >
      <div className="camera-name">{name}</div>
    </div>
  );
}

export default CameraCard;
