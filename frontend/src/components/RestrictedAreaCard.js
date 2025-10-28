import React from "react";
import "./RestrictedAreaCard.css";

function RestrictedAreaCard({ name, isActive, onClick }) {
  return (
    <div
      className={`restricted-card ${isActive ? "active" : ""}`}
      onClick={onClick}
    >
      <div className="restricted-name">{name}</div>
    </div>
  );
}

export default RestrictedAreaCard;
