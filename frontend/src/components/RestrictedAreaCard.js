import React from "react";
import "./RestrictedAreaCard.css";

function RestrictedAreaCard({ name, isActive, onClick, onDelete }) {
  return (
    <div
      className={`restricted-card ${isActive ? "active" : ""}`}
      onClick={onClick}
    >
      <div className="restricted-name">{name}</div>
      <button
        className="delete-area-btn"
        onClick={e => {
          e.stopPropagation();
          onDelete && onDelete();
        }}
        title="Remove area"
      >✕</button>
    </div>
  );
}

export default RestrictedAreaCard;
