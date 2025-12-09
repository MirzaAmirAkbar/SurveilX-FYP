import React from "react";
import "./RestrictedAreaCard.css";

// Added 'variant' prop with a default value of 'red'
function RestrictedAreaCard({ name, isActive, onClick, onDelete, variant = 'red' }) {
  return (
    <div
      className={`restricted-card ${isActive ? "active" : ""} variant-${variant}`}
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