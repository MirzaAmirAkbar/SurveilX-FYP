import React, { useState } from "react";
import { Shield } from "lucide-react"; // same icon used in Navbar
import "./login.css";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!email || !password) {
      alert("Please fill in both fields.");
      return;
    }

    try {
      const response = await fetch("http://127.0.0.1:8000/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      if (response.ok) {
        // Successful login
        const data = await response.json();
        console.log("Login successful:", data);
        window.location.href = "/dashboard"; // Navigate to the dashboard
      } else {
        // Handle failed login (e.g., 401 Invalid credentials)
        const errorData = await response.json();
        alert(`Login Failed: ${errorData.detail || "Check your email and password."}`);
      }
    } catch (error) {
      // Handle network errors (e.g., backend is down)
      console.error("Network error during login:", error);
      alert("Could not connect to the server. Is the backend running?");
    }
  };

  return (
    <div className="login">
      <div className="login-body">
        <div className="login-card">
          {/* 🛡️ SurveilX Header */}
          <div className="login-brand">
            <Shield className="login-icon" size={42} />
            <div className="login-title">
              <h1>SurveilX</h1>
              <p>Intelligent Surveillance System</p>
            </div>
          </div>

          {/* 🔐 Form */}
          <form className="login-form" onSubmit={handleSubmit}>
            <label>Email</label>
            <input
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />

            <label>Password</label>
            <input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />

            <button type="submit">Login</button>
          </form>

          <div className="login-footer">
            <a href="#">Forgot Password?</a>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Login;
