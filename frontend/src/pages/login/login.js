import React, { useState } from "react";
import { Shield } from "lucide-react"; // same icon used in Navbar
import "./login.css";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();

    if (email && password) {
      console.log("Logging in with:", { email, password });
      window.location.href = "/dashboard"; // or useNavigate if router is set up
    } else {
      alert("Please fill in both fields.");
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
            <span>Don’t have an account?</span> <a href="#">Sign up</a>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Login;
