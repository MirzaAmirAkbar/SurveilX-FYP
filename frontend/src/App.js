import { Route, Routes, Navigate } from "react-router-dom";
import Dashboard from "./pages/dashboard/dashboard.js";
import Login from "./pages/login/login.js"
import AlertsHistory from "./pages/history/AlertsHistory.js"; // IMPORT NEW PAGE

function App() {
  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<Navigate to="/login" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/login" element={<Login />} />
        <Route path="/history" element={<AlertsHistory />} /> {/* NEW ROUTE */}
      </Routes>
    </div>
  );
}

export default App;