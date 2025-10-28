import { Route, Routes, Navigate } from "react-router-dom";
import Dashboard from "./pages/dashboard/dashboard.js";

function App() {
  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </div>
  );
}

export default App;
