import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, NavLink } from 'react-router-dom';
import Dashboard from '@components/Dashboard';
import CustomerApp from '@components/CustomerApp';
import Analytics from '@components/Analytics';
import AgentChat from '@components/AgentChat';
import Heatmap from '@components/Heatmap';
import ThreeDGlobe from '@components/ThreeDGlobe';
import SustainabilityTracker from '@components/SustainabilityTracker';
import ReportGenerator from '@components/ReportGenerator';
import BlockchainAnalytics from '@components/BlockchainAnalytics';
import { SocketIOProvider } from '@components/SocketIOContext';
import Lottie from 'react-lottie';
import animationData from '@assets/eco-animation.json';

function App() {
  const [highContrast, setHighContrast] = useState(false);
  const [currentQuote, setCurrentQuote] = useState(0);
  const quotes = [
    "Reducing waste today ensures a greener tomorrow. - Sustainable Future",
    "Blockchain ensures transparency in every donation, powering trust. - Stellar Network",
    "AI predicts spoilage to save resources, ethically guiding sustainability. - Zero Waste",
    "Every action counts towards achieving SDG 12: Responsible Consumption. - United Nations",
    "Ethical AI drives decisions for a planet-first future. - Red Cross Mission"
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentQuote((prev) => (prev + 1) % quotes.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const lottieOptions = {
    loop: true,
    autoplay: true,
    animationData,
    rendererSettings: { preserveAspectRatio: 'xMidYMid slice' }
  };

  return (
    <SocketIOProvider>
      <Router>
        <div className={`min-h-screen ${highContrast ? 'high-contrast' : 'bg-tron-bg'} text-white font-mono relative overflow-hidden`} role="application">
          <div className="absolute inset-0 bg-[url('/assets/tron-bg.jpg')] opacity-10 bg-cover bg-center animate-pulse-slow"></div>
          <header className="p-8 bg-tron-dark text-center relative z-10" role="banner">
            <img src="/assets/logo.png" alt="Zero Waste Energy Optimizer Logo" className="w-16 h-16 mx-auto mb-4 animate-spin-slow" />
            <h1 className="text-6xl text-cyan-400 animate-glow">Zero Waste Energy Optimizer</h1>
            <p className="text-2xl text-magenta-400 mt-4 animate-pulse">AI & Blockchain for a Sustainable Future</p>
            <div className="mt-4 text-lg text-cyan-400 animate-pulse">{quotes[currentQuote]}</div>
            <button
              onClick={() => setHighContrast(!highContrast)}
              className="mt-4 p-2 bg-tron-dark text-cyan-400 rounded-lg hover:bg-cyan-400 hover:text-tron-bg transition duration-300"
              aria-label={highContrast ? "Disable high contrast mode" : "Enable high contrast mode"}
            >
              {highContrast ? 'Normal Mode' : 'High Contrast Mode'}
            </button>
          </header>
          <nav className="p-6 bg-tron-dark shadow-neon sticky top-0 z-20" role="navigation">
            <ul className="flex justify-center space-x-8">
              {[
                { path: "/", label: "Dashboard", icon: "/assets/store-icon.svg" },
                { path: "/customer", label: "Customer App", icon: "/assets/eco-icon.svg" },
                { path: "/analytics", label: "Analytics", icon: "/assets/eco-icon.svg" },
                { path: "/chat", label: "AI Chat", icon: "/assets/chat-icon.svg" },
                { path: "/heatmap", label: "Heatmap", icon: "/assets/eco-icon.svg" },
                { path: "/globe", label: "3D Globe", icon: "/assets/eco-icon.svg" },
                { path: "/sustainability", label: "Sustainability Tracker", icon: "/assets/eco-icon.svg" },
                { path: "/reports", label: "Reports", icon: "/assets/eco-icon.svg" },
                { path: "/blockchain-analytics", label: "Blockchain Analytics", icon: "/assets/eco-icon.svg" },
                { path: "/about", label: "About", icon: "/assets/eco-icon.svg" }
              ].map(({ path, label, icon }) => (
                <li key={path}>
                  <NavLink
                    to={path}
                    className={({ isActive }) =>
                      `flex items-center space-x-2 text-xl hover:text-cyan-400 transition duration-300 ${
                        isActive ? 'text-cyan-400 border-b-4 border-cyan-400' : ''
                      }`
                    }
                    aria-current={path === window.location.pathname ? "page" : undefined}
                  >
                    <img src={icon} alt={`${label} icon`} className="w-6 h-6" />
                    <span>{label}</span>
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          <main className="p-8 relative z-10" role="main">
            <Lottie options={lottieOptions} height={100} width={100} style={{ position: 'absolute', top: 0, right: 0, opacity: 0.5 }} />
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/customer" element={<CustomerApp />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/chat" element={<AgentChat />} />
              <Route path="/heatmap" element={<Heatmap />} />
              <Route path="/globe" element={<ThreeDGlobe />} />
              <Route path="/sustainability" element={<SustainabilityTracker />} />
              <Route path="/reports" element={<ReportGenerator />} />
              <Route path="/blockchain-analytics" element={<BlockchainAnalytics />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </main>
          <footer className="p-6 bg-tron-dark text-center text-cyan-400 shadow-neon relative z-10" role="contentinfo">
            Created by Adamya Sharma | Powered by AI and ML | Join the Zero Waste Revolution
          </footer>
        </div>
      </Router>
    </SocketIOProvider>
  );
}

function About() {
  return (
    <div className="p-8">
      <h1 className="text-5xl text-cyan-400 mb-6 text-center animate-glow">About Zero Waste Energy Optimizer</h1>
      <div className="bg-tron-dark p-8 rounded-lg shadow-neon">
        <h2 className="text-3xl text-magenta-400 mb-4">Our Mission</h2>
        <p className="text-lg text-white mb-4">
          Inspired by the futuristic vision of Tron, we leverage AI and blockchain to drive sustainability. Our platform predicts spoilage, optimizes inventory, and tracks donations transparently on the Stellar Testnet.
        </p>
        <h2 className="text-3xl text-magenta-400 mb-4">Why Blockchain?</h2>
        <p className="text-lg text-white mb-4">
          Blockchain ensures every donation and sustainability update is transparent and immutable, fostering trust in our zero-waste mission.
        </p>
        <h2 className="text-3xl text-magenta-400 mb-4">Ethical AI</h2>
        <p className="text-lg text-white mb-4">
          Our AI models prioritize ethical decision-making, reducing waste while respecting environmental and social impacts, aligned with SDG 12 and 13.
        </p>
        <h2 className="text-3xl text-magenta-400 mb-4">Join the Revolution</h2>
        <p className="text-lg text-white">
          Be part of a sustainable future. Every donation reduces CO2 emissions, equivalent to planting trees!
        </p>
      </div>
    </div>
  );
}

export default App;