import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useSocket } from '@components/SocketIOContext';

function Analytics() {
  const [activeTab, setActiveTab] = useState(null);
  const [visualizations, setVisualizations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const socket = useSocket();

  const visList = [
    { key: 'blockchain_transactions_by_store', type: 'iframe', url: '/analytics/blockchain_transactions_by_store.html', title: 'Blockchain Transactions by Type' },
    { key: '3d_spoilage_co2_sustainability', type: 'png', url: '/analytics/3d_spoilage_co2_sustainability.png', title: '3D Spoilage & CO2 Sustainability' },
    { key: 'carbon_emission_trajectory', type: 'iframe', url: '/analytics/carbon_emission_trajectory.html', title: 'Carbon Emission Trajectory' },
    { key: 'carbon_footprint_spoilage', type: 'png', url: '/analytics/carbon_footprint_spoilage.png', title: 'Carbon Footprint vs. Spoilage' },
    { key: 'correlation_heatmap', type: 'png', url: '/analytics/correlation_heatmap.png', title: 'Correlation Heatmap' },
    { key: 'feature_distribution', type: 'png', url: '/analytics/feature_distribution.png', title: 'Feature Distribution' },
    { key: 'global_sustainability', type: 'iframe', url: '/analytics/global_sustainability.html', title: 'Global Sustainability' },
    { key: 'neon_pulse_waste_reduction', type: 'png', url: '/analytics/neon_pulse_waste_reduction.png', title: 'Neon Pulse Waste Reduction' },
    { key: 'performance_metrics', type: 'csv', url: '/analytics/performance_metrics.csv', title: 'Performance Metrics' },
    { key: 'supply_chain_delay_network', type: 'iframe', url: '/analytics/supply_chain_delay_network.html', title: 'Supply Chain Delay Network' },
    { key: 'supply_chain_optimization_report', type: 'csv', url: '/analytics/supply_chain_optimization_report.csv', title: 'Supply Chain Optimization Report' },
    { key: 'sustainability_by_quarter', type: 'png', url: '/analytics/sustainability_by_quarter.png', title: 'Sustainability by Quarter' },
    { key: 'sustainability_efficiency_heatmap', type: 'png', url: '/analytics/sustainability_efficiency_heatmap.png', title: 'Sustainability Efficiency Heatmap' },
    { key: 'sustainability_metrics_report', type: 'csv', url: '/analytics/sustainability_metrics_report.csv', title: 'Sustainability Metrics Report' },
    { key: 'sustainability_trajectory', type: 'iframe', url: '/analytics/sustainability_trajectory.html', title: 'Sustainability Trajectory' },
    { key: 'top_supply_efficiency_stores', type: 'png', url: '/analytics/top_supply_efficiency_stores.png', title: 'Top Supply Efficiency Stores' },
    { key: 'top_sustainability_stores', type: 'png', url: '/analytics/top_sustainability_stores.png', title: 'Top Sustainability Stores' },
    { key: 'waste_reduction_bars', type: 'iframe', url: '/analytics/waste_reduction_bars.html', title: 'Waste Reduction Bars' },
    { key: 'weather_impact_globe', type: 'iframe', url: '/analytics/weather_impact_globe.html', title: 'Weather Impact Globe' }
  ];

  const fetchVisualizations = async () => {
    setLoading(true);
    setError(null);
    try {
      const available = [];
      for (const vis of visList) {
        try {
          const res = await fetch(vis.url);
          if (res.ok) available.push(vis);
        } catch {}
      }
      setVisualizations(available);
      setActiveTab(available[0]?.key || null);
    } catch (err) {
      setError('Failed to load visualizations');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchVisualizations();
    if (socket) {
      socket.on('analytics_update', (data) => {
        console.log('Received analytics update via WebSocket:', data);
        fetchVisualizations();
      });
      return () => socket.off('analytics_update');
    }
  }, [socket]);

  const handleCSVDownload = (url, title) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = `${title}.csv`;
    link.click();
  };

  if (loading) return <div className="text-center text-cyan-400 text-3xl animate-pulse" role="status">Loading Visualization...</div>;
  if (error || !activeTab) return <div className="text-center text-red-500 text-3xl" role="alert">{error || 'No visualizations available'}</div>;

  return (
    <div className="p-8" role="region" aria-label="Analytics Dashboard">
      <h1 className="text-5xl text-cyan-400 mb-8 text-center animate-glow">Analytics Dashboard</h1>
      <p className="text-2xl text-magenta-400 mb-6 text-center animate-pulse">
        Real-time insights for sustainability and efficiency, powered by AI and blockchain
      </p>
      <div className="flex flex-wrap justify-center mb-8 gap-4" role="tablist">
        {visualizations.map(({ key, title }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`px-6 py-3 rounded-lg text-lg transition duration-300 ${
              activeTab === key ? 'bg-cyan-400 text-tron-bg' : 'bg-tron-dark text-cyan-400 hover:bg-cyan-400 hover:text-tron-bg'
            } shadow-neon`}
            role="tab"
            aria-selected={activeTab === key}
            aria-controls={`panel-${key}`}
          >
            {title}
          </button>
        ))}
      </div>
      <div id={`panel-${activeTab}`} className="bg-tron-dark p-8 rounded-lg shadow-neon" role="tabpanel">
        <h2 className="text-3xl text-cyan-400 mb-6">{visualizations.find(v => v.key === activeTab)?.title}</h2>
        <p className="text-lg text-magenta-400 mb-4 italic">
          {activeTab.includes('sustainability') || activeTab.includes('blockchain') ? 'Driving a zero-waste future through data!' : 'Optimizing supply chains for a greener planet!'}
        </p>
        {visualizations.find(v => v.key === activeTab)?.type === 'png' ? (
          <img
            src={visualizations.find(v => v.key === activeTab)?.url}
            alt={visualizations.find(v => v.key === activeTab)?.title}
            className="w-full max-h-[700px] object-contain rounded-lg border-4 border-cyan-400 animate-glow"
            onError={(e) => (e.target.src = '/assets/logo.png')}
          />
        ) : visualizations.find(v => v.key === activeTab)?.type === 'iframe' ? (
          <iframe
            src={visualizations.find(v => v.key === activeTab)?.url}
            className="w-full h-[700px] rounded-lg border-4 border-cyan-400"
            title={visualizations.find(v => v.key === activeTab)?.title}
            aria-label={visualizations.find(v => v.key === activeTab)?.title}
          />
        ) : (
          <div className="text-center">
            <button
              onClick={() => handleCSVDownload(visualizations.find(v => v.key === activeTab)?.url, visualizations.find(v => v.key === activeTab)?.title)}
              className="p-4 bg-tron-dark text-cyan-400 rounded-lg hover:bg-cyan-400 hover:text-tron-bg transition duration-300 shadow-neon"
              aria-label={`Download ${visualizations.find(v => v.key === activeTab)?.title} CSV`}
            >
              Download {visualizations.find(v => v.key === activeTab)?.title} (CSV)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default Analytics;