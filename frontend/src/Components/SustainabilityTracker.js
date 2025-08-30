import React, { useEffect, useState } from 'react';
import axios from 'axios';

function SustainabilityTracker() {
  const [metrics, setMetrics] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetch = async () => {
      try {
        const res = await axios.get('/api/analytics/store_1');
        setMetrics(res.data || {});
      } catch (err) {
        console.error("Sustainability Tracker Error:", err);
        setError('Failed to load sustainability metrics');
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, []);

  if (loading) return <div className="text-cyan-400 text-center p-8">Loading metrics…</div>;
  if (error) return <div className="text-red-500 text-center p-8">{error}</div>;

  return (
    <div className="p-6">
      <h1 className="text-4xl tron-neon mb-6 text-center">Sustainability Tracker</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-tron-dark p-6 rounded-lg shadow-neon">
          <h2 className="text-2xl tron-purple mb-4">Key Performance</h2>
          <p className="text-white">Donation Potential: {metrics.donation_potential?.toFixed(2) || 0} lbs</p>
          <p className="text-white">Sustainability Score: {metrics.sustainability_score_avg?.toFixed(2) || 0}</p>
          <p className="text-white">SDG Alignment: {metrics.sdg_alignment_avg?.toFixed(2) || 0}</p>
        </div>
        <div className="bg-tron-dark p-6 rounded-lg shadow-neon">
          <h2 className="text-2xl tron-purple mb-4">SDG Goals</h2>
          <p className="text-white">Goal 12 (Responsible Consumption): {(metrics.sdg_alignment_avg || 0).toFixed(2)}</p>
          <p className="text-white">Goal 13 (Climate Action): {(metrics.waste_reduction_trend || 0).toFixed(2)}%</p>
        </div>
      </div>
    </div>
  );
}

export default SustainabilityTracker;
