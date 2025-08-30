import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useSocket } from '@components/SocketIOContext';
import Lottie from 'react-lottie';
import animationData from '../assets/eco-animation.json';

function Dashboard() {
  const [inventory, setInventory] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const socket = useSocket();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [inventoryRes, analyticsRes] = await Promise.all([
          axios.get('/api/inventory/store_1'),
          axios.get('/api/analytics/store_1')
        ]);
        setInventory(inventoryRes.data || []);
        setMetrics(analyticsRes.data || {});
      } catch (err) {
        console.error('Dashboard fetch error', err);
        setError('Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    if (socket) {
      socket.on('inventory_update', (data) => {
        if (data && !data.error) {
          setInventory(data);
        } else if (data && data.error) {
          console.error('Socket update error:', data.error);
        }
      });
      return () => socket.off('inventory_update');
    }
  }, [socket]);

  if (loading) return <div className="text-center text-cyan-400 text-3xl animate-pulse">Loading dashboard...</div>;
  if (error) return <div className="text-center text-red-500 text-3xl">{error}</div>;

  const lottieOptions = {
    loop: true,
    autoplay: true,
    animationData,
    rendererSettings: { preserveAspectRatio: 'xMidYMid slice' }
  };

  return (
    <div className="p-8">
      <h1 className="text-5xl text-cyan-400 mb-6">Manager Dashboard</h1>
      <div className="mb-8">
        <h2 className="text-2xl text-magenta-400">Key Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div className="p-4 bg-tron-dark rounded shadow-neon">
            <div className="text-3xl">{(metrics.sustainability_score_avg || 0).toFixed(2)}</div>
            <div className="text-sm">Avg Sustainability</div>
          </div>
          <div className="p-4 bg-tron-dark rounded shadow-neon">
            <div className="text-3xl">{(metrics.donation_potential || 0).toFixed(2)}</div>
            <div className="text-sm">Donation Potential (lbs)</div>
          </div>
          <div className="p-4 bg-tron-dark rounded shadow-neon">
            <div className="text-3xl">{(metrics.co2_reduction || 0).toFixed(2)}</div>
            <div className="text-sm">Estimated CO2 Reduction</div>
          </div>
        </div>
      </div>
      <div className="bg-tron-dark p-4 rounded shadow-neon">
        <h2 className="text-2xl mb-4">Inventory</h2>
        <table className="w-full">
          <thead>
            <tr>
              <th className="text-left p-4">Item</th>
              <th className="p-4">Stock (lbs)</th>
              <th className="p-4">Expiry</th>
              <th className="p-4">Predicted Risk</th>
              <th className="p-4">CO2 Factor</th>
              <th className="p-4">Packaging Waste</th>
              <th className="p-4">Action</th>
            </tr>
          </thead>
          <tbody>
            {inventory.map(item => (
              <tr key={item.id || item.item || Math.random()}>
                <td className="p-4">{item.item || 'Unknown'}</td>
                <td className="p-4">{(item.stock_lbs || 0).toFixed(2)}</td>
                <td className="p-4">{item.expiry_date ? new Date(item.expiry_date).toLocaleDateString() : 'N/A'}</td>
                <td className="p-4">{(item.predicted_spoilage_risk || 0).toFixed(2)}</td>
                <td className="p-4">{(item.co2_emission_factor || 0).toFixed(2)}</td>
                <td className="p-4">{(item.packaging_waste || 0).toFixed(2)}</td>
                <td className="p-4 text-cyan-400">{item.action || 'N/A'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Lottie options={lottieOptions} height={100} width={100} style={{ marginTop: '20px' }} />
    </div>
  );
}

export default Dashboard;