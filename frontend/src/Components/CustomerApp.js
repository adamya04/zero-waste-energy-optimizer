import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Lottie from 'react-lottie';
import animationData from '../assets/eco-animation.json';
import ecoIcon from '../assets/eco-icon.svg';

function CustomerApp() {
  const [inventory, setInventory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [donation, setDonation] = useState({ item: '', amount_lbs: '' });

  const itemImages = {
    banana: '/assets/banana.png',
    apple: '/assets/apple.png',
    berries: '/assets/berries.png',
    lettuce: '/assets/lettuce.png'
  };

  useEffect(() => {
    const fetchInventory = async () => {
      try {
        const res = await axios.get('/api/inventory/store_1');
        setInventory(res.data);
      } catch (err) {
        console.error('Failed to fetch inventory', err);
        setError('Failed to fetch inventory');
      } finally {
        setLoading(false);
      }
    };
    fetchInventory();
  }, []);

  const handleDonation = async () => {
    if (!donation.item || !donation.amount_lbs || isNaN(donation.amount_lbs) || parseFloat(donation.amount_lbs) <= 0) {
      alert('Please enter a valid item and positive amount');
      return;
    }
    try {
      const response = await axios.post('/api/blockchain/donate', {
        store_id: 'store_1',
        item: donation.item,
        amount_lbs: parseFloat(donation.amount_lbs)
      });
      alert(`Donation successful! Tx Hash: ${response.data.tx_hash}`);
      setDonation({ item: '', amount_lbs: '' });
      const res = await axios.get('/api/inventory/store_1');
      setInventory(res.data);
    } catch (err) {
      console.error('Donation failed', err);
      alert('Donation failed: ' + (err?.response?.data?.error || err.message));
    }
  };

  if (loading) return <div className="text-center text-cyan-400 text-3xl animate-pulse" role="status">Loading...</div>;
  if (error) return <div className="text-center text-red-500 text-3xl" role="alert">{error}</div>;

  const lottieOptions = {
    loop: true,
    autoplay: true,
    animationData,
    rendererSettings: { preserveAspectRatio: 'xMidYMid slice' }
  };

  return (
    <div className="p-8">
      <h1 className="text-4xl text-cyan-400 mb-6">Customer App</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <h2 className="text-2xl text-magenta-400 mb-2">Available Items</h2>
          <ul>
            {inventory.map((item) => (
              <li key={item.id || item.item} className="mb-2 flex items-center">
                <img
                  src={itemImages[item.item?.toLowerCase()] || '/assets/logo.png'}
                  alt={item.item}
                  className="w-8 h-8 mr-2"
                />
                <div>
                  <strong>{item.item || 'Unknown'}</strong> — {item.stock_lbs?.toFixed(2) || 0} lbs — {item.action || 'N/A'}
                </div>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h2 className="text-2xl text-magenta-400 mb-2">Donate</h2>
          <input
            className="p-2 rounded w-full mb-2 bg-tron-dark text-white"
            placeholder="Item"
            value={donation.item}
            onChange={(e) => setDonation({ ...donation, item: e.target.value })}
          />
          <input
            className="p-2 rounded w-full mb-2 bg-tron-dark text-white"
            placeholder="Amount (lbs)"
            value={donation.amount_lbs}
            onChange={(e) => setDonation({ ...donation, amount_lbs: e.target.value })}
            type="number"
            min="0"
            step="0.01"
          />
          <button onClick={handleDonation} className="px-4 py-2 bg-cyan-400 rounded text-tron-bg">Donate</button>
        </div>
      </div>
      <p className="text-lg text-magenta-400 flex items-center mt-6">
        <img src={ecoIcon} alt="Eco Icon" className="w-6 h-6 mr-2" />
        Your donations reduce waste and support sustainability!
      </p>
      <Lottie options={lottieOptions} height={100} width={100} style={{ marginTop: '20px' }} />
    </div>
  );
}

export default CustomerApp;