import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

function BlockchainAnalytics() {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchTransactions = async () => {
      try {
        // Use relative path so Vite proxy routes correctly
        const response = await axios.get('/api/blockchain/transactions/store_1');
        setTransactions(response.data || []);
      } catch (err) {
        console.error('Failed to fetch blockchain transactions:', err);
        setError('Failed to fetch blockchain transactions');
      } finally {
        setLoading(false);
      }
    };
    fetchTransactions();
  }, []);

  if (loading) return <div className="text-cyan-400 text-center p-8">Loading blockchain analytics…</div>;
  if (error) return <div className="text-red-500 text-center p-8">{error}</div>;

  const donationCount = transactions.filter(t => t.transaction_type === 'donation').length;
  const sustainabilityCount = transactions.filter(t => t.transaction_type === 'sustainability').length;
  const totalCO2Reduction = transactions.reduce((acc, t) => acc + (t.co2_emission_factor || 0) * (t.stock_lbs || 0), 0) * -1;

  return (
    <div className="p-8">
      <h1 className="text-4xl text-cyan-400 mb-4">Blockchain Analytics</h1>
      <p className="text-lg text-magenta-400 mb-6">
        Transactions are recorded on the Stellar testnet. Donations and sustainability score updates are stored in the Inventory table with tx hashes.
      </p>

      <div className="bg-tron-dark p-8 rounded-lg shadow-neon mb-8">
        <h2 className="text-3xl text-cyan-400 mb-6">Transaction Summary</h2>
        <p className="text-lg text-white">Donation Transactions: {donationCount}</p>
        <p className="text-lg text-white">Sustainability Updates: {sustainabilityCount}</p>
        <p className="text-lg text-white">Total CO2 Reduction: {isNaN(totalCO2Reduction) ? '0.00' : totalCO2Reduction.toFixed(2)} tons</p>
      </div>

      <div className="bg-tron-dark p-8 rounded-lg shadow-neon">
        <h2 className="text-3xl text-cyan-400 mb-6">Transaction Chart</h2>
        <Plot
          data={[
            {
              x: ['Donations', 'Sustainability Updates'],
              y: [donationCount, sustainabilityCount],
              type: 'bar',
              marker: { color: ['#00FFFF', '#FF00FF'] }
            }
          ]}
          layout={{
            title: 'Transactions by Type',
            paper_bgcolor: '#1A1A2E',
            plot_bgcolor: '#1A1A2E',
            font: { color: '#00FFFF' }
          }}
          style={{ width: '100%', height: '400px' }}
        />
      </div>

      <div className="bg-tron-dark p-6 rounded-lg shadow-neon mt-8">
        <h2 className="text-2xl text-magenta-400 mb-4">Recent Transactions</h2>
        <div className="space-y-4">
          {transactions.slice(0, 10).map((tx, idx) => (
            <div key={idx} className="p-4 bg-[#0f1622] rounded">
              <div className="flex justify-between">
                <div>
                  <div className="font-bold">{tx.item}</div>
                  <div className="text-sm">Type: {tx.transaction_type}</div>
                </div>
                <div className="text-right">
                  <div className="text-sm">Amount: {tx.stock_lbs}</div>
                  <div className="text-xs">{new Date(tx.date).toLocaleString()}</div>
                </div>
              </div>
              <div className="mt-2 text-xs text-cyan-400">Tx: {tx.tx_hash}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default BlockchainAnalytics;
