import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

function Heatmap() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHeatmapData = async () => {
      try {
        const response = await axios.get('/api/analytics/store_1');
        const df = response.data.records || [];
        if (!df.length) {
          setError('No data available for heatmap');
          setLoading(false);
          return;
        }
        const numericFields = ['temperature_c', 'humidity_percent', 'pressure_mb', 'wind_speed_mps', 'stock_lbs', 'sales_lbs_daily', 'spoilage_rate', 'co2_emission_factor', 'supply_chain_delay', 'packaging_waste', 'transport_distance_km'];
        const dataMatrix = df.map(item => numericFields.map(field => item[field] || 0));
        setData({ matrix: dataMatrix, fields: numericFields });
      } catch (err) {
        console.error('Heatmap fetch error:', err);
        setError('Failed to load heatmap data');
      } finally {
        setLoading(false);
      }
    };
    fetchHeatmapData();
  }, []);

  if (loading) return <div className="text-cyan-400 text-center p-8">Loading heatmap...</div>;
  if (error) return <div className="text-red-500 text-center p-8">{error}</div>;
  if (!data || !data.matrix || data.matrix.length === 0) return <div className="p-8 text-white">No data available for heatmap</div>;

  const fields = data.fields;
  const matrix = (() => {
    const cols = fields.length;
    const transposed = Array.from({ length: cols }, (_, j) => data.matrix.map(row => row[j]));
    const corr = Array.from({ length: cols }, () => Array(cols).fill(0));
    const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    const std = arr => Math.sqrt(arr.map(v => Math.pow(v - mean(arr), 2)).reduce((a, b) => a + b, 0) / arr.length || 1);
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < cols; j++) {
        const xi = transposed[i];
        const xj = transposed[j];
        const mi = mean(xi), mj = mean(xj);
        const numerator = xi.map((v, k) => (v - mi) * (xj[k] - mj)).reduce((a, b) => a + b, 0);
        const denom = (xi.length * std(xi) * std(xj)) || 1;
        corr[i][j] = denom ? numerator / denom : 0;
      }
    }
    return corr;
  })();

  return (
    <div className="p-8">
      <h1 className="text-4xl text-cyan-400 mb-6">Correlation Heatmap</h1>
      <Plot
        data={[
          {
            z: matrix,
            x: fields,
            y: fields,
            type: 'heatmap',
            colorscale: 'Jet',
            showscale: true
          }
        ]}
        layout={{
          title: { text: 'Correlation Heatmap', font: { color: '#00FFFF' } },
          paper_bgcolor: '#1A1A2E',
          plot_bgcolor: '#1A1A2E',
          font: { color: '#00FFFF' },
          xaxis: { tickangle: 45 },
          yaxis: { automargin: true },
          margin: { t: 100, b: 150 }
        }}
        style={{ width: '100%', height: '700px' }}
      />
    </div>
  );
}

export default Heatmap;