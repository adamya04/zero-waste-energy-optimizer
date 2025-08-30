import React, { useState } from 'react';
import axios from 'axios';

function ReportGenerator() {
  const [reportType, setReportType] = useState('sustainability');
  const [pdfUrl, setPdfUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generateReport = async () => {
    setLoading(true);
    setPdfUrl('');
    setError(null);
    try {
      const response = await axios.post(
        `/api/reports/${reportType}`,
        { store_id: 'store_1' },
        { responseType: 'blob' }
      );
      // create blob URL for download
      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      setPdfUrl(url);
    } catch (err) {
      console.error('Report generation failed', err);
      setError('Report generation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8">
      <h1 className="text-4xl text-cyan-400 mb-4">Report Generator</h1>
      <div className="bg-tron-dark p-6 rounded shadow-neon">
        <label className="block mb-2 text-magenta-400">Select report type</label>
        <select value={reportType} onChange={(e) => setReportType(e.target.value)} className="p-2 mb-4 rounded bg-[#0f1622]">
          <option value="sustainability">Sustainability Report</option>
          <option value="blockchain">Blockchain Transactions Report</option>
        </select>
        <button onClick={generateReport} className="px-4 py-2 bg-cyan-400 rounded text-tron-bg">
          {loading ? 'Generating…' : 'Generate Report'}
        </button>

        {error && <div className="text-red-500 mt-4">{error}</div>}

        {pdfUrl && (
          <div className="mt-4">
            <a href={pdfUrl} download={`${reportType}_report_store_1.pdf`} className="p-3 bg-tron-dark text-cyan-400 rounded shadow-neon">
              Download PDF
            </a>
            <iframe src={pdfUrl} title="Report Preview" className="w-full h-[600px] mt-4" />
          </div>
        )}
      </div>
    </div>
  );
}

export default ReportGenerator;
