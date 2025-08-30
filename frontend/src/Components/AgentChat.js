// frontend/src/Components/AgentChat.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function AgentChat() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query) return;
    setLoading(true);
    try {
      // Use relative path so Vite dev proxy handles routing
      const res = await axios.post('/api/agent', { query });
      setResponse(res.data.response || res.data);
      setHistory(prev => [{ query, response: res.data.response || res.data }, ...prev]);
    } catch (err) {
      console.error('Agent API error:', err);
      setResponse('Error: Could not process query (see console).');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8">
      <h1 className="text-4xl text-cyan-400 mb-4">AI Agent Chat</h1>
      <form onSubmit={handleSubmit} className="mb-4">
        <input
          className="p-3 rounded w-full bg-tron-dark text-white"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about inventory, sustainability or optimization..."
          aria-label="Agent query"
        />
        <button type="submit" className="mt-3 px-4 py-2 bg-cyan-400 text-tron-bg rounded">Ask</button>
      </form>

      {loading ? (
        <div className="text-cyan-400">Thinking…</div>
      ) : (
        response && <div className="whitespace-pre-wrap bg-tron-dark p-4 rounded">{response}</div>
      )}

      <div className="mt-6">
        <h2 className="text-2xl text-magenta-400 mb-2">History</h2>
        <ul>
          {history.map((h, i) => (
            <li key={i} className="mb-4">
              <div className="font-bold">{h.query}</div>
              <div className="whitespace-pre-wrap">{h.response}</div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default AgentChat;
