import React, { useState } from 'react';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://<backend-url>:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      const data = await res.json();
      if (res.ok) {
        setResponse(data.answer);
      } else {
        setError(data.error || 'An error occurred');
      }
    } catch (err) {
      setError('Failed to connect to the server');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold mb-4">MedicoBot</h1>
      <input
        type="text"
        className="border p-2 rounded w-full mb-4"
        placeholder="Enter your medical question"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />
      <button
        className="bg-blue-500 text-white p-2 rounded"
        onClick={handlePredict}
        disabled={loading}
      >
        {loading ? 'Loading...' : 'Predict'}
      </button>
      {response && <div className="mt-4 p-4 bg-green-100 rounded">Answer: {response}</div>}
      {error && <div className="mt-4 p-4 bg-red-100 rounded">Error: {error}</div>}
    </div>
  );
}

export default App;
