'use client';
import React from 'react';
import Chat from '../components/Chat';
import AdminPanel from '../components/AdminPanel';

export default function Page() {
  const [showAdmin, setShowAdmin] = React.useState(false);

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 text-gray-800">
      <div className="max-w-5xl mx-auto px-6 py-10 space-y-10">

        {/* Header */}
        <header className="text-center space-y-3">
          <h1 className="text-4xl font-bold text-gray-900">
            AI Policy & Product Helper
          </h1>
        </header>

        {/* Toggle Button */}
        <div className="text-center">
          <button
            onClick={() => setShowAdmin(!showAdmin)}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-all duration-200
              ${showAdmin
                ? 'bg-red-100 text-red-700 hover:bg-red-200'
                : 'bg-blue-600 text-white hover:bg-blue-700 shadow-sm'
              }`}
          >
            {showAdmin ? 'Hide Admin Panel' : 'Show Admin Panel'}
          </button>
        </div>

        {/* Admin Panel (conditionally rendered) */}
        {showAdmin && (
          <section className="bg-white rounded-2xl shadow-md p-6 border border-gray-200 transition-all duration-300 animate-fadeIn">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xl font-semibold flex items-center gap-2">
                <span>📂</span> Admin Panel
              </h2>
              <span className="text-xs text-gray-500 italic">
                Manage data ingestion & metrics
              </span>
            </div>
            <p className="text-sm text-gray-500 mb-4">
              Ingest Markdown documents from the <code>/data</code> folder and monitor embedding statistics.
            </p>
            <AdminPanel />
          </section>
        )}

        {/* Chat Section */}
        <section className="bg-white rounded-2xl shadow-md p-6 border border-gray-200 transition-all duration-300">
          <h2 className="text-xl font-semibold mb-3 flex items-center gap-2">
            <span>💬</span> Ask the AI
          </h2>
          <p className="text-sm text-gray-500 mb-4">
            Chat with your company policy assistant and receive answers grounded in the ingested documents.
          </p>
          <Chat />
        </section>

        {/* Testing Guide */}
        <section className="bg-gray-50 border border-gray-200 rounded-2xl p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
            How to Test
          </h3>
          <ol className="list-decimal list-inside space-y-2 text-gray-700">
            <li>Click <b>Ingest sample docs</b> in the Admin Panel.</li>
            <li>Ask: <i>“Can a customer return a damaged blender after 20 days?”</i></li>
            <li>Ask: <i>“What’s the shipping SLA to East Malaysia for bulky items?”</i></li>
          </ol>
        </section>
      </div>
    </main>
  );
}
