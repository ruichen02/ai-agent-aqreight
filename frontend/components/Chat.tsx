'use client';
import React, { useEffect, useRef } from 'react';
import { apiAsk } from '../lib/api';

type Message = {
  role: 'user' | 'assistant';
  content: string;
  citations?: { title: string; section?: string }[];
  chunks?: { title: string; section?: string; text: string }[];
  time?: string;
};

export default function Chat() {
  const [messages, setMessages] = React.useState<Message[]>([
    {
      role: 'assistant',
      content: "Hi, I'm an AI Policy & Product Helper. You may ask me questions about your company's policies and products. How can I assist you today?",
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [q, setQ] = React.useState('');
  const [loading, setLoading] = React.useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async () => {
    if (!q.trim()) return;

    const my: Message = { 
      role: 'user', 
      content: q,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(m => [...m, my]);
    setLoading(true);

    try {
      const res = await apiAsk(q);
      const ai: Message = { 
        role: 'assistant', 
        content: res.answer, 
        citations: res.citations, 
        chunks: res.chunks,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(m => [...m, ai]);
    } catch (e: any) {
      setMessages(m => [
        ...m, 
        { 
          role: 'assistant', 
          content: 'Error: ' + e.message,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } finally {
      setLoading(false);
      setQ('');
    }
  };

  return (
    <div className="card" style={{ padding: 20 }}>
      <h2>Chat</h2>

      <div
        style={{
          maxHeight: 400,
          overflowY: 'auto',
          marginBottom: 16,
          display: 'flex',
          flexDirection: 'column',
          gap: 16
        }}
      >
        {messages.map((m, i) => (
          <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {m.role === 'user' ? (
              <div
                style={{
                  alignSelf: 'flex-end',
                  background: '#f3f3f3',
                  borderRadius: '12px 12px 0 12px',
                  padding: '10px 14px',
                  maxWidth: '80%',
                  fontWeight: 500
                }}
              >
                {m.content}
                {m.time && (
                  <div
                    style={{
                      fontSize: 11,
                      color: '#999',
                      marginTop: 4,
                      textAlign: 'right'
                    }}
                  >
                    {m.time}
                  </div>
                )}
              </div>
            ) : (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 8
                }}
              >
                <div
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: '50%',
                    background: '#E0E0E0',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#333',
                    fontWeight: 600,
                    fontSize: 16,
                    flexShrink: 0
                  }}
                >
                  AI
                </div>

                <div
                  style={{
                    alignSelf: 'flex-start',
                    background: '#ffffff',
                    border: '1px solid #e5e5e5',
                    borderRadius: '12px 12px 12px 0',
                    padding: '12px 16px',
                    maxWidth: '90%',
                    boxShadow: '0 2px 5px rgba(0,0,0,0.05)'
                  }}
                >
                  <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>

                  {m.citations && m.citations.length > 0 && (
                    <div style={{ marginTop: 8 }}>
                      {m.citations.map((c, idx) => (
                        <span
                          key={idx}
                          className="badge"
                          style={{
                            display: 'inline-block',
                            background: '#eef',
                            color: '#336',
                            borderRadius: 6,
                            padding: '2px 6px',
                            fontSize: 12,
                            marginRight: 6
                          }}
                          title={c.section || ''}
                        >
                          {c.title}
                        </span>
                      ))}
                    </div>
                  )}

                  {m.chunks && m.chunks.length > 0 && (
                    <details style={{ marginTop: 8 }}>
                      <summary style={{ cursor: 'pointer', fontSize: 14, color: '#555' }}>
                        View supporting chunks
                      </summary>
                      {m.chunks.map((c, idx) => (
                        <div
                          key={idx}
                          style={{
                            background: '#f9f9f9',
                            borderRadius: 8,
                            padding: 8,
                            marginTop: 6,
                            border: '1px solid #eee'
                          }}
                        >
                          <div style={{ fontWeight: 600 }}>
                            {c.title}
                            {c.section ? ' — ' + c.section : ''}
                          </div>
                          <div style={{ whiteSpace: 'pre-wrap', fontSize: 14 }}>{c.text}</div>
                        </div>
                      ))}
                    </details>
                  )}

                  {m.time && (
                    <div
                      style={{
                        fontSize: 11,
                        color: '#999',
                        marginTop: 6,
                        textAlign: 'left'
                      }}
                    >
                      {m.time}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        <input
          placeholder="Ask about policy or products..."
          value={q}
          onChange={e => setQ(e.target.value)}
          style={{
            flex: 1,
            padding: 12,
            borderRadius: 8,
            border: '1px solid #ddd'
          }}
          onKeyDown={e => {
            if (e.key === 'Enter') send();
          }}
        />
        <button
          onClick={send}
          disabled={loading}
          style={{
            padding: '10px 16px',
            borderRadius: 8,
            border: 'none',
            background: '#111',
            color: '#fff',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? 'Thinking...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
