import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

function App() {
  const [tab, setTab] = useState('chat')
  // Held in memory only (never persisted) - re-enter after a page reload.
  const [keys, setKeys] = useState({ apiKey: '', adminKey: '' })
  return (
    <div className="app">
      <header className="header">
        <h1>⚡ Semantic Cache Gateway</h1>
        <nav className="tabs">
          {['chat', 'dashboard', 'events', 'policies'].map(t => (
            <button
              key={t}
              className={`tab-btn ${tab === t ? 'active' : ''}`}
              onClick={() => setTab(t)}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
      </header>

      <div className="keys-row">
        <input type="password" placeholder="Gateway API key (if auth enabled)" value={keys.apiKey}
               onChange={e => setKeys({ ...keys, apiKey: e.target.value })} />
        <input type="password" placeholder="Admin key (dashboard / invalidation)" value={keys.adminKey}
               onChange={e => setKeys({ ...keys, adminKey: e.target.value })} />
      </div>

      {tab === 'chat' && <ChatPanel apiKey={keys.apiKey} />}
      {tab === 'dashboard' && <Dashboard adminKey={keys.adminKey} />}
      {tab === 'events' && <EventsLog />}
      {tab === 'policies' && <Policies />}
    </div>
  )
}

function ChatPanel({ apiKey }) {
  const [message, setMessage] = useState('')
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful assistant.')
  const [feature, setFeature] = useState('default')
  const [tag, setTag] = useState('default')
  const [userId, setUserId] = useState('alice')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const scrollRef = useRef(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, loading])

  const sendMessage = async () => {
    if (!message.trim()) return
    const userMessage = message
    setMessage('')
    setLoading(true)
    setMessages(prev => [...prev, { role: 'user', text: userMessage }])

    const started = performance.now()
    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(apiKey ? { 'X-API-Key': apiKey } : {}) },
        body: JSON.stringify({
          message: userMessage,
          system_prompt: systemPrompt,
          temperature: 0,
          feature,
          tag,
          user_id: tag === 'user_specific' ? userId : null
        })
      })
      if (!res.ok) {
        const errBody = await res.json().catch(() => null)
        throw new Error(errBody?.detail || `Server returned ${res.status}`)
      }
      const data = await res.json()
      const elapsed = Math.round(performance.now() - started)

      setMessages(prev => [...prev, {
        role: 'assistant',
        text: data.answer,
        cacheHit: data.cache_hit,
        similarity: data.similarity,
        matchedQuestion: data.matched_question,
        hitType: data.hit_type,
        note: data.note,
        elapsed
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: `Error: ${err.message || 'could not reach the server.'}`,
        error: true
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const badgeInfo = (m) => {
    if (m.error) return null
    if (!m.cacheHit) return { label: '🔄 Fresh LLM call', cls: 'miss' }
    if (m.hitType === 'exact') return { label: `⚡ Exact cache hit`, cls: 'exact' }
    if (m.hitType === 'semantic_verified') return { label: `🛡️ Verified semantic hit · ${m.similarity}`, cls: 'verified' }
    return { label: `🧠 Semantic hit · ${m.similarity}`, cls: 'hit' }
  }

  return (
    <div className="panel">
      <div className="controls-row">
        <div className="control">
          <label>Feature</label>
          <input value={feature} onChange={e => setFeature(e.target.value)} placeholder="e.g. support_chat" />
        </div>
        <div className="control">
          <label>Tag / Policy</label>
          <select value={tag} onChange={e => setTag(e.target.value)}>
            <option value="default">default</option>
            <option value="faq">faq</option>
            <option value="user_specific">user_specific</option>
            <option value="current_events">current_events</option>
            <option value="legal_sensitive">legal_sensitive</option>
          </select>
        </div>
        {tag === 'user_specific' && (
          <div className="control">
            <label>User ID</label>
            <select value={userId} onChange={e => setUserId(e.target.value)}>
              <option value="alice">alice</option>
              <option value="bob">bob</option>
            </select>
          </div>
        )}
      </div>

      <div className="control full">
        <label>System prompt</label>
        <input value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)} />
      </div>

      <div className="chat-window" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="empty-state">
            Ask a question, then rephrase it and ask again. Try switching the tag to
            "user_specific" and asking the same thing as "alice" then "bob" — the
            cache should never cross between them.
          </div>
        )}
        {messages.map((m, i) => {
          const badge = badgeInfo(m)
          return (
            <div key={i} className={`bubble ${m.role} ${m.error ? 'error' : ''}`}>
              <div className="bubble-text">{m.text}</div>
              {badge && <div className={`badge ${badge.cls}`}>{badge.label}</div>}
              {m.matchedQuestion && <div className="matched-question">Matched: "{m.matchedQuestion}"</div>}
              {m.note && <div className="note">{m.note}</div>}
              {m.role === 'assistant' && !m.error && (
                <div className="latency">{m.elapsed}ms</div>
              )}
            </div>
          )
        })}
        {loading && <div className="bubble assistant loading">Thinking...</div>}
      </div>

      <div className="input-row">
        <input
          value={message}
          onChange={e => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask something..."
        />
        <button onClick={sendMessage} disabled={loading}>Send</button>
      </div>
    </div>
  )
}

const fmtMs = v => (v === null || v === undefined ? '—' : `${Math.round(v * 10) / 10} ms`)
const fmtUsd = v => (v === null || v === undefined ? '—' : `$${Number(v).toFixed(v < 1 ? 5 : 2)}`)
const fmtPct = v => (v === null || v === undefined ? '—' : `${Math.round(v * 1000) / 10}%`)

function Dashboard({ adminKey }) {
  const [report, setReport] = useState(null)
  const [error, setError] = useState(null)
  const [clearing, setClearing] = useState(false)

  const fetchReport = async () => {
    if (!adminKey) return
    try {
      const res = await fetch(`${API_URL}/admin/report`, { headers: { 'X-Admin-Key': adminKey } })
      if (!res.ok) throw new Error(res.status === 401 ? 'Invalid admin key' : `Server returned ${res.status}`)
      setReport(await res.json())
      setError(null)
    } catch (e) {
      setError(e.message)
    }
  }

  useEffect(() => {
    fetchReport()
    const id = setInterval(fetchReport, 4000)
    return () => clearInterval(id)
  }, [adminKey])

  const clearCache = async () => {
    if (!window.confirm('Clear the ENTIRE cache for all tenants?')) return
    setClearing(true)
    await fetch(`${API_URL}/cache`, { method: 'DELETE', headers: { 'X-Admin-Key': adminKey } })
    await fetchReport()
    setClearing(false)
  }

  if (!adminKey) return <div className="panel"><p className="muted">Enter the admin key above to load metrics.</p></div>
  if (error) return <div className="panel"><div className="fail-open-banner">⚠️ {error}</div></div>
  if (!report) return <div className="panel"><p className="muted">Loading metrics...</p></div>

  const { overview: o, cost: c, latency: l, quality: q, reliability: rel } = report
  const failOpen = rel.fail_open_count > 0

  return (
    <div className="panel">
      {failOpen && (
        <div className="fail-open-banner">
          ⚠️ {rel.fail_open_count} request(s) were served live because the cache layer failed (fail-open).
        </div>
      )}

      <h3 className="section">Cost &amp; ROI</h3>
      <div className="stat-grid">
        <StatCard label="LLM $ Saved" value={fmtUsd(c.saved_usd)} accent="good" />
        <StatCard label="Net ROI (after OpEx)" value={fmtUsd(c.net_roi_usd)} accent="good" />
        <StatCard label="Tokens Saved (in / out)" value={`${c.saved_input_tokens} / ${c.saved_output_tokens}`} />
        <StatCard label="Live LLM Spend" value={fmtUsd(c.spent_usd)} />
      </div>

      <h3 className="section">Hit rate</h3>
      <div className="stat-grid">
        <StatCard label="Requests" value={o.requests} />
        <StatCard label="Hit Rate" value={fmtPct(o.hit_rate)} accent="good" />
        <StatCard label="Exact / Semantic" value={`${o.exact_hits} / ${o.semantic_hits}`} />
        <StatCard label="Misses / Bypass" value={`${o.misses} / ${o.bypass}`} />
      </div>

      <h3 className="section">Latency (measured)</h3>
      <div className="stat-grid">
        <StatCard label="Lookup overhead P95" value={fmtMs(l.lookup_overhead_ms.p95)}
                  accent={l.lookup_overhead_ms.slo_met_p95 ? 'good' : 'warn'} />
        <StatCard label="Cached P95 / P99" value={`${fmtMs(l.cached_ms.p95)} / ${fmtMs(l.cached_ms.p99)}`} />
        <StatCard label="Live P95 / P99" value={`${fmtMs(l.live_ms.p95)} / ${fmtMs(l.live_ms.p99)}`} />
        <StatCard label="P99 reduction" value={fmtMs(l.p99_reduction_ms)} accent="good" />
        <StatCard label="TTFT cached P50" value={fmtMs(l.ttft_cached_ms.p50)} />
        <StatCard label="TTFT live P50" value={fmtMs(l.ttft_live_ms.p50)} />
        <StatCard label="Lookup SLO breaches" value={l.lookup_overhead_ms.slo_breaches} accent="warn" />
        <StatCard label="Lookup SLO" value={`${l.lookup_overhead_ms.slo_ms} ms`} />
      </div>

      <h3 className="section">Cache quality</h3>
      <div className="stat-grid">
        <StatCard label="False-positive rate" value={fmtPct(q.false_positive_rate)} accent="warn" />
        <StatCard label="Judged hits (bad)" value={`${q.judged_hits} (${q.bad_hits})`} />
        <StatCard label="Threshold drift" value={q.threshold_drift.drift ?? '—'} />
        <StatCard label="Divergence index" value={q.semantic_divergence.index ?? '—'} />
      </div>
      <p className="muted small">Threshold advice: {q.threshold_drift.advice || q.threshold_drift.note}</p>

      <h3 className="section">Reliability &amp; security</h3>
      <div className="stat-grid">
        <StatCard label="Fail-open events" value={rel.fail_open_count} accent={failOpen ? 'warn' : ''} />
        <StatCard label="Isolation blocks" value={rel.isolation_blocks} accent={rel.isolation_blocks ? 'warn' : ''} />
        <StatCard label="Budget throttled" value={rel.budget_throttled} />
        <StatCard label="Budget denied" value={rel.budget_denied} />
      </div>

      <h3 className="section">By tenant (chargeback)</h3>
      <table className="data-table">
        <thead><tr><th>Tenant</th><th>Requests</th><th>Hit rate</th><th>Saved</th><th>Live spend</th><th>Blocks</th></tr></thead>
        <tbody>
          {Object.entries(report.by_tenant).map(([t, g]) => (
            <tr key={t}><td>{t}</td><td>{g.requests}</td><td>{fmtPct(g.hit_rate)}</td>
                <td>{fmtUsd(g.saved_usd)}</td><td>{fmtUsd(g.spent_usd)}</td><td>{g.isolation_blocks}</td></tr>
          ))}
        </tbody>
      </table>

      <p className="muted small">
        All figures are measured from live traffic. Dollar figures use the price table in pricing.json /
        gateway/pricing.py - verify those prices against your provider's current rates.
      </p>
      <button className="clear-btn" onClick={clearCache} disabled={clearing}>
        {clearing ? 'Clearing...' : 'Clear Entire Cache'}
      </button>
    </div>
  )
}

function StatCard({ label, value, accent }) {
  return (
    <div className={`stat-card ${accent || ''}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

function EventsLog() {
  const [events, setEvents] = useState([])

  const fetchEvents = async () => {
    try {
      const res = await fetch(`${API_URL}/events?limit=30`)
      const data = await res.json()
      setEvents(data.events || [])
    } catch {
      setEvents([])
    }
  }

  useEffect(() => {
    fetchEvents()
    const id = setInterval(fetchEvents, 3000)
    return () => clearInterval(id)
  }, [])

  const typeColor = (type) => ({
    exact_hit: 'exact',
    semantic_hit: 'hit',
    semantic_verified: 'verified',
    miss: 'miss',
    near_miss: 'warn',
  }[type] || '')

  return (
    <div className="panel">
      <p className="muted small">Live decision log — refreshes every 3s. Newest first.</p>
      <div className="events-list">
        {events.length === 0 && <div className="empty-state">No events yet — send a chat message first.</div>}
        {events.map((e, i) => (
          <div key={i} className={`event-row ${typeColor(e.type)}`}>
            <span className="event-type">{e.type}</span>
            <span className="event-feature">{e.feature} / {e.tag}</span>
            <span className="event-sim">{e.similarity != null ? `sim ${Number(e.similarity).toFixed(3)}` : '—'}</span>
            <span className="event-time">{new Date(e.timestamp * 1000).toLocaleTimeString()}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function Policies() {
  const [policies, setPolicies] = useState(null)

  useEffect(() => {
    fetch(`${API_URL}/policies`).then(r => r.json()).then(setPolicies).catch(() => setPolicies({}))
  }, [])

  if (!policies) return <div className="panel"><p className="muted">Loading...</p></div>

  return (
    <div className="panel">
      <p className="muted small">Per-tag policy configuration currently active on the gateway (loaded from config.json).</p>
      <table className="policy-table">
        <thead>
          <tr><th>Tag</th><th>Auto-hit</th><th>Borderline low</th><th>TTL (s)</th><th>Caching Allowed</th></tr>
        </thead>
        <tbody>
          {Object.entries(policies).map(([tag, p]) => (
            <tr key={tag}>
              <td>{tag}</td>
              <td>{p.auto_hit}</td>
              <td>{p.borderline_low}</td>
              <td>{p.ttl}</td>
              <td>{p.cache_allowed ? '✅' : '🚫'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default App
