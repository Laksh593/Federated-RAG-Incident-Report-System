import { useState } from 'react'
import ReactMarkdown from 'react-markdown'

const INCIDENT_TYPES = [
    'Fighting',
    'Vandalism',
    'Theft',
    'Suspicious Activity',
    'Cyber Intrusion',
    'Fire / Hazard',
]

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function App() {
    const [formData, setFormData] = useState({
        type: 'Fighting',
        location: '',
        description: 'Physical altercation detected between two subjects.',
        timestamp: new Date().toISOString(),
    })

    const [report, setReport] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [videoFile, setVideoFile] = useState(null)

    const handleGenerate = async () => {
        setLoading(true)
        setError(null)
        try {
            let response;
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 5 * 60 * 1000); // 5 min timeout

            const resolvedLocation = formData.location.trim() || 'Unknown';

            if (videoFile) {
                const formDataPayload = new FormData();
                formDataPayload.append('type', formData.type);
                formDataPayload.append('location', resolvedLocation);
                formDataPayload.append('description', formData.description);
                formDataPayload.append('timestamp', formData.timestamp);
                formDataPayload.append('file', videoFile);

                console.log(`[DEBUG] Uploading video: ${videoFile.name} (${(videoFile.size / 1024).toFixed(1)} KB) to ${API_BASE_URL}/api/generate-video`);
                response = await fetch(`${API_BASE_URL}/api/generate-video`, {
                    method: 'POST',
                    body: formDataPayload,
                    signal: controller.signal,
                });
            } else {
                console.log(`[DEBUG] Sending text-only request to ${API_BASE_URL}/api/generate`);
                response = await fetch(`${API_BASE_URL}/api/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        ...formData,
                        location: resolvedLocation,
                    }),
                    signal: controller.signal,
                })
            }
            clearTimeout(timeout);
            console.log(`[DEBUG] Response received: ${response.status} ${response.statusText}`);
            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error: ${response.status} ${response.statusText}`);
            }
            const data = await response.json()
            setReport(data.report)
        } catch (err) {
            if (err.name === 'AbortError') {
                setError('Request timed out after 5 minutes. The video may be too large or the server is busy. Try a shorter video clip.');
            } else {
                console.error('[DEBUG] Fetch error:', err);
                setError(`Connection error: ${err.message}. Make sure the backend is running at ${API_BASE_URL}`);
            }
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="app-wrapper">
            {/* ── Header ─────────────────────────────────── */}
            <header className="header">
                <div className="container">
                    <div className="header-inner">
                        <div className="logo-group">
                            <div className="logo-icon">🛡️</div>
                            <div>
                                <div className="logo-title">Sentinel AI</div>
                                <div className="logo-sub">Federated RAG · Incident Report Generator</div>
                            </div>
                        </div>
                        <div className="status-badge">
                            <span className="status-dot" />
                            System Online
                        </div>
                    </div>
                </div>
            </header>

            {/* ── Body ───────────────────────────────────── */}
            <main className="main-content">
                <div className="container">
                    <div className="grid-layout">

                        {/* ── Input Panel ─────────────────────── */}
                        <div>
                            <div className="panel">
                                <div className="panel-header">
                                    <div className="panel-icon blue">⚡</div>
                                    <span className="panel-title">Incident Details</span>
                                </div>

                                <div className="field-group">

                                    {/* Incident Type Chips */}
                                    <div className="field">
                                        <label>Incident Type</label>
                                        <div className="type-chips">
                                            {INCIDENT_TYPES.map(t => (
                                                <button
                                                    key={t}
                                                    className={`chip${formData.type === t ? ' active' : ''}`}
                                                    onClick={() => setFormData({ ...formData, type: t })}
                                                >
                                                    {t}
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Location */}
                                    <div className="field">
                                        <label>Incident Location</label>
                                        <div className="input-wrap">
                                            <span className="input-icon">📍</span>
                                            <input
                                                type="text"
                                                className="has-icon"
                                                value={formData.location}
                                                onChange={e => setFormData({ ...formData, location: e.target.value })}
                                                placeholder="e.g. North Gate, Building 4 (default: Unknown)"
                                            />
                                        </div>
                                    </div>

                                    {/* Description */}
                                    <div className="field">
                                        <label>Description</label>
                                        <textarea
                                            value={formData.description}
                                            onChange={e => setFormData({ ...formData, description: e.target.value })}
                                            placeholder="Describe the anomaly or incident in detail…"
                                            rows={5}
                                        />
                                    </div>

                                    {/* Video Upload */}
                                    <div className="field">
                                        <label>Attach Video Evidence (Optional)</label>
                                        <div className="input-wrap file-wrap" style={{ position: 'relative' }}>
                                            <span className="input-icon" style={{ zIndex: 10 }}>📹</span>
                                            <input
                                                type="file"
                                                accept="video/mp4,video/quicktime,video/x-msvideo"
                                                onChange={e => setVideoFile(e.target.files[0])}
                                                style={{
                                                    backgroundColor: 'rgba(255,255,255,0.03)',
                                                    border: '1px solid rgba(255,255,255,0.1)',
                                                    borderRadius: '8px',
                                                    padding: '10px 10px 10px 45px',
                                                    color: 'var(--text-main)',
                                                    width: '100%',
                                                    outline: 'none',
                                                    cursor: 'pointer'
                                                }}
                                            />
                                        </div>
                                    </div>

                                    {/* Timestamp */}
                                    <div className="timestamp-bar">
                                        🕐&nbsp; <strong style={{ color: 'var(--text-label)' }}>Timestamp:</strong>
                                        &nbsp;{new Date(formData.timestamp).toLocaleString()}
                                    </div>

                                    {/* Generate */}
                                    <button
                                        className="btn-generate"
                                        onClick={handleGenerate}
                                        disabled={loading}
                                    >
                                        {loading ? (
                                            <><span className="spin">⟳</span> Generating Report…</>
                                        ) : (
                                            <><span>✦</span> Generate Report</>
                                        )}
                                    </button>
                                </div>
                            </div>

                            {error && (
                                <div className="error-box">
                                    <span className="error-icon">⚠</span>
                                    <span>{error}</span>
                                </div>
                            )}
                        </div>

                        {/* ── Report Panel ─────────────────────── */}
                        <div className={`panel report-panel ${report ? 'filled' : 'empty'}`}>
                            <div className="panel-header">
                                <div className="panel-icon indigo">📄</div>
                                <span className="panel-title">Generated Report</span>
                            </div>

                            {report ? (
                                <div className="report-body">
                                    <ReactMarkdown>{report}</ReactMarkdown>
                                </div>
                            ) : (
                                <div className="report-empty">
                                    <div className="report-empty-icon">📋</div>
                                    <p className="report-empty-text">
                                        {loading ? 'Processing your incident data…' : 'No report generated yet. Fill in the details and click Generate.'}
                                    </p>
                                </div>
                            )}
                        </div>

                    </div>
                </div>
            </main>

            {/* ── Footer ─────────────────────────────────── */}
            <footer className="footer">
                <div className="container">
                    Federated RAG · Privacy-Preserving Incident Intelligence
                </div>
            </footer>
        </div>
    )
}

export default App
