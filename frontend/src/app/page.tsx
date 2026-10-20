"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  Crosshair,
  Gauge,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Trophy,
  Users,
  Wallet,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type PlayerForm = {
  age: string;
  position: "Goalkeeper" | "Defender" | "Midfielder" | "Forward";
  minutes: string;
  goals_scored: string;
  assists: string;
  total_points: string;
  goals_conceded: string;
  clean_sheets: string;
  bonus: string;
  bps: string;
  influence: string;
  creativity: string;
  threat: string;
  ict_index: string;
  red_cards: string;
  yellow_cards: string;
  selected_by_percent: string;
  now_cost: string;
};

const initialForm: PlayerForm = {
  age: "23",
  position: "Midfielder",
  minutes: "2450",
  goals_scored: "9",
  assists: "7",
  total_points: "164",
  goals_conceded: "25",
  clean_sheets: "8",
  bonus: "14",
  bps: "510",
  influence: "690",
  creativity: "720",
  threat: "640",
  ict_index: "205",
  red_cards: "0",
  yellow_cards: "4",
  selected_by_percent: "18.5",
  now_cost: "75",
};

const statFields: { key: Exclude<keyof PlayerForm, "position">; label: string; hint?: string }[] = [
  { key: "age", label: "Age", hint: "Years" },
  { key: "minutes", label: "Minutes played", hint: "Season total" },
  { key: "goals_scored", label: "Goals" },
  { key: "assists", label: "Assists" },
  { key: "total_points", label: "FPL points" },
  { key: "goals_conceded", label: "Goals conceded" },
  { key: "clean_sheets", label: "Clean sheets" },
  { key: "bonus", label: "Bonus points" },
  { key: "bps", label: "BPS" },
  { key: "influence", label: "Influence" },
  { key: "creativity", label: "Creativity" },
  { key: "threat", label: "Threat" },
  { key: "ict_index", label: "ICT index" },
  { key: "red_cards", label: "Red cards" },
  { key: "yellow_cards", label: "Yellow cards" },
  { key: "selected_by_percent", label: "Selected by", hint: "%" },
  { key: "now_cost", label: "FPL price", hint: "Tenths of £m" },
];

function money(value: number) {
  if (value >= 1_000_000) return "€" + (value / 1_000_000).toFixed(2) + "m";
  return new Intl.NumberFormat("en-GB", { style: "currency", currency: "EUR", maximumFractionDigits: 0 }).format(value);
}

export default function Home() {
  const [form, setForm] = useState<PlayerForm>(initialForm);
  const [fee, setFee] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(true);
  const [lastUpdated, setLastUpdated] = useState("");

  const goalsInvolvement = useMemo(
    () => Number(form.goals_scored || 0) + Number(form.assists || 0),
    [form.goals_scored, form.assists],
  );
  const per90 = useMemo(() => {
    const mins = Number(form.minutes);
    return mins > 0 ? (goalsInvolvement / mins) * 90 : 0;
  }, [form.minutes, goalsInvolvement]);

  function update<K extends keyof PlayerForm>(key: K, value: PlayerForm[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function estimate(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setFee(null);
    try {
      const payload = {
        ...Object.fromEntries(
          Object.entries(form).filter(([key]) => key !== "position").map(([key, value]) => [key, Number(value)]),
        ),
        position: form.position,
        element_type: ({ Goalkeeper: "1", Defender: "2", Midfielder: "3", Forward: "4" } as const)[form.position],
      };
      const response = await fetch(`${API_URL}/api/v1/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "Prediction request failed.");
      setFee(Number(body.predicted_fee_eur));
      setLastUpdated(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not connect to the TransferIQ API.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="TransferIQ home">
          <span className="brand-mark"><Activity size={21} strokeWidth={2.6} /></span>
          <span>Transfer<span className="brand-accent">IQ</span><small>FOOTBALL INTELLIGENCE</small></span>
        </a>
        <div className="side-label">WORKSPACE</div>
        <a className="side-link active" href="#valuation"><Gauge size={18} /> Valuation studio <span className="side-dot" /></a>
        <a className="side-link" href="#methodology"><TrendingUp size={18} /> Model insights</a>
        <a className="side-link" href="#about"><Trophy size={18} /> About the project</a>
        <div className="sidebar-bottom">
          <div className="status-card">
            <span className="status-pulse" />
            <div><strong>Model workspace</strong><small>Inference API configured</small></div>
          </div>
          <div className="profile">
            <div className="profile-avatar">TI</div>
            <div><strong>TransferIQ</strong><small>Research edition</small></div>
            <ChevronDown size={16} className="profile-chevron" />
          </div>
        </div>
      </aside>

      <section className="main-area">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> <strong>Valuation studio</strong></div>
          <div className="topbar-right"><span className="live-indicator"><i /> API-driven estimates</span><span className="topbar-avatar">IQ</span></div>
        </header>

        <div className="content">
          <section className="hero">
            <div className="hero-copy">
              <div className="eyebrow"><Sparkles size={13} /> TRANSFER MARKET ANALYTICS</div>
              <h1>Every player has a value.<br /><span>Find the signal.</span></h1>
              <p>Explore data-informed transfer fee estimates built from player performance, availability and age — not guesswork.</p>
              <div className="hero-tags"><span><ShieldCheck size={14} /> Machine learning</span><span><Clock3 size={14} /> Season-level data</span><span><Crosshair size={14} /> Premier League</span></div>
            </div>
            <div className="hero-graphic" aria-hidden="true">
              <div className="graphic-orbit orbit-one" /><div className="graphic-orbit orbit-two" />
              <div className="graphic-ball"><span>IQ</span><div className="ball-line line-a" /><div className="ball-line line-b" /><div className="ball-line line-c" /></div>
              <div className="graphic-chip chip-top"><TrendingUp size={15} /> Performance</div>
              <div className="graphic-chip chip-bottom"><Wallet size={15} /> Valuation</div>
              <div className="graphic-glow" />
            </div>
          </section>

          <section className="metric-grid" aria-label="Project highlights">
            <div className="metric-card"><div className="metric-icon violet"><Activity size={17} /></div><span className="metric-label">VALUATION ENGINE</span><strong>Regression</strong><small>Supervised ML pipeline</small></div>
            <div className="metric-card"><div className="metric-icon green"><TrendingUp size={17} /></div><span className="metric-label">INPUT SIGNALS</span><strong>17 statistics</strong><small>Performance & availability</small></div>
            <div className="metric-card"><div className="metric-icon amber"><Users size={17} /></div><span className="metric-label">TARGET MARKET</span><strong>Premier League</strong><small>Historical transfer records</small></div>
            <div className="metric-card"><div className="metric-icon blue"><ShieldCheck size={17} /></div><span className="metric-label">PREDICTION MODE</span><strong>Model-backed</strong><small>Served by FastAPI</small></div>
          </section>

          <section className="workspace-grid" id="valuation">
            <div className="panel form-panel">
              <div className="panel-heading">
                <div><div className="section-kicker">01 / PLAYER PROFILE</div><h2>Build a valuation</h2><p>Enter season statistics to generate an estimate.</p></div>
                <span className="step-badge">INPUT</span>
              </div>
              <form onSubmit={estimate}>
                <div className="field-grid profile-fields">
                  <label className="field"><span>Player age</span><div className="input-wrap"><input type="number" min="16" max="45" step="1" required value={form.age} onChange={(e) => update("age", e.target.value)} /><small>yrs</small></div></label>
                  <label className="field"><span>Primary position</span><select value={form.position} onChange={(e) => update("position", e.target.value as PlayerForm["position"])}><option>Goalkeeper</option><option>Defender</option><option>Midfielder</option><option>Forward</option></select></label>
                  <div className="field full-field"><span>Attacking contribution</span><div className="contribution-card"><div><span>Goals + assists</span><strong>{goalsInvolvement}</strong></div><div className="contribution-divider" /><div><span>Contributions / 90</span><strong>{per90.toFixed(2)}</strong></div><div className="contribution-icon"><Crosshair size={19} /></div></div></div>
                </div>
                <div className="section-divider"><span>PERFORMANCE METRICS</span><span>01 — 08</span></div>
                <div className="field-grid">
                  {statFields.slice(1, 9).map((field) => <label className="field" key={field.key}><span>{field.label}</span><div className="input-wrap"><input type="number" min="0" step="any" required value={form[field.key]} onChange={(e) => update(field.key, e.target.value)} />{field.hint && <small>{field.hint}</small>}</div></label>)}
                </div>
                <button type="button" className="advanced-toggle" onClick={() => setShowAdvanced((v) => !v)} aria-expanded={showAdvanced}><span><Activity size={16} /> Advanced metrics <small>Discipline, ICT & popularity</small></span><ChevronDown size={17} className={showAdvanced ? "rotate" : ""} /></button>
                {showAdvanced && <div className="field-grid advanced-grid">{statFields.slice(9).map((field) => <label className="field" key={field.key}><span>{field.label}</span><div className="input-wrap"><input type="number" min="0" step="any" required value={form[field.key]} onChange={(e) => update(field.key, e.target.value)} />{field.hint && <small>{field.hint}</small>}</div></label>)}</div>}
                {error && <div className="error-banner"><CircleHelp size={17} /><span>{error}</span></div>}
                <button className="submit-button" type="submit" disabled={loading}>{loading ? <><span className="spinner" /> Calculating estimate…</> : <>Generate valuation <ArrowRight size={17} /></>}</button>
                <p className="form-footnote"><ShieldCheck size={13} /> Your inputs are sent to the local API for inference.</p>
              </form>
            </div>

            <div className="result-column">
              <div className="panel result-panel">
                <div className="result-topline"><div className="section-kicker">02 / ESTIMATION</div><span className="result-icon"><Wallet size={17} /></span></div>
                <h2>Estimated transfer fee</h2>
                <p className="result-subtitle">Model output · Euro currency</p>
                <div className={fee === null ? "result-value empty-value" : "result-value"}>{fee === null ? "€ —" : money(fee)}</div>
                <div className="result-caption">{fee === null ? "Your estimate will appear here" : "Prediction generated from your submitted statistics"}</div>
                <div className="result-rule" />
                <div className="result-details"><span>Currency</span><strong>EUR (€)</strong></div>
                <div className="result-details"><span>Input profile</span><strong>{form.position}</strong></div>
                <div className="result-details"><span>Goals + assists</span><strong>{goalsInvolvement}</strong></div>
                <div className="result-details"><span>Last calculation</span><strong>{lastUpdated || "Not yet run"}</strong></div>
                {fee !== null && <div className="success-note"><CheckCircle2 size={15} /> API prediction received successfully.</div>}
                <div className="result-disclaimer"><CircleHelp size={15} /><p>This is a research estimate, not an official market valuation or a guaranteed transfer fee.</p></div>
              </div>

              <div className="panel insight-panel" id="methodology">
                <div className="insight-heading"><div className="insight-symbol"><TrendingUp size={17} /></div><div><h3>What influences value?</h3><p>Signals represented in this model</p></div></div>
                <div className="signal-row"><span className="signal-dot signal-purple" /><div><strong>Age & development curve</strong><small>Age and non-linear age effect</small></div><ArrowUpRight size={16} /></div>
                <div className="signal-row"><span className="signal-dot signal-green" /><div><strong>On-pitch production</strong><small>Goals, assists and FPL performance</small></div><ArrowUpRight size={16} /></div>
                <div className="signal-row"><span className="signal-dot signal-orange" /><div><strong>Availability & discipline</strong><small>Minutes, cards and playing time</small></div><ArrowDownRight size={16} /></div>
                <div className="insight-note">The model cannot fully capture contracts, injuries, club finances or negotiation context.</div>
              </div>
            </div>
          </section>

          <section className="methodology-strip" id="about">
            <div className="method-icon"><Trophy size={19} /></div><div><strong>Built for exploration, not certainty.</strong><p>TransferIQ combines a Python feature pipeline, a scikit-learn regression model and a FastAPI service behind a modern web interface.</p></div><a href="https://github.com/thisisahmad24/transfer-iq" target="_blank" rel="noreferrer">View source <ArrowRight size={15} /></a>
          </section>
          <footer className="footer"><span>© 2026 TransferIQ</span><span>Independent football analytics · Estimates are not financial advice</span><span className="footer-status"><i /> API interface ready</span></footer>
        </div>
      </section>
    </main>
  );
}
