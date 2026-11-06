import { useCallback, useEffect, useRef, useState } from 'react'
import { animate, createScope, stagger, svg } from 'animejs'
import { api, type Division, type Hit, type Match, type Profile, type Similar, type Status } from './api'
import StyleReel, { octagon } from './StyleReel'
import Compare, { type Picked } from './Compare'
import StyleMap from './StyleMap'
import Replay from './Replay'
import Home from './Home'
import Avatar from './Avatar'

const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches
const show = (v: number | null) => (v == null ? '—' : String(v))
const reason = (m: Match) => {
  const c = m.closest_characteristics[0]
  const d = m.largest_differences[0]
  const part = (label: string, x: typeof c) =>
    x ? `${label}: ${x.label} ${show(x.query_value)} vs ${show(x.match_value)} ${x.unit}` : ''
  return [part('Closest', c), part('Differs', d)].filter(Boolean)
}

export default function App() {
  const root = useRef<HTMLDivElement>(null)
  const [status, setStatus] = useState<Status | null>(null)
  const [statusErr, setStatusErr] = useState<string | null>(null)
  const [divisions, setDivisions] = useState<Division[]>([])
  const [q, setQ] = useState('')
  const [hits, setHits] = useState<Hit[]>([])
  const [searching, setSearching] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [profile, setProfile] = useState<Profile | null>(null)
  const [similar, setSimilar] = useState<Similar | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [division, setDivision] = useState('')
  const [category, setCategory] = useState('')
  const [minMinutes, setMinMinutes] = useState('')
  const [route, setRoute] = useState(() => location.hash.slice(1) || 'scout')
  const [picks, setPicks] = useState<Picked[]>([])
  const [pickMsg, setPickMsg] = useState<string | null>(null)

  useEffect(() => {
    const on = () => setRoute(location.hash.slice(1) || 'scout')
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  const togglePick = (id: string, name: string | null) => {
    if (picks.some((p) => p.id === id)) { setPicks(picks.filter((p) => p.id !== id)); setPickMsg(null) }
    else if (picks.length >= 4) setPickMsg('Compare holds up to 4 fighters. Remove one first.')
    else { setPicks([...picks, { id, name: name ?? id }]); setPickMsg(null) }
  }

  const loadStatus = useCallback(() => {
    setStatusErr(null)
    api.status().then(setStatus).catch((e: Error) => setStatusErr(e.message))
    api.divisions().then((d) => setDivisions(d.divisions)).catch(() => undefined)
  }, [])
  useEffect(loadStatus, [loadStatus])

  // Opening sequence: draw the octagon frame, then reveal the workspace (about 800 ms).
  useEffect(() => {
    if (reduced()) return
    const scope = createScope({ root }).add(() => {
      animate(svg.createDrawable('.frame-path'), { draw: ['0 0', '0 1'], duration: 800, ease: 'inOut(2)' })
      animate('.reveal', { opacity: [0, 1], translateY: [10, 0], duration: 600, delay: stagger(90, { start: 150 }) })
    })
    return () => scope.revert()
  }, [])

  // Debounced name search.
  useEffect(() => {
    if (q.trim().length < 2) { setHits([]); return }
    setSearching(true)
    const t = setTimeout(() => {
      api.search(q.trim()).then(setHits).catch((e: Error) => setError(e.message)).finally(() => setSearching(false))
    }, 250)
    return () => clearTimeout(t)
  }, [q])

  // Load the profile and matches whenever the fighter or a filter changes.
  useEffect(() => {
    if (!selectedId) return
    let live = true
    setLoading(true); setError(null)
    Promise.all([
      api.profile(selectedId),
      api.similar(selectedId, {
        n: 10, division: division || undefined, category: category || undefined,
        min_minutes: minMinutes || undefined,
      }),
    ])
      .then(([p, s]) => { if (live) { setProfile(p); setSimilar(s) } })
      .catch((e: Error) => { if (live) setError(e.message) })
      .finally(() => { if (live) setLoading(false) })
    return () => { live = false }
  }, [selectedId, division, category, minMinutes])

  // Short fade when the match list changes.
  useEffect(() => {
    if (!similar || reduced()) return
    const scope = createScope({ root }).add(() => {
      animate('.match', { opacity: [0, 1], duration: 200, delay: stagger(25) })
    })
    return () => scope.revert()
  }, [similar])

  const choose = (id: string) => {
    setSelectedId(id); setQ(''); setHits([])
    if (location.hash !== '#scout') location.hash = '#scout'
  }
  const resetFilters = () => { setDivision(''); setCategory(''); setMinMinutes('') }
  const badge = status
    ? `${status.data_mode_label} · UFC through ${status.organizations[0]?.data_through ?? 'unknown date'}`
    : statusErr ? 'Data unavailable' : 'Checking data…'

  return (
    <div className="app" ref={root}>
      <svg className="frame" viewBox="0 0 400 400" aria-hidden="true">
        <polygon className="frame-path" points={octagon(190)} />
      </svg>
      <nav className="nav reveal" aria-label="Main">
        <span className="logo">CAGE ATLAS</span>
        <a className="navlink" href="#scout" aria-current={route === 'scout' ? 'page' : undefined}>Scout</a>
        <a className="navlink" href="#compare" aria-current={route === 'compare' ? 'page' : undefined}>Compare{picks.length ? ` (${picks.length})` : ''}</a>
        <a className="navlink" href="#replay" aria-current={route === 'replay' ? 'page' : undefined}>Replay</a>
        <a className="navlink" href="#map" aria-current={route === 'map' ? 'page' : undefined}>Map</a>
      </nav>
      <header className="top reveal">
        <div className={`search${profile ? '' : ' attn'}`}>
          <label className="sr" htmlFor="q">Search fighters</label>
          <input id="q" value={q} onChange={(e) => setQ(e.target.value)} autoComplete="off"
            placeholder="Search a UFC fighter by name" />
          {q.trim().length >= 2 && (
            <ul className="hits">
              {hits.map((h) => (
                <li key={h.fighter_id}>
                  <button onClick={() => choose(h.fighter_id)}>
                    <Avatar id={h.fighter_id} name={h.fighter_name} size={30} />
                    <div>{h.fighter_name} <span>{h.weight_class ?? ''} · {h.fights} fights{h.in_model_cohort ? '' : ' · low evidence'}</span></div>
                  </button>
                </li>
              ))}
              {!searching && hits.length === 0 && <li className="none">No fighters match “{q.trim()}”.</li>}
            </ul>
          )}
        </div>
        <div className="badge" role="status">{badge}</div>
      </header>
      {route === 'compare' && (
        <main className="stage wide reveal">
          <Compare picks={picks} onRemove={(id) => setPicks(picks.filter((p) => p.id !== id))} onClear={() => setPicks([])} />
        </main>
      )}
      {route === 'map' && (
        <main className="stage wide reveal">
          <StyleMap selectedId={selectedId} matchIds={similar?.results.map((m) => m.fighter_id) ?? []} onSelect={choose} />
        </main>
      )}
      {route === 'replay' && (
        <main className="stage wide reveal">
          <Replay fighterId={selectedId} fighterName={profile?.fighter_name} />
        </main>
      )}
      {route !== 'compare' && route !== 'map' && route !== 'replay' && (<>
      <main className="stage reveal">
        {statusErr && (
          <div className="alert" role="alert">
            {statusErr} <button onClick={loadStatus}>Retry</button>
          </div>
        )}
        {error && <div className="alert" role="alert">{error}</div>}
        {!profile && !loading && !statusErr && (
          <Home onPick={choose} />
        )}
        {profile && (
          <>
            <div className="fhead">
              <Avatar id={profile.fighter_id} name={profile.fighter_name} size={84} />
              <h1>{profile.fighter_name}</h1>
              <p className="meta">
                {profile.weight_class ?? 'Unknown division'}
                {profile.most_recent_weight_class && profile.most_recent_weight_class !== profile.weight_class
                  ? ` (most recent: ${profile.most_recent_weight_class})` : ''}
                {' · '}{profile.fights} fights · {profile.minutes} min · {profile.first_fight} to {profile.last_fight}
              </p>
              <p className="meta">Style group: {profile.group}</p>
              {profile.evidence_warning && <p className="warn">{profile.evidence_warning}</p>}
              <button className="mini inline" onClick={() => togglePick(profile.fighter_id, profile.fighter_name)}>
                {picks.some((p) => p.id === profile.fighter_id) ? 'Remove from compare' : 'Add to compare'}
              </button>
              <a className="mini inline" href="#replay" style={{ marginLeft: 8, textDecoration: 'none' }}>Past fights</a>
              {pickMsg && <p className="warn" role="status">{pickMsg}</p>}
            </div>
            <StyleReel key={profile.fighter_id} profile={profile} />
          </>
        )}
      </main>
      <aside className="side reveal" aria-label="Closest matches">
        <h2>Closest matches</h2>
        {!profile && <p className="note">Matches appear here once you pick a fighter.</p>}
        {profile && (
          <div className="filters">
            <label>Division
              <select value={division} onChange={(e) => setDivision(e.target.value)}>
                <option value="">Any division</option>
                <option value="same">Same as {profile.fighter_name?.split(' ')[0]}</option>
                {divisions.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
              </select>
            </label>
            <label>Men / women
              <select value={category} onChange={(e) => setCategory(e.target.value)}>
                <option value="">Everyone</option>
                <option value="same">Same as {profile.fighter_name?.split(' ')[0]}</option>
                <option value="men">Men's divisions</option>
                <option value="women">Women's divisions</option>
              </select>
            </label>
            <label>Minimum fight time
              <select value={minMinutes} onChange={(e) => setMinMinutes(e.target.value)}>
                <option value="">Any (15+ min)</option>
                <option value="50">50+ minutes</option>
                <option value="100">100+ minutes</option>
                <option value="200">200+ minutes</option>
              </select>
            </label>
          </div>
        )}
        {similar?.empty_reason && (
          <div className="alert">{similar.empty_reason} <button onClick={resetFilters}>Reset filters</button></div>
        )}
        <ol className="matches">
          {similar?.results.map((m) => (
            <li key={m.fighter_id}>
              <button className="match" onClick={() => choose(m.fighter_id)}>
                <span className="rank">{m.rank}</span>
                <span className="mname"><Avatar id={m.fighter_id} name={m.fighter_name} size={24} />{m.fighter_name}</span>
                <span className="mmeta">{m.weight_class ?? ''} · {m.fights} fights · {m.minutes} min</span>
                <span className="mscore">distance {m.distance} · index {m.similarity_index}{m.same_group ? ' · same group' : ''}</span>
                {reason(m).map((line) => <span className="mreason" key={line}>{line}</span>)}
              </button>
              <button className="mini" onClick={() => togglePick(m.fighter_id, m.fighter_name)}>
                {picks.some((p) => p.id === m.fighter_id) ? 'Added' : 'Compare'}
              </button>
            </li>
          ))}
        </ol>
        {similar && (
          <p className="note">
            Lower distance means more similar. The index is a fixed 0–100 scale, not a probability.
            Compared with {similar.comparison_cohort.candidates_after_filters} of {similar.comparison_cohort.size} fighters.{' '}
            {similar.limits}
          </p>
        )}
      </aside>
      </>)}
    </div>
  )
}
