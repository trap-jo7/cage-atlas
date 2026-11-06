import { useEffect, useState } from 'react'
import { api, compareQuery, type CompareData } from './api'
import Reel, { type Side } from './Reel'

export type Picked = { id: string; name: string }
const COLORS = ['#63b3ff', '#e8c67a', '#7fd0c4', '#ff9a8a']
const fmt = (v: number | null) => (v == null ? 'undefined' : String(v))

const sideOf = (d: CompareData, i: number): Side => {
  const get = (f: string) => d.metrics.find((m) => m.feature === f)?.values[i]?.value ?? 0
  return {
    name: d.fighters[i].fighter_name ?? 'Fighter',
    strikes: Math.round(get('sig_landed_per_min') * 5),
    shots: Math.round(get('td_attempted_per15') / 3),
    control: get('ctrl_share'),
  }
}

type Props = { picks: Picked[]; onRemove: (id: string) => void; onClear: () => void }

export default function Compare({ picks, onRemove, onClear }: Props) {
  const [data, setData] = useState<CompareData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const ids = picks.map((p) => p.id)
  const key = ids.join(',')

  useEffect(() => {
    if (ids.length < 2) { setData(null); return }
    let live = true
    setLoading(true); setError(null)
    api.compare(ids)
      .then((d) => { if (live) setData(d) })
      .catch((e: Error) => { if (live) setError(e.message) })
      .finally(() => { if (live) setLoading(false) })
    return () => { live = false }
  }, [key]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="compare">
      <h1>Compare fighters</h1>
      <div className="chips">
        {picks.map((p, i) => (
          <span className="chip" key={p.id}>
            <i style={{ background: COLORS[i] }} />{p.name}
            <button onClick={() => onRemove(p.id)} aria-label={`Remove ${p.name}`}>×</button>
          </span>
        ))}
        {picks.length > 0 && <button className="linkbtn" onClick={onClear}>Clear all</button>}
      </div>
      {picks.length < 2 && (
        <p className="note">Pick two to four fighters: open Scout, choose a fighter, and press “Add to compare” (matches have a Compare button too).</p>
      )}
      {error && <div className="alert" role="alert">{error}</div>}
      {loading && <p className="note">Loading comparison…</p>}
      {data && picks.length >= 2 && (
        <>
          <div className="evidence">
            {data.fighters.map((f, i) => (
              <div key={f.fighter_id}>
                <b style={{ color: COLORS[i] }}>{f.fighter_name}</b>
                <span>{f.weight_class ?? 'Unknown division'} · {f.fights} fights · {f.minutes} min</span>
                {f.undefined_features.length > 0 && <span>Undefined (no attempts): {f.undefined_features.join(', ')}</span>}
                {f.evidence_warning && <span className="warn">{f.evidence_warning}</span>}
              </div>
            ))}
          </div>
          <section className="metric">
            <h2>Style reel <small>{data.fighters[0].fighter_name} (blue) and {data.fighters[1].fighter_name} (gold)</small></h2>
            <div style={{ maxWidth: 560, margin: '12px 0' }}>
              <Reel key={key} a={sideOf(data, 0)} b={sideOf(data, 1)} />
            </div>
            <p className="note">
              Each fighter acts out their own career rates: strikes thrown, takedown shots, and share of the round in control.
              This shows how the two styles differ; it is not a prediction of how they would fight each other.
              {data.fighters.length > 2 ? ' The animation shows the first two fighters you picked.' : ''}
            </p>
          </section>
          {data.metrics.map((m) => {
            const vals = m.values.map((v) => v.value ?? 0)
            const max = Math.max(...vals, 1e-9)
            const share = m.unit.startsWith('share')
            return (
              <section className="metric" key={m.feature}>
                <h2>{m.label} <small>{m.unit} · {m.role}</small></h2>
                <p className="note">{m.higher_means}</p>
                {m.values.map((v, i) => (
                  <div className="mrow" key={v.fighter_id}>
                    <span>{data.fighters[i].fighter_name}</span>
                    <div className="bar"><i style={{ width: `${Math.max(1, ((v.value ?? 0) / (share ? 1 : max)) * 100)}%`, background: COLORS[i] }} /></div>
                    <b>{fmt(v.value)}</b>
                    <em>{v.percentile == null ? '' : `percentile ${Math.round(v.percentile)}`}</em>
                  </div>
                ))}
              </section>
            )
          })}
          <p className="note">
            Percentile = share of the {data.reference_cohort.size} fighters in the {data.reference_cohort.name} at or below this value. {data.limits}
          </p>
          <a className="play dl" href={`/api/compare.csv?${compareQuery(ids)}`} download>Download CSV</a>
        </>
      )}
    </div>
  )
}
