import { useEffect, useState } from 'react'
import { api, type FightsData } from './api'
import Reel, { type Side } from './Reel'

type Fight = FightsData['fights'][number]
type Stats = Fight['fighter']
const RESULT: Record<string, { tag: string; color: string }> = {
  win: { tag: 'W', color: '#63b3ff' },
  loss: { tag: 'L', color: '#ff9a8a' },
  draw_nc_or_unknown: { tag: 'D', color: '#aab8c7' },
}
const clock = (s: number | null) => (s == null ? '–' : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`)
const pair = (a: number | null, b: number | null) => `${a ?? '–'} of ${b ?? '–'}`

type Props = { fighterId: string | null; fighterName: string | null | undefined }

export default function Replay({ fighterId, fighterName }: Props) {
  const [data, setData] = useState<FightsData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [boutId, setBoutId] = useState<string | null>(null)

  useEffect(() => {
    if (!fighterId) return
    let live = true
    setData(null); setBoutId(null); setError(null)
    api.fights(fighterId)
      .then((d) => { if (live) setData(d) })
      .catch((e: Error) => { if (live) setError(e.message) })
    return () => { live = false }
  }, [fighterId])

  if (!fighterId) {
    return (
      <div>
        <h1>Fight replay</h1>
        <p className="note">Pick a fighter on <a href="#scout">Scout</a> first, then come back to replay their past fights.</p>
      </div>
    )
  }

  const name = data?.fighter.fighter_name ?? fighterName ?? 'Fighter'
  const fight = data?.fights.find((f) => f.bout_id === boutId) ?? null
  const opp = fight?.opponent_name ?? 'Opponent'
  // Keep both fighters in proportion when a fight has many strikes.
  const scale = fight ? Math.max(1, Math.ceil(Math.max(fight.fighter.sig_landed ?? 0, fight.opponent.sig_landed ?? 0) / 40)) : 1
  const side = (f: Fight, who: string, s: Stats): Side => ({
    name: who,
    strikes: Math.round((s.sig_landed ?? 0) / scale),
    shots: s.td_attempted ?? 0,
    control: s.ctrl_sec == null || !f.duration_sec ? null : s.ctrl_sec / f.duration_sec,
  })
  const row = (label: string, f: (s: Stats) => string) => (
    <tr><td>{label}</td><td>{f(fight!.fighter)}</td><td>{f(fight!.opponent)}</td></tr>
  )

  return (
    <div>
      <h1>Fight replay</h1>
      <p className="meta">{name}'s past fights, newest first. Pick one to replay it.</p>
      {error && <div className="alert" role="alert">{error}</div>}
      {!data && !error && <p className="note">Loading fights…</p>}
      {data && (
        <div style={{ display: 'grid', gap: 24, gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', alignItems: 'start', marginTop: 16 }}>
          <ol className="matches" style={{ maxHeight: 640, overflowY: 'auto' }}>
            {data.fights.map((f) => {
              const r = RESULT[f.result] ?? RESULT.draw_nc_or_unknown
              return (
                <li key={f.bout_id}>
                  <button className="match" aria-pressed={f.bout_id === boutId} onClick={() => setBoutId(f.bout_id)}>
                    <span className="rank" style={{ color: r.color }}>{r.tag}</span>
                    <span className="mname">vs {f.opponent_name ?? 'Unknown opponent'}</span>
                    <span className="mmeta">{f.date} · {f.weight_class ?? ''}{f.is_title_fight ? ' · title fight' : ''}</span>
                    <span className="mscore">{f.method}, round {f.round}{f.finish_time ? ` ${f.finish_time}` : ''}</span>
                  </button>
                </li>
              )
            })}
          </ol>
          <div>
            {!fight && <p className="note">Choose a fight on the left.</p>}
            {fight && (
              <>
                <h2>{name} vs {opp}</h2>
                <p className="meta">{fight.date} · {fight.weight_class ?? ''}{fight.is_title_fight ? ' · title fight' : ''}</p>
                <p>{fight.how_won}</p>
                {fight.stats_complete && (
                  <>
                    <div style={{ maxWidth: 560, margin: '12px 0' }}>
                      <Reel key={fight.bout_id} a={side(fight, name, fight.fighter)} b={side(fight, opp, fight.opponent)} />
                    </div>
                    <div className="tablewrap">
                      <table>
                        <thead><tr><th></th><th>{name}</th><th>{opp}</th></tr></thead>
                        <tbody>
                          {row('Significant strikes landed', (s) => pair(s.sig_landed, s.sig_attempted))}
                          {row('Takedowns landed', (s) => pair(s.td_landed, s.td_attempted))}
                          {row('Knockdowns', (s) => String(s.kd ?? '–'))}
                          {row('Control time', (s) => clock(s.ctrl_sec))}
                        </tbody>
                      </table>
                    </div>
                    {scale > 1 && (
                      <p className="note">Each punch in the animation stands for about {scale} real strikes, so the two fighters stay in proportion.</p>
                    )}
                  </>
                )}
              </>
            )}
            <p className="note">{data.note}</p>
          </div>
        </div>
      )}
      {data && data.total_fights > data.fights.length && (
        <p className="note">Showing the latest {data.fights.length} of {data.total_fights} fights.</p>
      )}
    </div>
  )
}
