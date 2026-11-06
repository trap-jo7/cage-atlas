import { useEffect, useState } from 'react'
import { api, type RecentData } from './api'
import Avatar from './Avatar'

const when = (iso: string) =>
  new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })

export default function Home({ onPick }: { onPick: (id: string) => void }) {
  const [data, setData] = useState<RecentData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const load = () => { setError(null); api.recent().then(setData).catch((e: Error) => setError(e.message)) }
  useEffect(load, [])

  return (
    <div className="home">
      <h1>Scout fighters by how they fight</h1>
      <p className="lede">Search any UFC fighter, watch their measured style play out, and find the fighters whose numbers look most like theirs.</p>
      {error && <div className="alert" role="alert">{error} <button onClick={load}>Retry</button></div>}
      {!data && !error && <p className="note">Loading the latest fights…</p>}
      {data && (
        <>
          <section aria-labelledby="latest">
            <h2 id="latest"><i className="dot gold" />Latest results <small>{when(data.latest_event_date)}</small></h2>
            <div className="cards">
              {data.results.map((r) => (
                <div className="rcard" key={r.bout_id}>
                  <div className="vs">
                    <button className="who" onClick={() => onPick(r.a_id)}><Avatar id={r.a_id} name={r.a_name} size={30} />{r.a_name}</button>
                    <em>{r.outcome === 'a_won' ? 'def.' : 'vs'}</em>
                    <button className="who" onClick={() => onPick(r.b_id)}><Avatar id={r.b_id} name={r.b_name} size={30} />{r.b_name}</button>
                  </div>
                  <p className="rmeta">
                    {r.outcome === 'a_won' ? '' : 'Draw or no contest · '}{r.method} · round {r.round}{r.finish_time ? ` ${r.finish_time}` : ''}
                    {r.weight_class ? ` · ${r.weight_class}` : ''}{r.is_title_fight ? ' · title fight' : ''}
                  </p>
                </div>
              ))}
            </div>
          </section>
          <section aria-labelledby="active">
            <h2 id="active"><i className="dot" />Recently active fighters</h2>
            <div className="tiles">
              {data.fighters.map((f) => (
                <button className="tile" key={f.fighter_id} onClick={() => onPick(f.fighter_id)}>
                  <Avatar id={f.fighter_id} name={f.fighter_name} size={48} />
                  <span className="tname">{f.fighter_name}</span>
                  <span className="tmeta">{f.weight_class ?? ''} · last fought {when(f.last_fight)}</span>
                  <span className="tmeta">{f.group.replace(/^Group [A-Z]: /, '')}</span>
                </button>
              ))}
            </div>
          </section>
          <p className="note">{data.note} Data through {data.data_through ?? 'an unknown date'}.</p>
        </>
      )}
    </div>
  )
}
