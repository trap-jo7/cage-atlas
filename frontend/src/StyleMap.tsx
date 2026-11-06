import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type MapData, type MapPoint } from './api'

const SHAPES = ['circle', 'rect', 'triangle', 'diamond', 'pin', 'arrow']
const COLORS = ['#63b3ff', '#e8c67a', '#7fd0c4', '#ff9a8a', '#b8c4d0', '#9fb4ff']
const pct = (v?: number) => (v == null ? '?' : `${Math.round(v * 100)}%`)

type Props = { selectedId: string | null; matchIds: string[]; onSelect: (id: string) => void }

export default function StyleMap({ selectedId, matchIds, onSelect }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const [data, setData] = useState<MapData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState('')
  const matchKey = matchIds.join(',')

  const load = () => { setError(null); api.map().then(setData).catch((e: Error) => setError(e.message)) }
  useEffect(load, [])

  useEffect(() => {
    if (!data || !box.current) return
    let cancelled = false
    let chart: { dispose(): void; resize(): void } | undefined
    const el = box.current
    // The chart library is loaded only when this page opens.
    import('echarts').then((echarts) => {
      if (cancelled) return
      const c = echarts.init(el)
      chart = c
      const hi = new Set([selectedId, ...matchIds].filter(Boolean) as string[])
      const pt = (p: MapPoint) => [p.x, p.y, p.fighter_id, p.fighter_name, p.weight_class, p.group]
      const ratio = data.axes.explained_variance_ratio
      const axis = (name: string) => ({
        name, nameLocation: 'middle', nameGap: 28, nameTextStyle: { color: '#aab8c7' },
        axisLabel: { color: '#aab8c7' }, axisLine: { lineStyle: { color: '#2e3f52' } },
        splitLine: { lineStyle: { color: '#1b2836' } },
      })
      const option = {
        animation: !window.matchMedia('(prefers-reduced-motion: reduce)').matches,
        animationDuration: 400,
        grid: { left: 56, right: 16, top: 16, bottom: 96 },
        tooltip: {
          trigger: 'item',
          formatter: (p: unknown) => {
            const d = (p as { data: unknown[] }).data
            return `${d[3]}<br/>${d[4] ?? ''}<br/>${d[5]}`
          },
        },
        legend: { type: 'scroll', bottom: 0, textStyle: { color: '#aab8c7' } },
        xAxis: axis(`Axis 1 (${pct(ratio[0])} of the variation)`),
        yAxis: axis(`Axis 2 (${pct(ratio[1])})`),
        series: [
          ...data.groups.map((g, i) => ({
            name: g.name, type: 'scatter', symbol: SHAPES[i % SHAPES.length], symbolSize: 7,
            itemStyle: { color: COLORS[i % COLORS.length], opacity: 0.75 },
            data: data.points.filter((p) => p.cluster === g.cluster).map(pt),
          })),
          {
            name: 'Selected fighter and closest matches', type: 'scatter', symbolSize: 16, z: 10,
            itemStyle: { color: 'rgba(0,0,0,0)', borderColor: '#f3f6fa', borderWidth: 2 },
            data: data.points.filter((p) => hi.has(p.fighter_id)).map(pt),
          },
        ],
      }
      c.setOption(option as never)
      c.on('click', (p) => {
        const d = (p as { data?: unknown[] }).data
        if (d && typeof d[2] === 'string') onSelect(d[2])
      })
    })
    const onResize = () => chart?.resize()
    window.addEventListener('resize', onResize)
    return () => { cancelled = true; window.removeEventListener('resize', onResize); chart?.dispose() }
  }, [data, selectedId, matchKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const rows = useMemo(() => {
    const f = filter.trim().toLowerCase()
    return (data?.points ?? []).filter((p) => (p.fighter_name ?? '').toLowerCase().includes(f)).slice(0, 60)
  }, [data, filter])

  return (
    <div className="map">
      <h1>Style map</h1>
      {error && <div className="alert" role="alert">{error} <button onClick={load}>Retry</button></div>}
      {data && <p className="note">{data.note} Click a dot to scout that fighter. White rings mark the selected fighter and their closest matches.</p>}
      <div ref={box} className="chart" role="img"
        aria-label="Scatter map of UFC fighters by measured style. A searchable table with the same fighters follows." />
      {data && (
        <>
          <h2>Find a fighter (table version)</h2>
          <label className="sr" htmlFor="mapfilter">Filter fighters by name</label>
          <input id="mapfilter" className="mapfilter" placeholder="Filter by name" value={filter}
            onChange={(e) => setFilter(e.target.value)} />
          <div className="tablewrap">
            <table>
              <caption className="note">First {rows.length} matching fighters. Type to narrow the list.</caption>
              <thead><tr><th>Fighter</th><th>Division</th><th>Style group</th><th>Axis 1</th><th>Axis 2</th></tr></thead>
              <tbody>
                {rows.map((p) => (
                  <tr key={p.fighter_id}>
                    <td><button className="linkbtn" onClick={() => onSelect(p.fighter_id)}>{p.fighter_name}</button></td>
                    <td>{p.weight_class}</td><td>{p.group}</td><td>{p.x}</td><td>{p.y}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
