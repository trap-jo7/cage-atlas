import { useEffect, useRef } from 'react'
import { createScope, createTimeline } from 'animejs'

const ROUND_MS = 8000 // one 5-minute round, compressed

export function octagon(r: number, cx = 200, cy = 200) {
  return Array.from({ length: 8 }, (_, k) => {
    const a = (Math.PI / 4) * k + Math.PI / 8
    return `${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`
  }).join(' ')
}

export type Side = { name: string; strikes: number; shots: number; control: number | null }
type C = { main: string; rear: string }
const BLUE: C = { main: '#63b3ff', rear: '#4a8fd0' }
const GOLD: C = { main: '#e8c67a', rear: '#c9a85c' }
const GREY: C = { main: '#3a4d63', rear: '#2e3f52' }

const L = (pts: string, w: number, c: string) => <polyline points={pts} stroke={c} strokeWidth={w} />
const dot = (x: number, y: number, r: number, c: string) => <circle cx={x} cy={y} r={r} fill={c} stroke="none" />

function Pose({ kind, c }: { kind: 'guard' | 'strike' | 'shot'; c: C }) {
  const body =
    kind === 'guard' ? (
      <>
        {L('-4,-82 -20,-44 -32,0', 10, c.rear)}{L('-2,-130 12,-104 28,-122', 8, c.rear)}
        {L('0,-132 -2,-84', 16, c.main)}{L('2,-82 22,-44 32,0', 10, c.main)}
        {dot(4, -150, 12, c.main)}{L('4,-130 24,-108 40,-130', 8, c.main)}
        {dot(40, -130, 6, c.main)}{dot(28, -122, 6, c.rear)}
      </>
    ) : kind === 'strike' ? (
      <>
        {L('-4,-82 -24,-42 -38,0', 10, c.rear)}{L('2,-130 14,-108 28,-126', 8, c.rear)}
        {L('6,-132 -2,-84', 16, c.main)}{L('2,-82 30,-44 40,0', 10, c.main)}
        {dot(12, -148, 12, c.main)}{L('8,-130 36,-132 62,-132', 8, c.main)}
        {dot(62, -132, 7, c.main)}{dot(28, -126, 6, c.rear)}
      </>
    ) : (
      <>
        {L('-16,-70 -36,-30 -62,-2', 10, c.rear)}{L('20,-92 42,-80 64,-62', 8, c.rear)}
        {L('22,-92 -14,-70', 16, c.main)}{L('-12,-70 30,-40 34,0', 10, c.main)}
        {dot(36, -102, 12, c.main)}{L('24,-90 46,-70 68,-50', 8, c.main)}
        {dot(68, -50, 6, c.main)}{dot(64, -62, 6, c.rear)}
      </>
    )
  return <g fill="none" strokeLinecap="round" strokeLinejoin="round">{body}</g>
}

function Fighter({ id, x, dir, c }: { id: string; x: number; dir: 1 | -1; c: C }) {
  return (
    <g transform={`translate(${x},250) scale(${dir},1)`}>
      <g className={`${id}-body`}>
        <g className={`${id}-guard`}><Pose kind="guard" c={c} /></g>
        <g className={`${id}-strike`} style={{ opacity: 0 }}><Pose kind="strike" c={c} /></g>
        <g className={`${id}-shot`} style={{ opacity: 0 }}><Pose kind="shot" c={c} /></g>
        <circle className={`${id}-spark`} cx="76" cy="-132" r="4" fill="#e8c67a" style={{ opacity: 0 }} />
      </g>
    </g>
  )
}

// Lay strikes and takedown shots along the round without overlapping each other.
function layout(strikes: number, shots: number) {
  const s = Math.min(strikes, 40)
  const t = Math.min(shots, 10)
  const events = [
    ...Array.from({ length: s }, (_, k) => ({ kind: 'strike' as const, p: (k + 0.5) / s, d: 160 })),
    ...Array.from({ length: t }, (_, k) => ({ kind: 'shot' as const, p: (k + 0.5) / t, d: 620 })),
  ].sort((a, b) => a.p - b.p)
  const out: { kind: 'strike' | 'shot'; start: number; d: number }[] = []
  let free = 0
  for (const e of events) {
    const start = Math.max(free, e.p * ROUND_MS - e.d / 2)
    if (start + e.d > ROUND_MS) break
    out.push({ kind: e.kind, start, d: e.d })
    free = start + e.d + 40
  }
  return out
}

type Props = { a: Side; b: Side; opponent?: boolean }

export default function Reel({ a, b, opponent = false }: Props) {
  const root = useRef<HTMLDivElement>(null)
  const scope = useRef<ReturnType<typeof createScope> | null>(null)
  const tlRef = useRef<ReturnType<typeof createTimeline> | null>(null)
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const sig = [a.strikes, a.shots, a.control, b.strikes, b.shots, b.control].join(',')

  useEffect(() => {
    scope.current = createScope({ root }).add((self) => {
      self?.add('play', () => {
        tlRef.current?.revert()
        const tl = createTimeline()
        tlRef.current = tl
        ;([['a', a], ['b', b]] as const).forEach(([id, side]) => {
          layout(side.strikes, side.shots).forEach((e) => {
            tl.add(`.${id}-guard`, { opacity: [{ to: 0, duration: 10 }, { to: 0, duration: e.d - 20 }, { to: 1, duration: 10 }] }, e.start)
            tl.add(`.${id}-${e.kind}`, { opacity: [{ to: 1, duration: 10 }, { to: 1, duration: e.d - 20 }, { to: 0, duration: 10 }] }, e.start)
            if (e.kind === 'strike')
              tl.add(`.${id}-spark`, { translateX: [0, 40], opacity: [{ to: 1, duration: 10 }, { to: 0, duration: 130 }], duration: 150, ease: 'out(2)' }, e.start + 20)
            else
              tl.add(`.${id}-body`, { translateX: [{ to: 45, duration: 280 }, { to: 0, duration: 320 }], ease: 'inOut(2)' }, e.start)
          })
          if (side.control != null) tl.add(`.${id}-ctrl`, { scaleX: [0, side.control], duration: ROUND_MS, ease: 'linear' }, 0)
        })
      })
    })
    return () => scope.current?.revert()
  }, [sig]) // eslint-disable-line react-hooks/exhaustive-deps

  const bar = (id: string, x: number, v: number | null) =>
    v == null ? null : (
      <>
        <text x={x} y="288" className="svgtext">Control time</text>
        <rect className="ctrl-track" x={x} y="294" width="150" height="8" rx="4" />
        <rect className={`ctrl-fill ${id}-ctrl`} x={x} y="294" width="150" height="8" rx="4"
          style={{ transformBox: 'fill-box', transformOrigin: 'left center', transform: `scaleX(${v})` }} />
      </>
    )

  return (
    <div className="reelbox" ref={root} style={{ display: 'grid', gap: 12 }}>
      <svg viewBox="0 0 500 330" role="img" aria-label={`Statistical schematic: ${a.name} and ${b.name} acting out their measured striking, takedown and control rates`}>
        <polygon className="oct" points={octagon(160, 250, 165)} />
        <line x1="90" y1="250" x2="410" y2="250" stroke="#2e3f52" strokeWidth="2" />
        <Fighter id="a" x={170} dir={1} c={BLUE} />
        <Fighter id="b" x={330} dir={-1} c={opponent ? GREY : GOLD} />
        <text x="170" y="62" textAnchor="middle" className="svgtext">{a.name}</text>
        <text x="330" y="62" textAnchor="middle" className="svgtext">{b.name}</text>
        {bar('a', 70, a.control)}{bar('b', 280, b.control)}
      </svg>
      <button className="play" onClick={() => scope.current?.methods.play()} disabled={reduce} style={{ justifySelf: 'start' }}>
        Play a 5-minute round
      </button>
      {reduce && <p className="note">Reduced motion is on, so the animation is off. The numbers carry the same information.</p>}
    </div>
  )
}
