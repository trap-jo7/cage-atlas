import Reel, { type Side } from './Reel'
import type { Profile } from './api'

export { octagon } from './Reel'

const val = (p: Profile, f: string) => p.features.find((x) => x.feature === f)?.value ?? null
const num = (v: number | null, d = 2) => (v == null ? 'undefined' : v.toFixed(d))
const pct = (v: number | null) => (v == null ? 'undefined' : `${Math.round(v * 100)}%`)

export default function StyleReel({ profile }: { profile: Profile }) {
  const landed = val(profile, 'sig_landed_per_min')
  const absorbed = val(profile, 'sig_absorbed_per_min')
  const accuracy = val(profile, 'sig_accuracy')
  const td = val(profile, 'td_attempted_per15')
  const ctrl = val(profile, 'ctrl_share')
  const strikes = Math.round((landed ?? 0) * 5)
  const incoming = Math.round((absorbed ?? 0) * 5)
  const a: Side = { name: profile.fighter_name ?? 'Fighter', strikes, shots: Math.round((td ?? 0) / 3), control: ctrl ?? 0 }
  const b: Side = { name: 'Typical opponent', strikes: incoming, shots: 0, control: null }

  return (
    <div className="reel">
      <Reel key={profile.fighter_id} a={a} b={b} opponent />
      <div className="reel-side">
        <ul className="facts">
          <li><span>Strikes landed</span><b>{num(landed)} /min</b><i>about {strikes} per 5-min round</i></li>
          <li><span>Strikes absorbed</span><b>{num(absorbed)} /min</b><i>about {incoming} per round (thrown by the grey opponent)</i></li>
          <li><span>Striking accuracy</span><b>{pct(accuracy)}</b></li>
          <li><span>Takedown attempts</span><b>{num(td)} /15 min</b><i>about {((td ?? 0) / 3).toFixed(1)} per round</i></li>
          <li><span>Control time</span><b>{pct(ctrl)}</b><i>of fight time</i></li>
        </ul>
        <p className="note">
          Statistical schematic, not footage. One 8-second play stands for a 5-minute round at this fighter's
          career averages: the blue fighter throws strikes and shoots takedowns as often as he really does, and the
          grey opponent throws as many strikes as this fighter typically absorbs.
        </p>
      </div>
    </div>
  )
}
