export type Status = { data_mode_label: string; organizations: { name: string; data_through: string | null }[] }
export type Hit = { fighter_id: string; fighter_name: string | null; weight_class: string | null; fights: number; in_model_cohort: boolean }
export type Feature = { feature: string; label: string; unit: string; value: number | null; undefined: boolean }
export type Profile = {
  fighter_id: string; fighter_name: string | null; weight_class: string | null
  most_recent_weight_class: string | null; fights: number; minutes: number
  first_fight: string; last_fight: string; group: string; evidence_warning: string | null
  features: Feature[]
}
export type Explain = { label: string; unit: string; query_value: number | null; match_value: number | null }
export type Match = {
  rank: number; fighter_id: string; fighter_name: string | null; weight_class: string | null
  distance: number; similarity_index: number; same_group: boolean; fights: number; minutes: number
  closest_characteristics: Explain[]; largest_differences: Explain[]
}
export type Similar = {
  results: Match[]; empty_reason: string | null; limits: string
  comparison_cohort: { candidates_after_filters: number; size: number }
}
export type Division = { name: string; fighters: number }
export type RecentData = {
  data_through: string | null; latest_event_date: string; note: string
  results: { bout_id: string; a_id: string; a_name: string | null; b_id: string; b_name: string | null; outcome: string
    method: string | null; round: number; finish_time: string | null; weight_class: string | null; is_title_fight: boolean }[]
  fighters: { fighter_id: string; fighter_name: string | null; weight_class: string | null; last_fight: string; group: string; fights: number }[]
}
export type FightStats = { sig_landed: number | null; sig_attempted: number | null; td_landed: number | null; td_attempted: number | null; kd: number | null; ctrl_sec: number | null }
export type FightsData = {
  fighter: { fighter_id: string; fighter_name: string | null }; total_fights: number; note: string
  fights: {
    bout_id: string; date: string; opponent_id: string; opponent_name: string | null; weight_class: string | null
    result: string; method: string | null; round: number; finish_time: string | null; is_title_fight: boolean
    duration_sec: number | null; stats_complete: boolean; fighter: FightStats; opponent: FightStats; how_won: string
  }[]
}
export type MapPoint = { fighter_id: string; fighter_name: string | null; weight_class: string | null; cluster: number; group: string; x: number; y: number }
export type MapData = {
  points: MapPoint[]; note: string
  axes: { explained_variance_ratio: number[]; explained_variance_total: number | null }
  groups: { cluster: number; name: string; size: number }[]
}
export type CompareData = {
  fighters: { fighter_id: string; fighter_name: string | null; weight_class: string | null; fights: number; minutes: number; evidence_warning: string | null; undefined_features: string[] }[]
  metrics: { feature: string; label: string; unit: string; role: string; higher_means: string
    values: { fighter_id: string; value: number | null; percentile: number | null; undefined: boolean }[] }[]
  reference_cohort: { name: string; size: number }
  limits: string
}
export const compareQuery = (ids: string[]) => ids.map((i) => `ids=${encodeURIComponent(i)}`).join('&')

type Params = Record<string, string | number | boolean | undefined>

async function get<T>(path: string, params: Params = {}, rawQuery = ''): Promise<T> {
  const qs = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== '') qs.set(k, String(v))
  const query = [qs.toString(), rawQuery].filter(Boolean).join('&')
  let res: Response
  try {
    res = await fetch(`/api${path}${query ? `?${query}` : ''}`)
  } catch {
    throw new Error("Can't reach the CAGE ATLAS server. Start it with: uvicorn backend.main:app --port 8000")
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try { detail = (await res.json()).detail ?? detail } catch { /* keep default */ }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  status: () => get<Status>('/status'),
  map: () => get<MapData>('/map'),
  recent: () => get<RecentData>('/recent'),
  fights: (id: string) => get<FightsData>(`/fighters/${id}/fights`),
  compare: (ids: string[]) => get<CompareData>('/compare', {}, compareQuery(ids)),
  divisions: () => get<{ divisions: Division[] }>('/divisions'),
  search: (q: string) => get<Hit[]>('/fighters/search', { q }),
  profile: (id: string) => get<Profile>(`/fighters/${id}`),
  similar: (id: string, p: Params) => get<Similar>(`/fighters/${id}/similar`, p),
}
