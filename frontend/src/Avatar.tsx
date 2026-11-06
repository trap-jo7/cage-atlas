import { useEffect, useState } from 'react'

// Licensed headshots go in public/headshots/<fighter_id>.jpg and are listed in
// public/headshots/manifest.json. Without a photo, a clean initials badge is shown.
let manifest: Promise<Set<string>> | null = null
const loadManifest = () =>
  (manifest ??= fetch('/headshots/manifest.json')
    .then((r) => r.json())
    .then((j: { photos?: string[] }) => new Set(j.photos ?? []))
    .catch(() => new Set<string>()))

const TONES = ['#63b3ff', '#e8c67a', '#7fd0c4', '#ff9a8a', '#b8c4d0']
const tone = (id: string) => TONES[[...id].reduce((a, c) => a + c.charCodeAt(0), 0) % TONES.length]
const initials = (n: string | null) =>
  (n ?? '?').split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0].toUpperCase()).join('')

type Props = { id: string; name: string | null; size?: number }

export default function Avatar({ id, name, size = 40 }: Props) {
  const [photo, setPhoto] = useState(false)
  useEffect(() => {
    let live = true
    loadManifest().then((s) => { if (live) setPhoto(s.has(id)) })
    return () => { live = false }
  }, [id])
  const box = { width: size, height: size, fontSize: Math.round(size * 0.38) }
  if (photo)
    return <img className="avatar" src={`/headshots/${id}.jpg`} alt="" style={{ ...box, objectFit: 'cover' }} onError={() => setPhoto(false)} />
  return <span className="avatar" aria-hidden="true" style={{ ...box, color: tone(id), borderColor: tone(id) }}>{initials(name)}</span>
}
