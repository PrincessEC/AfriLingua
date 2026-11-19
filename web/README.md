# AfriLingua web

The browser frontend for AfriLingua v2: landing page, live interpretation (host console, listener view, projector mode), text and document translation, the community contribution hub, translation memory search and engine status.

Vite + React 19 + TypeScript (strict), Vike (vike-react, prerendered to static HTML), Tailwind CSS v4, TanStack Query, Motion and Lucide icons. Fully offline: fonts are self-hosted through `@fontsource`, and there are no CDN scripts or remote images.

## Develop

```bash
cd web
pnpm install
pnpm dev            # http://localhost:5173
```

The dev server proxies `/api`, `/v1` and `/mcp` (including WebSockets) to `http://127.0.0.1:8000`. Start the engine in another terminal with `afrilingua serve`. Point the proxy elsewhere with `AFRILINGUA_ENGINE_URL=http://host:port pnpm dev`.

Every page works without the engine: it shows a "Start the engine: `afrilingua serve`" state instead of failing, and recovers automatically once the health check (polled every 10 s) succeeds.

## Build

```bash
pnpm run typecheck
pnpm run build      # typecheck + vite build + Vike prerender -> web/dist/client
pnpm run preview    # vike preview on http://localhost:4173 (same /api proxy as dev)
```

The site is **statically prerendered** (Vike SSG); no Node runs in production. FastAPI serves `web/dist/client` (`afrilingua serve`). `dist/server` is build-time only and is not deployed.

`pnpm run preview` serves only the prerendered files, so use `/room/_` (not `/room/<id>`) to try the room page there; FastAPI maps every `/room/<id>` to that template (see below).

### Routing and prerendering

| Route | Rendering |
| --- | --- |
| `/` | Prerendered with full SSR HTML (the SEO page): headline, sections, title, description, canonical, Open Graph and Twitter tags |
| `/translate` `/documents` `/live` `/memory` `/contribute` `/status` | Prerendered HTML shell with its own `<title>` and meta description; the page itself renders on the client (`ssr: false`, uses localStorage, mic, WebSocket) |
| `/room/<id>` | Client-only, no site chrome. One template, `/room/_`, is prerendered (`pages/room/+onBeforePrerenderStart.ts`); the backend serves it for any `/room/<id>` and `Room.tsx` reads the real id from `window.location` |
| anything else | Vike `_error` page, prerendered as `404.html` and served with HTTP 404 |

Backend rules (directory index, `/room/<id>` template, `404.html`) are in the FastAPI SPA handler (`afrilingua/api/app.py`).

### SEO notes

- Per-page `title` / `description` live in each `pages/**/+config.ts`; shared tags (theme script, canonical, Open Graph) in `pages/+Head.tsx`. There is no `og:image` yet.
- `public/robots.txt` allows everything and points to `public/sitemap.xml`.
- **The origin `https://afrilingua.dev` is a placeholder.** Replace it in `src/lib/site.ts`, `public/sitemap.xml` and `public/robots.txt` before going public.
- Add new public pages to `public/sitemap.xml` by hand.

## Layout

| Path | What it is |
| --- | --- |
| `src/lib/api.ts` | Typed client for `docs/api-contract.md` (REST, WebSocket URLs, room events) |
| `src/lib/audio.ts` | Gapless WAV queue player, mic capture via AudioWorklet, recorder helpers |
| `public/pcm-worklet.js` | AudioWorklet: mono, 16 kHz, PCM16 little-endian, ~40 ms frames |
| `src/lib/socket.ts` | Listener WebSocket with exponential backoff and jitter |
| `pages/` | Vike routes: `+config.ts` (global), `+Wrapper.tsx` (providers), `+Head.tsx`, `(site)/` (pages with nav and footer), `room/` (no chrome), `_error/` (404) |
| `src/pages/` | Page components that the Vike routes re-export |
| `src/lib/nav.ts` | Routing helpers on top of Vike (`usePathname`, `useSearchParams`) |
| `src/components/` | App shell, brand marks and textile motifs, UI primitives |

## Live rooms on a local network

Phones cannot open `localhost`. The host console builds the QR code from the current origin; when that origin is loopback it uses the server's `join_url` instead (expected to carry the laptop's LAN address), and shows a warning if neither is reachable from a phone.

Microphone capture needs a secure context: `localhost` works, LAN IPs over plain HTTP do not in most browsers. So run the host console on the laptop itself (localhost) and let listeners use the LAN address from the QR code; listeners never need the microphone.
