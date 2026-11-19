# AfriLingua web

The browser frontend for AfriLingua v2: landing page, live interpretation (host console, listener view, projector mode), text and document translation, the community contribution hub, translation memory search and engine status.

Vite + React 19 + TypeScript (strict), Tailwind CSS v4, React Router, TanStack Query, Motion and Lucide icons. Fully offline: fonts are self-hosted through `@fontsource`, and there are no CDN scripts or remote images.

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
pnpm run build      # outputs web/dist, served by FastAPI at / with SPA fallback
pnpm run preview    # http://localhost:4173, same proxy as dev
```

## Layout

| Path | What it is |
| --- | --- |
| `src/lib/api.ts` | Typed client for `docs/api-contract.md` (REST, WebSocket URLs, room events) |
| `src/lib/audio.ts` | Gapless WAV queue player, mic capture via AudioWorklet, recorder helpers |
| `public/pcm-worklet.js` | AudioWorklet: mono, 16 kHz, PCM16 little-endian, ~40 ms frames |
| `src/lib/socket.ts` | Listener WebSocket with exponential backoff and jitter |
| `src/pages/` | Route components, lazy-loaded except the landing page |
| `src/components/` | App shell, brand marks and textile motifs, UI primitives |

## Live rooms on a local network

Phones cannot open `localhost`. The host console builds the QR code from the current origin; when that origin is loopback it uses the server's `join_url` instead (expected to carry the laptop's LAN address), and shows a warning if neither is reachable from a phone.

Microphone capture needs a secure context: `localhost` works, LAN IPs over plain HTTP do not in most browsers. So run the host console on the laptop itself (localhost) and let listeners use the LAN address from the QR code; listeners never need the microphone.
