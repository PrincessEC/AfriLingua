import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { motion, useMotionValue, useTransform } from "motion/react";
import {
  Check,
  Copy,
  ExternalLink,
  Info,
  Mic,
  Projector,
  Radio,
  RotateCcw,
  Square,
  Timer,
  Users,
} from "lucide-react";
import {
  api,
  errorMessage,
  hostSocketUrl,
  isOffline,
  parseRoomEvent,
  type RoomCreated,
  type RoomEvent,
} from "../lib/api";
import { langLabel, langShort, useLanguages } from "../lib/languages";
import { MicCapture, listMicrophones } from "../lib/audio";
import { useHealth, useInterval } from "../lib/hooks";
import { useToast } from "../lib/toast";
import { cn, copyText, formatDuration } from "../lib/utils";
import { Button, Field, PageHeader, Pill, Select } from "../components/ui";
import { EngineOffline } from "../components/states";
import { LanguageSelect } from "../components/LanguageSelect";
import { QrCode } from "../components/QrCode";

// ─── Types & persistence ───────────────────────────────────────────────────

interface HostRoom extends RoomCreated {
  title: string;
  src_lang: string;
  targets: string[];
  pin?: string;
  createdAt: number;
}

interface CaptionLine {
  seq: number;
  text: string;
  final: boolean;
  translations: Record<string, { text: string; origin?: string; latency_ms?: number }>;
}

const ROOM_KEY = "afrilingua.hostRoom";

function loadRoom(): HostRoom | null {
  try {
    const raw = sessionStorage.getItem(ROOM_KEY);
    return raw ? (JSON.parse(raw) as HostRoom) : null;
  } catch {
    return null;
  }
}
function saveRoom(r: HostRoom | null) {
  try {
    if (r) sessionStorage.setItem(ROOM_KEY, JSON.stringify(r));
    else sessionStorage.removeItem(ROOM_KEY);
  } catch {
    /* ignore */
  }
}

const LOOPBACK = /^(localhost|127\.\d+\.\d+\.\d+|\[?::1\]?)$/i;

/** Join URL for phones. Prefer this origin; fall back to the server's LAN join_url when we're on loopback. */
function resolveJoinUrl(room: RoomCreated): { url: string; loopbackOnly: boolean } {
  const local = `${location.origin}/room/${encodeURIComponent(room.id)}`;
  if (!LOOPBACK.test(location.hostname)) return { url: local, loopbackOnly: false };
  try {
    const server = new URL(room.join_url, location.origin);
    if (!LOOPBACK.test(server.hostname)) {
      return { url: `${server.origin}/room/${encodeURIComponent(room.id)}`, loopbackOnly: false };
    }
  } catch {
    /* ignore malformed join_url */
  }
  return { url: local, loopbackOnly: true };
}

// ─── Page ──────────────────────────────────────────────────────────────────

export default function Live() {
  const [room, setRoom] = useState<HostRoom | null>(loadRoom);
  return (
    <div className="mx-auto max-w-[1400px] px-5 py-12 md:px-8 md:py-16">
      {room ? (
        <HostConsole
          room={room}
          onEnd={() => {
            saveRoom(null);
            setRoom(null);
          }}
        />
      ) : (
        <RoomSetup
          onCreated={(r) => {
            saveRoom(r);
            setRoom(r);
          }}
        />
      )}
    </div>
  );
}

// ─── Setup ─────────────────────────────────────────────────────────────────

function RoomSetup({ onCreated }: { onCreated: (r: HostRoom) => void }) {
  const { languages } = useLanguages();
  const { online, checking, refetch } = useHealth();
  const toast = useToast();
  const [title, setTitle] = useState("");
  const [src, setSrc] = useState("english");
  const [targets, setTargets] = useState<string[]>(["yoruba", "igbo", "hausa"]);
  const [pin, setPin] = useState("");

  const effectiveTargets = targets.filter((t) => t !== src);

  const create = useMutation({
    mutationFn: () =>
      api.createRoom({
        title: title.trim() || "Untitled session",
        src_lang: src,
        targets: effectiveTargets,
        pin: pin.trim() || undefined,
      }),
    onSuccess: (r) =>
      onCreated({
        ...r,
        title: title.trim() || "Untitled session",
        src_lang: src,
        targets: effectiveTargets,
        pin: pin.trim() || undefined,
        createdAt: Date.now(),
      }),
    onError: (e) => toast.error("Room not created", isOffline(e) ? "The engine is offline." : errorMessage(e)),
  });

  return (
    <>
      <PageHeader
        title="Host a live session"
        description="Create a room, show the QR code, and speak. Listeners follow on their phones with captions and audio in their own language."
      />
      {!online && !checking && <EngineOffline className="mb-8" onRetry={() => void refetch()} />}

      <div className="grid gap-8 lg:grid-cols-[1.2fr_1fr]">
        <form
          className="shell"
          onSubmit={(e) => {
            e.preventDefault();
            if (effectiveTargets.length === 0) {
              toast.error("Pick at least one listener language");
              return;
            }
            create.mutate();
          }}
        >
          <div className="core flex flex-col gap-6 p-6 md:p-8">
            <Field label="Session title" htmlFor="room-title" help="Shown to listeners when they join." optional>
              <input
                id="room-title"
                className="field-input"
                placeholder="Opening plenary"
                value={title}
                maxLength={120}
                onChange={(e) => setTitle(e.target.value)}
              />
            </Field>

            <LanguageSelect
              label="Speaker's language"
              value={src}
              onChange={setSrc}
              languages={languages.filter((l) => l.supports_asr)}
            />

            <fieldset>
              <legend className="field-label">Listener languages</legend>
              <p className="field-help mt-1">Each one gets live captions. Voice is added where a voice model exists.</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {languages
                  .filter((l) => l.name !== src)
                  .map((l) => {
                    const on = targets.includes(l.name);
                    return (
                      <label
                        key={l.name}
                        className={cn(
                          "inline-flex cursor-pointer items-center gap-2 rounded-full px-4 py-2 text-[16px] font-medium ring-1 transition-colors has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent",
                          on ? "bg-fg text-bg ring-fg" : "bg-raised text-muted ring-line-strong hover:text-fg",
                        )}
                      >
                        <input
                          type="checkbox"
                          className="sr-only"
                          checked={on}
                          onChange={() =>
                            setTargets((ts) => (on ? ts.filter((t) => t !== l.name) : [...ts, l.name]))
                          }
                        />
                        {on && <Check className="size-3.5" strokeWidth={2.25} aria-hidden />}
                        {l.display_name}
                      </label>
                    );
                  })}
              </div>
            </fieldset>

            <Field label="Room PIN" htmlFor="room-pin" help="Listeners must enter it to join. Leave empty for an open room." optional>
              <input
                id="room-pin"
                className="field-input max-w-[12rem] font-mono tracking-[0.3em]"
                inputMode="numeric"
                autoComplete="off"
                maxLength={8}
                value={pin}
                onChange={(e) => setPin(e.target.value.replace(/\D/g, ""))}
              />
            </Field>

            <div className="flex items-center gap-3 border-t border-line pt-6">
              <Button type="submit" size="lg" loading={create.isPending} disabled={!online || effectiveTargets.length === 0}>
                Create room
              </Button>
              {!online && <span className="text-[14px] text-subtle">Start the engine to create a room.</span>}
            </div>
          </div>
        </form>

        <aside className="flex flex-col gap-4">
          <div className="rounded-2xl bg-ink p-7 text-ink-fg">
            <h2 className="text-lg font-semibold tracking-tight">Before you start</h2>
            <ul className="mt-4 space-y-3 text-[16px] leading-relaxed text-ink-muted">
              <li>Put the laptop and listeners on the same Wi-Fi network or phone hotspot. No internet is needed.</li>
              <li>A lapel or podium microphone beats the laptop's built-in mic in a noisy room.</li>
              <li>Start the engine listening on your local network so phones can reach the join link.</li>
              <li>Ask listeners to bring earphones if they want audio.</li>
            </ul>
          </div>
          <div className="rounded-2xl border border-line p-6 text-[16px] text-muted">
            <p className="flex items-center gap-2 font-semibold text-fg">
              <Projector className="size-4" strokeWidth={1.6} /> Projector mode
            </p>
            <p className="mt-2 leading-relaxed">
              Every room also has a big-screen view with two caption lanes, for a projector or TV at the front of the hall.
            </p>
          </div>
        </aside>
      </div>
    </>
  );
}

// ─── Console ───────────────────────────────────────────────────────────────

type Phase = "ready" | "connecting" | "live" | "reconnecting" | "stopping";

function HostConsole({ room, onEnd }: { room: HostRoom; onEnd: () => void }) {
  const { languages } = useLanguages();
  const toast = useToast();
  const [phase, setPhase] = useState<Phase>("ready");
  const [listeners, setListeners] = useState(0);
  const [lines, setLines] = useState<CaptionLine[]>([]);
  const [partial, setPartial] = useState("");
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [copied, setCopied] = useState(false);
  const [micLabel, setMicLabel] = useState("");

  const level = useMotionValue(0);
  const levelScale = useTransform(level, (v) => Math.min(1, Math.sqrt(v) * 2.2));
  const peakDb = useRef(0);

  const wsRef = useRef<WebSocket | null>(null);
  const micRef = useRef<MicCapture | null>(null);
  const wantLive = useRef(false);
  const retry = useRef(0);
  const retryTimer = useRef<number | undefined>(undefined);
  const framesSent = useRef(0);

  const join = useMemo(() => resolveJoinUrl(room), [room]);

  useInterval(() => setNow(Date.now()), phase === "live" || phase === "reconnecting" ? 1000 : null);

  useEffect(() => {
    listMicrophones()
      .then(setDevices)
      .catch(() => setDevices([]));
  }, [phase]);

  const onEvent = useCallback(
    (ev: RoomEvent) => {
      switch (ev.type) {
        case "status":
          setListeners(ev.listeners);
          break;
        case "partial":
          if (ev.lang === room.src_lang || !room.targets.includes(ev.lang)) setPartial(ev.text);
          break;
        case "final":
          setPartial("");
          setLines((ls) => {
            const idx = ls.findIndex((l) => l.seq === ev.seq);
            if (idx >= 0) {
              const copy = ls.slice();
              copy[idx] = { ...copy[idx]!, text: ev.text, final: true };
              return copy;
            }
            return [...ls.slice(-199), { seq: ev.seq, text: ev.text, final: true, translations: {} }];
          });
          break;
        case "translation":
          setLines((ls) => {
            const idx = ls.findIndex((l) => l.seq === ev.seq);
            const t = { text: ev.text, origin: ev.origin, latency_ms: ev.latency_ms };
            if (idx >= 0) {
              const copy = ls.slice();
              const l = copy[idx]!;
              copy[idx] = { ...l, translations: { ...l.translations, [ev.lang]: t } };
              return copy;
            }
            return [...ls.slice(-199), { seq: ev.seq, text: "", final: false, translations: { [ev.lang]: t } }];
          });
          break;
        case "error":
          toast.error("Engine reported a problem", ev.message || ev.code);
          break;
        default:
          break;
      }
    },
    [room.src_lang, room.targets, toast],
  );

  const openSocket = useCallback(() => {
    const ws = new WebSocket(hostSocketUrl(room.id, room.host_token));
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    ws.onopen = () => {
      retry.current = 0;
      ws.send(JSON.stringify({ type: "start", sample_rate: 16000, encoding: "pcm_s16le" }));
      setPhase("live");
      setStartedAt((s) => s ?? Date.now());
    };
    ws.onmessage = (e) => {
      const ev = parseRoomEvent(e.data);
      if (ev) onEvent(ev);
    };
    ws.onclose = (e) => {
      if (wsRef.current === ws) wsRef.current = null;
      if (!wantLive.current) return;
      if (e.code >= 4400 && e.code < 4500) {
        wantLive.current = false;
        void micRef.current?.stop();
        micRef.current = null;
        setPhase("ready");
        toast.error("The server closed this room", e.reason || "The host token may have expired. Create a new room.");
        return;
      }
      retry.current += 1;
      setPhase("reconnecting");
      const delay = Math.min(10_000, 400 * 2 ** Math.min(retry.current, 5));
      retryTimer.current = window.setTimeout(() => wantLive.current && openSocket(), delay);
    };
  }, [room.id, room.host_token, onEvent, toast]);

  const goLive = async () => {
    setPhase("connecting");
    wantLive.current = true;
    try {
      const mic = new MicCapture();
      await mic.start(
        {
          onPcm: (buf) => {
            const ws = wsRef.current;
            if (ws && ws.readyState === WebSocket.OPEN) {
              ws.send(buf);
              framesSent.current += 1;
            }
          },
          onLevel: (rms, peak) => {
            level.set(rms);
            peakDb.current = peak;
          },
        },
        deviceId || undefined,
      );
      micRef.current = mic;
      setMicLabel(mic.deviceLabel);
      openSocket();
    } catch (e) {
      wantLive.current = false;
      setPhase("ready");
      const msg =
        e instanceof DOMException && e.name === "NotAllowedError"
          ? "Microphone access was blocked. Allow it in the browser's site settings."
          : e instanceof DOMException && e.name === "NotFoundError"
            ? "No microphone was found."
            : errorMessage(e);
      toast.error("Could not start the microphone", msg);
    }
  };

  const stop = useCallback(async () => {
    wantLive.current = false;
    window.clearTimeout(retryTimer.current);
    setPhase("stopping");
    micRef.current?.flush();
    await new Promise((r) => setTimeout(r, 120));
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "stop" }));
      } catch {
        /* ignore */
      }
      window.setTimeout(() => ws.close(1000, "host stopped"), 400);
    }
    await micRef.current?.stop();
    micRef.current = null;
    level.set(0);
    setPartial("");
    setPhase("ready");
  }, [level]);

  // Clean up on unmount.
  useEffect(
    () => () => {
      wantLive.current = false;
      window.clearTimeout(retryTimer.current);
      wsRef.current?.close(1000, "host left");
      void micRef.current?.stop();
    },
    [],
  );

  // Warn before closing the tab while live.
  useEffect(() => {
    if (phase !== "live" && phase !== "reconnecting") return;
    const h = (e: BeforeUnloadEvent) => {
      e.preventDefault();
    };
    window.addEventListener("beforeunload", h);
    return () => window.removeEventListener("beforeunload", h);
  }, [phase]);

  const isLive = phase === "live" || phase === "reconnecting";
  const elapsed = startedAt ? (now - startedAt) / 1000 : 0;
  const columns = [room.src_lang, ...room.targets];

  return (
    <>
      <header className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            {isLive ? (
              <Pill tone={phase === "live" ? "accent" : "danger"}>
                <span className="relative flex size-2">
                  <span className="absolute inset-0 animate-ping rounded-full bg-current opacity-50 motion-reduce:hidden" />
                  <span className="relative size-2 rounded-full bg-current" />
                </span>
                {phase === "live" ? "Live" : "Reconnecting"}
              </Pill>
            ) : (
              <Pill>Not live</Pill>
            )}
            <span className="font-mono text-[13px] text-subtle">Room {room.id}</span>
          </div>
          <h1 className="mt-3 text-3xl font-semibold tracking-[-0.03em] md:text-[2.4rem]">{room.title}</h1>
          <p className="mt-1.5 text-muted">
            {langLabel(room.src_lang, languages)} to {room.targets.map((t) => langLabel(t, languages)).join(", ")}
          </p>
        </div>
        <dl className="flex flex-wrap gap-6">
          <div>
            <dt className="flex items-center gap-1.5 text-[14px] text-subtle">
              <Users className="size-3.5" strokeWidth={1.75} /> Listeners
            </dt>
            <dd className="mt-0.5 font-display text-3xl font-semibold tabular-nums" aria-live="polite">
              {listeners}
            </dd>
          </div>
          <div>
            <dt className="flex items-center gap-1.5 text-[14px] text-subtle">
              <Timer className="size-3.5" strokeWidth={1.75} /> Elapsed
            </dt>
            <dd className="mt-0.5 font-display text-3xl font-semibold tabular-nums">{formatDuration(elapsed)}</dd>
          </div>
        </dl>
      </header>

      <div className="grid gap-6 xl:grid-cols-[22rem_1fr]">
        {/* Left rail: join + controls */}
        <aside className="flex flex-col gap-6">
          <div className="shell">
            <div className="core flex flex-col items-center p-6 text-center">
              <QrCode value={join.url} size={208} label={`QR code to join room ${room.id}`} />
              <p className="mt-4 text-[14px] text-subtle">Scan to listen</p>
              <p className="mt-1 break-all font-mono text-[14px] text-fg">{join.url.replace(/^https?:\/\//, "")}</p>
              {room.pin && (
                <p className="mt-3 text-[15px] text-muted">
                  PIN <span className="font-mono text-lg font-semibold tracking-[0.25em] text-fg">{room.pin}</span>
                </p>
              )}
              {join.loopbackOnly && (
                <p className="mt-4 flex items-start gap-2 rounded-2xl bg-warn-soft p-3 text-left text-[14px] leading-snug text-fg">
                  <Info className="mt-0.5 size-4 shrink-0" strokeWidth={1.75} />
                  Phones cannot open a localhost address. Start the engine so it listens on your local network; the join link
                  then uses this laptop's network address.
                </p>
              )}
              <div className="mt-5 flex w-full flex-wrap justify-center gap-2">
                <Button
                  size="sm"
                  variant="secondary"
                  icon={copied ? <Check className="size-4 text-ok" strokeWidth={2} /> : <Copy className="size-4" strokeWidth={1.75} />}
                  onClick={async () => {
                    if (await copyText(join.url)) {
                      setCopied(true);
                      window.setTimeout(() => setCopied(false), 1500);
                    }
                  }}
                >
                  {copied ? "Copied" : "Copy link"}
                </Button>
                <a
                  href={`/room/${encodeURIComponent(room.id)}?view=projector`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex h-9 items-center gap-2 rounded-full px-3.5 text-[15px] font-medium text-fg ring-1 ring-line-strong transition-colors hover:bg-sunken"
                >
                  <Projector className="size-4" strokeWidth={1.75} /> Projector
                  <ExternalLink className="size-3.5 text-subtle" strokeWidth={1.75} />
                </a>
              </div>
            </div>
          </div>

          <div className="shell">
            <div className="core flex flex-col gap-5 p-6">
              <Field label="Microphone" htmlFor="mic-select">
                <Select
                  id="mic-select"
                  value={deviceId}
                  disabled={isLive || phase === "connecting"}
                  onChange={(e) => setDeviceId(e.target.value)}
                >
                  <option value="">System default</option>
                  {devices
                    .filter((d) => d.deviceId && d.deviceId !== "default")
                    .map((d, i) => (
                      <option key={d.deviceId} value={d.deviceId}>
                        {d.label || `Microphone ${i + 1}`}
                      </option>
                    ))}
                </Select>
              </Field>

              <div>
                <div className="flex items-center justify-between text-[14px] text-subtle">
                  <span className="flex items-center gap-1.5">
                    <Mic className="size-3.5" strokeWidth={1.75} /> Input level
                  </span>
                  {micLabel && isLive && <span className="max-w-[12rem] truncate">{micLabel}</span>}
                </div>
                <div
                  className="mt-2 h-2.5 overflow-hidden rounded-full bg-sunken"
                  role="meter"
                  aria-label="Microphone input level"
                  aria-valuemin={0}
                  aria-valuemax={1}
                >
                  <motion.div
                    className="h-full origin-left rounded-full bg-[linear-gradient(90deg,var(--ok)_0%,var(--ok)_60%,var(--warn)_80%,var(--accent)_100%)]"
                    style={{ scaleX: levelScale }}
                  />
                </div>
              </div>

              {isLive ? (
                <Button
                  variant="danger"
                  size="lg"
                  onClick={() => void stop()}
                  loading={false}
                  icon={<Square className="size-4" strokeWidth={2} />}
                >
                  Stop
                </Button>
              ) : (
                <Button
                  size="lg"
                  onClick={() => void goLive()}
                  loading={phase === "connecting" || phase === "stopping"}
                  icon={<Radio className="size-4" strokeWidth={1.75} />}
                  disabled={!MicCapture.supported}
                >
                  {lines.length > 0 ? "Resume" : "Go live"}
                </Button>
              )}
              {!MicCapture.supported && (
                <p className="text-[14px] text-danger">This browser cannot capture audio. Use a recent Chrome, Edge, Firefox or Safari.</p>
              )}
              <Button
                variant="ghost"
                size="sm"
                disabled={isLive}
                icon={<RotateCcw className="size-4" strokeWidth={1.75} />}
                onClick={() => {
                  if (window.confirm("End this room and create a new one? Listeners will be disconnected.")) onEnd();
                }}
              >
                New room
              </Button>
            </div>
          </div>
        </aside>

        {/* Captions */}
        <section aria-label="Live captions" className="min-w-0">
          <div
            className="grid gap-4"
            style={{ gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, 16rem), 1fr))` }}
          >
            {columns.map((lang, ci) => (
              <CaptionColumn
                key={lang}
                lang={lang}
                label={langLabel(lang, languages)}
                short={langShort(lang, languages)}
                source={ci === 0}
                lines={lines}
                partial={ci === 0 ? partial : ""}
                live={isLive}
              />
            ))}
          </div>
        </section>
      </div>
    </>
  );
}

function CaptionColumn({
  lang,
  label,
  short,
  source,
  lines,
  partial,
  live,
}: {
  lang: string;
  label: string;
  short: string;
  source: boolean;
  lines: CaptionLine[];
  partial: string;
  live: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines, partial]);

  const items = lines
    .map((l) => ({
      seq: l.seq,
      text: source ? l.text : l.translations[lang]?.text ?? "",
      meta: source ? undefined : l.translations[lang],
      pending: !source && !l.translations[lang] && l.final,
    }))
    .filter((i) => i.text || i.pending);

  return (
    <div className="flex h-[min(70vh,40rem)] flex-col overflow-hidden rounded-xl border border-line bg-raised">
      <div className="flex items-center justify-between border-b border-line px-5 py-3">
        <p className="flex items-center gap-2 text-[15px] font-semibold">
          <span className="font-mono text-[12px] text-accent-text">{short}</span>
          {label}
        </p>
        {source && <span className="text-[13px] text-subtle">Speaker</span>}
      </div>
      <div
        ref={ref}
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
        }}
        className="scrollbar-thin flex-1 space-y-3 overflow-y-auto px-5 py-4"
        aria-live={source ? "off" : "polite"}
        lang={lang}
      >
        {items.length === 0 && !partial && (
          <p className="pt-6 text-center text-[15px] text-subtle">
            {live ? "Listening. Captions appear as you speak." : "Captions appear here once you go live."}
          </p>
        )}
        {items.map((i) => (
          <div key={i.seq}>
            {i.pending ? (
              <div className="h-5 w-2/3 animate-pulse rounded bg-sunken" aria-hidden />
            ) : (
              <p className="text-[17px] leading-relaxed text-fg">{i.text}</p>
            )}
            {i.meta?.origin && (
              <p className="mt-0.5 font-mono text-[12px] text-subtle">
                {i.meta.origin}
                {typeof i.meta.latency_ms === "number" && ` · ${Math.round(i.meta.latency_ms)} ms`}
              </p>
            )}
          </div>
        ))}
        {partial && <p className="text-[17px] italic leading-relaxed text-subtle">{partial}</p>}
      </div>
    </div>
  );
}
