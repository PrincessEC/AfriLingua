import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import {
  ArrowDown,
  Headphones,
  Lock,
  Maximize,
  Minus,
  Plus,
  Settings,
  Users,
  Volume2,
  VolumeX,
  WifiOff,
  X,
} from "lucide-react";
import { api, isOffline, listenSocketUrl, type RoomEvent, type RoomInfo } from "../lib/api";
import { langLabel, useLanguages } from "../lib/languages";
import { WavQueuePlayer } from "../lib/audio";
import { useLocalStorage, useWakeLock } from "../lib/hooks";
import { useReconnectingSocket, type SocketState } from "../lib/socket";
import { cn } from "../lib/utils";
import { LogoMark } from "../components/Brand";
import { Button } from "../components/ui";
import { ThemeToggle } from "../components/AppShell";

interface Caption {
  seq: number;
  text: string;
}

const MAX_LINES = 120;

function upsert(list: Caption[], seq: number, text: string): Caption[] {
  const idx = list.findIndex((c) => c.seq === seq);
  if (idx >= 0) {
    const copy = list.slice();
    copy[idx] = { seq, text };
    return copy;
  }
  const next = [...list, { seq, text }];
  next.sort((a, b) => a.seq - b.seq);
  return next.slice(-MAX_LINES);
}

export default function Room() {
  const { id = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const projector = params.get("view") === "projector";
  const { languages } = useLanguages();

  const info = useQuery({
    queryKey: ["room", id],
    queryFn: ({ signal }) => api.room(id, signal),
    retry: false,
    // Poll fast while unreachable so a listener joins as soon as the network is back.
    refetchInterval: (q) => (q.state.data ? 20_000 : q.state.error && !isOffline(q.state.error) ? 15_000 : 4_000),
    refetchIntervalInBackground: true,
  });

  useEffect(() => {
    const title = info.data?.title;
    document.title = title ? `${title} · AfriLingua` : "Live captions · AfriLingua";
  }, [info.data?.title]);

  if (info.isPending) return <RoomLoading />;
  if (info.isError && !info.data) {
    return <RoomUnavailable notFound={!isOffline(info.error)} onRetry={() => void info.refetch()} />;
  }
  const room = info.data!;
  const pinParam = params.get("pin") ?? "";
  const setPin = (pin: string) => {
    const next = new URLSearchParams(params);
    next.set("pin", pin);
    setParams(next, { replace: true });
  };

  return projector ? (
    <ProjectorView room={room} languages={languages} pin={pinParam} onPin={setPin} initialLang={params.get("lang")} />
  ) : (
    <ListenerView room={room} languages={languages} pin={pinParam} onPin={setPin} />
  );
}

// ─── States ────────────────────────────────────────────────────────────────

function RoomLoading() {
  return (
    <div className="grid min-h-[100dvh] place-items-center bg-bg" aria-busy="true">
      <div className="flex flex-col items-center gap-4">
        <LogoMark className="size-10 animate-pulse-soft" />
        <p className="text-sm text-muted">Joining the room</p>
      </div>
    </div>
  );
}

function RoomUnavailable({ notFound, onRetry }: { notFound: boolean; onRetry: () => void }) {
  return (
    <div className="grid min-h-[100dvh] place-items-center bg-bg px-6">
      <div className="max-w-sm text-center">
        <span className="mx-auto grid size-14 place-items-center rounded-full bg-sunken text-muted">
          <WifiOff className="size-6" strokeWidth={1.5} />
        </span>
        <h1 className="mt-6 text-2xl font-semibold tracking-tight">
          {notFound ? "This room is not available" : "Cannot reach the interpreter"}
        </h1>
        <p className="mt-3 text-[15px] leading-relaxed text-muted">
          {notFound
            ? "The session may have ended, or the link is incomplete. Ask the host for a fresh QR code."
            : "Make sure your phone is on the same Wi-Fi network as the host's laptop. This page keeps trying."}
        </p>
        <div className="mt-6 flex justify-center gap-2">
          <Button onClick={onRetry}>Try again</Button>
          <Link to="/" className="inline-flex h-11 items-center rounded-full px-5 text-[15px] font-medium text-fg ring-1 ring-line-strong">
            Home
          </Link>
        </div>
        {!notFound && (
          <p className="mt-8 text-[13px] text-subtle">
            Hosting? Start the engine with <span className="font-mono text-fg">afrilingua serve</span>.
          </p>
        )}
      </div>
    </div>
  );
}

function PinGate({ onSubmit, wrong }: { onSubmit: (pin: string) => void; wrong: boolean }) {
  const [pin, setPin] = useState("");
  return (
    <div className="mx-auto mt-16 max-w-xs px-6 text-center">
      <span className="mx-auto grid size-14 place-items-center rounded-full bg-sunken text-muted">
        <Lock className="size-6" strokeWidth={1.5} />
      </span>
      <h2 className="mt-5 text-xl font-semibold tracking-tight">This room has a PIN</h2>
      <p className="mt-2 text-[15px] text-muted">The host can see it on their screen.</p>
      <form
        className="mt-6 flex flex-col gap-3"
        onSubmit={(e) => {
          e.preventDefault();
          if (pin) onSubmit(pin);
        }}
      >
        <label htmlFor="pin" className="sr-only">
          Room PIN
        </label>
        <input
          id="pin"
          inputMode="numeric"
          autoComplete="one-time-code"
          autoFocus
          maxLength={8}
          value={pin}
          onChange={(e) => setPin(e.target.value.replace(/\D/g, ""))}
          className="field-input text-center font-mono text-2xl tracking-[0.4em]"
        />
        {wrong && (
          <p className="text-[13px] font-medium text-danger" role="alert">
            That PIN did not work. Check it and try again.
          </p>
        )}
        <Button type="submit" size="lg" disabled={!pin}>
          Join
        </Button>
      </form>
    </div>
  );
}

function ConnectionBadge({ state, live }: { state: SocketState; live: boolean }) {
  const text =
    state === "open" ? (live ? "Live" : "Waiting for speaker") : state === "reconnecting" ? "Reconnecting" : state === "connecting" ? "Connecting" : "Offline";
  const tone = state === "open" ? (live ? "bg-ok" : "bg-ochre") : state === "closed" ? "bg-danger" : "bg-ochre";
  return (
    <span className="inline-flex items-center gap-1.5 text-[13px] font-medium text-muted" role="status">
      <span className="relative flex size-2">
        {state === "open" && live && <span className={cn("absolute inset-0 animate-ping rounded-full opacity-60 motion-reduce:hidden", tone)} />}
        <span className={cn("relative size-2 rounded-full", tone)} />
      </span>
      {text}
    </span>
  );
}

// ─── Shared listening logic ────────────────────────────────────────────────

function useRoomStream(room: RoomInfo, lang: string, audio: boolean, pin: string, player: WavQueuePlayer | null) {
  const [source, setSource] = useState<Caption[]>([]);
  const [target, setTarget] = useState<Caption[]>([]);
  const [partial, setPartial] = useState("");
  const [live, setLive] = useState(room.live);
  const [listeners, setListeners] = useState(room.listeners);
  const [pinError, setPinError] = useState(false);
  const audioRef = useRef(audio);
  audioRef.current = audio;

  // Reset translated lane when the language changes.
  useEffect(() => {
    setTarget([]);
    player?.clear();
  }, [lang, player]);

  const onEvent = useCallback(
    (ev: RoomEvent) => {
      switch (ev.type) {
        case "status":
          setLive(ev.live);
          setListeners(ev.listeners);
          break;
        case "partial":
          if (ev.lang === room.src_lang) setPartial(ev.text);
          break;
        case "final":
          if (ev.lang === room.src_lang) {
            setPartial("");
            setSource((s) => upsert(s, ev.seq, ev.text));
          }
          break;
        case "translation":
          if (ev.lang === lang) setTarget((t) => upsert(t, ev.seq, ev.text));
          break;
        case "audio":
          if (ev.lang === lang && audioRef.current && player) player.enqueueBase64(ev.data);
          break;
        case "error":
          if (/pin|unauthori|forbidden/i.test(ev.code)) setPinError(true);
          break;
      }
    },
    [room.src_lang, lang, player],
  );

  const url = useMemo(
    () => (id(room) ? listenSocketUrl(room.id, { lang, audio, pin: pin || undefined }) : null),
    [room, lang, audio, pin],
  );
  const sock = useReconnectingSocket(url, onEvent);
  const needsPin = pinError || (sock.state === "closed" && (sock.closeCode === 4401 || sock.closeCode === 4403));

  useEffect(() => {
    if (sock.state === "open") setPinError(false);
  }, [sock.state]);

  const captions = lang === room.src_lang ? source : target;
  return { source, target, captions, partial, live, listeners, sock, needsPin };
}

function id(room: RoomInfo) {
  return Boolean(room.id);
}

// ─── Listener view (mobile-first) ──────────────────────────────────────────

function ListenerView({
  room,
  languages,
  pin,
  onPin,
}: {
  room: RoomInfo;
  languages: ReturnType<typeof useLanguages>["languages"];
  pin: string;
  onPin: (p: string) => void;
}) {
  const reduce = useReducedMotion();
  const options = useMemo(() => [...room.targets, room.src_lang], [room.targets, room.src_lang]);
  const [savedLang, setSavedLang] = useLocalStorage("afrilingua.listenLang", room.targets[0] ?? room.src_lang);
  const lang = options.includes(savedLang) ? savedLang : (room.targets[0] ?? room.src_lang);
  const [size, setSize] = useLocalStorage("afrilingua.captionSize", 30);
  const [audioOn, setAudioOn] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const playerRef = useRef<WavQueuePlayer | null>(null);
  if (!playerRef.current) playerRef.current = new WavQueuePlayer();
  const player = playerRef.current;
  const langSpec = languages.find((l) => l.name === lang);
  const voiceAvailable = langSpec ? langSpec.supports_tts : true;

  const { captions, partial, live, listeners, sock, needsPin } = useRoomStream(room, lang, audioOn, pin, player);
  const { held: wakeHeld, supported: wakeSupported } = useWakeLock(sock.state === "open");

  useEffect(() => () => void player.close(), [player]);

  const toggleAudio = async () => {
    if (audioOn) {
      player.clear();
      setAudioOn(false);
      return;
    }
    const ok = await player.unlock();
    setAudioOn(ok);
  };

  // Auto-scroll unless the reader scrolled up.
  const scroller = useRef<HTMLDivElement>(null);
  const [pinned, setPinned] = useState(true);
  useEffect(() => {
    const el = scroller.current;
    if (el && pinned) el.scrollTo({ top: el.scrollHeight, behavior: reduce ? "auto" : "smooth" });
  }, [captions, partial, pinned, reduce]);

  const speakingIndicator = lang !== room.src_lang && partial;

  return (
    <div className="flex h-[100dvh] flex-col bg-bg text-fg">
      {/* Top bar */}
      <header className="flex items-center gap-3 border-b border-line px-4 pb-3 pt-[max(0.75rem,env(safe-area-inset-top))]">
        <Link to="/" aria-label="AfriLingua home">
          <LogoMark className="size-8" />
        </Link>
        <div className="min-w-0 flex-1">
          <p className="truncate text-[15px] font-semibold leading-tight">{room.title || "Live session"}</p>
          <div className="mt-0.5 flex items-center gap-3">
            <ConnectionBadge state={sock.state} live={live} />
            <span className="inline-flex items-center gap-1 text-[13px] text-subtle">
              <Users className="size-3.5" strokeWidth={1.75} aria-hidden />
              <span className="sr-only">Listeners:</span>
              {listeners}
            </span>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setSettingsOpen(true)}
          className="inline-grid size-10 place-items-center rounded-full text-muted hover:bg-sunken hover:text-fg"
          aria-label="Caption settings"
        >
          <Settings className="size-5" strokeWidth={1.6} />
        </button>
      </header>

      {/* Language picker */}
      <nav aria-label="Caption language" className="no-scrollbar flex gap-2 overflow-x-auto border-b border-line px-4 py-2.5">
        {options.map((l) => (
          <button
            key={l}
            type="button"
            onClick={() => setSavedLang(l)}
            aria-pressed={l === lang}
            className={cn(
              "shrink-0 rounded-full px-4 py-2 text-[15px] font-medium transition-colors",
              l === lang ? "bg-fg text-bg" : "bg-sunken text-muted hover:text-fg",
            )}
          >
            {langLabel(l, languages)}
            {l === room.src_lang && <span className="ml-1.5 text-[12px] opacity-70">original</span>}
          </button>
        ))}
      </nav>

      {sock.state === "reconnecting" && (
        <div className="flex items-center justify-center gap-2 bg-ochre-soft px-4 py-2 text-[13px] font-medium text-fg" role="status">
          <WifiOff className="size-4" strokeWidth={1.75} />
          Connection lost. Reconnecting{sock.attempt > 1 ? ` (attempt ${sock.attempt})` : ""}.
        </div>
      )}

      {/* Captions */}
      {needsPin ? (
        <PinGate onSubmit={onPin} wrong={Boolean(pin)} />
      ) : (
        <div className="relative min-h-0 flex-1">
          <div
            ref={scroller}
            onScroll={(e) => {
              const el = e.currentTarget;
              setPinned(el.scrollHeight - el.scrollTop - el.clientHeight < 80);
            }}
            className="h-full overflow-y-auto px-5 pb-40 pt-6 sm:px-8"
          >
            <div
              className="mx-auto max-w-3xl"
              aria-live="polite"
              aria-atomic="false"
              aria-relevant="additions text"
              lang={langSpec?.iso639_1 ?? undefined}
            >
              {captions.length === 0 && (
                <div className="pt-[18vh] text-center">
                  <p className="font-display text-2xl font-semibold tracking-tight text-fg">
                    {live ? "Captions will appear here." : "Waiting for the speaker."}
                  </p>
                  <p className="mx-auto mt-2 max-w-xs text-[15px] text-muted">
                    Keep this page open. {wakeSupported ? "Your screen stays awake while you listen." : ""}
                  </p>
                </div>
              )}
              {captions.map((c, i) => {
                const latest = i === captions.length - 1;
                return (
                  <motion.p
                    key={c.seq}
                    initial={reduce ? false : { opacity: 0, y: 12 }}
                    animate={{ opacity: latest ? 1 : 0.5, y: 0 }}
                    transition={{ duration: 0.35, ease: [0.23, 1, 0.32, 1] }}
                    className="mb-[0.6em] font-medium leading-[1.3] tracking-[-0.01em]"
                    style={{ fontSize: `${size}px` }}
                  >
                    {c.text}
                  </motion.p>
                );
              })}
              {lang === room.src_lang && partial && (
                <p className="font-medium leading-[1.3] text-subtle" style={{ fontSize: `${size}px` }}>
                  {partial}
                </p>
              )}
              {speakingIndicator && (
                <p className="mt-2 inline-flex items-center gap-2 text-[14px] text-subtle" aria-hidden>
                  <span className="flex gap-1">
                    {[0, 1, 2].map((d) => (
                      <span key={d} className="size-1.5 animate-pulse-soft rounded-full bg-subtle" style={{ animationDelay: `${d * 160}ms` }} />
                    ))}
                  </span>
                  Speaker is talking
                </p>
              )}
            </div>
          </div>

          <AnimatePresence>
            {!pinned && captions.length > 0 && (
              <motion.button
                type="button"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 8 }}
                onClick={() => setPinned(true)}
                className="absolute bottom-28 left-1/2 inline-flex -translate-x-1/2 items-center gap-2 rounded-full bg-fg px-4 py-2 text-sm font-medium text-bg shadow-lg"
              >
                <ArrowDown className="size-4" strokeWidth={1.75} /> Latest
              </motion.button>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Bottom bar, thumb zone */}
      <footer className="border-t border-line bg-bg/95 px-4 pb-[max(1rem,env(safe-area-inset-bottom))] pt-3 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center gap-3">
          <button
            type="button"
            onClick={() => void toggleAudio()}
            disabled={!voiceAvailable || !player.supported}
            aria-pressed={audioOn}
            className={cn(
              "inline-flex h-12 flex-1 items-center justify-center gap-2 rounded-full text-[15px] font-semibold transition-[background-color,transform] active:scale-[0.98] disabled:opacity-50",
              audioOn ? "bg-accent text-accent-fg" : "bg-sunken text-fg ring-1 ring-line-strong",
            )}
          >
            {audioOn ? <Volume2 className="size-5" strokeWidth={1.75} /> : <VolumeX className="size-5" strokeWidth={1.75} />}
            {!voiceAvailable ? "No voice for this language" : audioOn ? "Audio on" : "Play audio"}
          </button>
          <div className="flex items-center rounded-full ring-1 ring-line-strong">
            <button
              type="button"
              className="inline-grid size-12 place-items-center rounded-full text-muted hover:text-fg"
              onClick={() => setSize((s) => Math.max(18, s - 4))}
              aria-label="Smaller captions"
            >
              <Minus className="size-4" strokeWidth={2} />
            </button>
            <span className="w-8 text-center font-mono text-[12px] text-subtle" aria-hidden>
              {size}
            </span>
            <button
              type="button"
              className="inline-grid size-12 place-items-center rounded-full text-muted hover:text-fg"
              onClick={() => setSize((s) => Math.min(64, s + 4))}
              aria-label="Larger captions"
            >
              <Plus className="size-4" strokeWidth={2} />
            </button>
          </div>
        </div>
        {audioOn && (
          <p className="mx-auto mt-2 flex max-w-3xl items-center gap-1.5 text-[12px] text-subtle">
            <Headphones className="size-3.5" strokeWidth={1.75} /> Earphones keep the room quiet for others.
          </p>
        )}
      </footer>

      {/* Settings sheet */}
      <AnimatePresence>
        {settingsOpen && (
          <div className="fixed inset-0 z-50">
            <motion.div
              className="absolute inset-0 bg-ink/40"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSettingsOpen(false)}
            />
            <motion.div
              role="dialog"
              aria-modal="true"
              aria-label="Caption settings"
              initial={reduce ? { opacity: 0 } : { y: "100%" }}
              animate={reduce ? { opacity: 1 } : { y: 0 }}
              exit={reduce ? { opacity: 0 } : { y: "100%" }}
              transition={{ duration: 0.35, ease: [0.32, 0.72, 0, 1] }}
              className="absolute inset-x-0 bottom-0 rounded-t-[1.75rem] border-t border-line bg-raised px-6 pb-[max(1.5rem,env(safe-area-inset-bottom))] pt-4"
            >
              <div className="mx-auto mb-4 h-1 w-10 rounded-full bg-line-strong" aria-hidden />
              <div className="mx-auto max-w-md">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold">Caption settings</h2>
                  <button
                    type="button"
                    onClick={() => setSettingsOpen(false)}
                    className="inline-grid size-9 place-items-center rounded-full text-muted hover:bg-sunken"
                    aria-label="Close settings"
                    autoFocus
                  >
                    <X className="size-4" strokeWidth={1.75} />
                  </button>
                </div>
                <label htmlFor="cap-size" className="mt-6 block text-sm font-medium">
                  Caption size <span className="font-mono text-subtle">{size}px</span>
                </label>
                <input
                  id="cap-size"
                  type="range"
                  min={18}
                  max={64}
                  step={2}
                  value={size}
                  onChange={(e) => setSize(Number(e.target.value))}
                  className="mt-3 w-full"
                />
                <p className="mt-3 font-medium leading-snug" style={{ fontSize: `${Math.min(size, 40)}px` }}>
                  Ẹ káàárọ̀
                </p>
                <div className="mt-6 flex items-center justify-between border-t border-line pt-5">
                  <span className="text-sm font-medium">Theme</span>
                  <ThemeToggle />
                </div>
                <div className="mt-4 flex items-center justify-between">
                  <span className="text-sm font-medium">Keep screen awake</span>
                  <span className="text-[13px] text-subtle">
                    {!wakeSupported ? "Not supported on this device" : wakeHeld ? "On" : "When connected"}
                  </span>
                </div>
                <div className="mt-4 flex items-center justify-between">
                  <span className="text-sm font-medium">Projector view</span>
                  <Link
                    to={`/room/${encodeURIComponent(room.id)}?view=projector${pin ? `&pin=${encodeURIComponent(pin)}` : ""}`}
                    className="text-sm font-medium text-accent-text underline-offset-4 hover:underline"
                  >
                    Open
                  </Link>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Projector view ────────────────────────────────────────────────────────

function ProjectorView({
  room,
  languages,
  pin,
  onPin,
  initialLang,
}: {
  room: RoomInfo;
  languages: ReturnType<typeof useLanguages>["languages"];
  pin: string;
  onPin: (p: string) => void;
  initialLang: string | null;
}) {
  const reduce = useReducedMotion();
  const [lang, setLang] = useState(
    initialLang && room.targets.includes(initialLang) ? initialLang : (room.targets[0] ?? room.src_lang),
  );
  const { source, target, partial, live, sock, needsPin } = useRoomStream(room, lang, false, pin, null);
  useWakeLock(sock.state === "open");

  // Projector is always dark for contrast in a lit hall.
  useEffect(() => {
    const root = document.documentElement;
    const had = root.classList.contains("dark");
    root.classList.add("dark");
    return () => {
      if (!had) root.classList.remove("dark");
    };
  }, []);

  const fullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void document.documentElement.requestFullscreen?.().catch(() => undefined);
  };

  const lane = (items: Caption[], extra?: string) => {
    const last = items.slice(-3);
    return (
      <div className="flex h-full flex-col justify-end gap-[0.4em] overflow-hidden">
        <AnimatePresence initial={false}>
          {last.map((c, i) => (
            <motion.p
              key={c.seq}
              layout={!reduce}
              initial={reduce ? false : { opacity: 0, y: 30 }}
              animate={{ opacity: i === last.length - 1 && !extra ? 1 : 0.42, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.45, ease: [0.23, 1, 0.32, 1] }}
              className="font-medium leading-[1.18] tracking-[-0.015em]"
            >
              {c.text}
            </motion.p>
          ))}
        </AnimatePresence>
        {extra && <p className="font-medium leading-[1.18] text-white/60">{extra}</p>}
      </div>
    );
  };

  return (
    <div className="flex h-[100dvh] flex-col bg-[#070918] text-[#f1ece1]">
      <header className="flex items-center justify-between gap-4 px-8 py-5">
        <div className="flex items-center gap-4">
          <LogoMark className="size-9" />
          <div>
            <p className="text-lg font-semibold">{room.title || "Live session"}</p>
            <ConnectionBadge state={sock.state} live={live} />
          </div>
        </div>
        <div className="flex items-center gap-3 opacity-40 transition-opacity hover:opacity-100 focus-within:opacity-100">
          <label htmlFor="proj-lang" className="sr-only">
            Translation lane language
          </label>
          <select
            id="proj-lang"
            value={lang}
            onChange={(e) => setLang(e.target.value)}
            className="rounded-full border border-white/20 bg-transparent px-4 py-2 text-sm"
          >
            {room.targets.map((t) => (
              <option key={t} value={t} className="bg-[#141833]">
                {langLabel(t, languages)}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={fullscreen}
            className="inline-grid size-10 place-items-center rounded-full border border-white/20 hover:bg-white/10"
            aria-label="Toggle full screen"
          >
            <Maximize className="size-4" strokeWidth={1.75} />
          </button>
        </div>
      </header>
      {needsPin ? (
        <PinGate onSubmit={onPin} wrong={Boolean(pin)} />
      ) : (
        <main className="grid min-h-0 flex-1 grid-rows-2 gap-0 px-8 pb-8 md:px-14">
          <section aria-label={`${langLabel(room.src_lang, languages)} captions`} className="flex min-h-0 flex-col border-b border-white/10 pb-6">
            <p className="mb-3 font-mono text-sm uppercase tracking-[0.2em] text-white/45">{langLabel(room.src_lang, languages)}</p>
            <div className="min-h-0 flex-1 text-[clamp(1.6rem,3.6vw,3.6rem)] text-white/80" aria-live="off">
              {lane(source, partial || undefined)}
            </div>
          </section>
          <section aria-label={`${langLabel(lang, languages)} captions`} className="flex min-h-0 flex-col pt-6">
            <p className="mb-3 font-mono text-sm uppercase tracking-[0.2em] text-[#e8b04a]">{langLabel(lang, languages)}</p>
            <div className="min-h-0 flex-1 text-[clamp(2rem,4.6vw,4.8rem)]" aria-live="polite" lang={languages.find((l) => l.name === lang)?.iso639_1 ?? undefined}>
              {target.length ? lane(target) : <p className="text-white/35">{live ? "Waiting for the first sentence." : "Waiting for the speaker."}</p>}
            </div>
          </section>
        </main>
      )}
    </div>
  );
}
