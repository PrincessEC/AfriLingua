import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion, useMotionValue, useTransform } from "motion/react";
import { Mic, RotateCcw, SkipForward, Square, Upload } from "lucide-react";
import { api, errorMessage, isOffline } from "../../lib/api";
import { pickRecorderMime } from "../../lib/audio";
import { useContributor } from "../../lib/contributor";
import { langLabel, useLanguages } from "../../lib/languages";
import { useToast } from "../../lib/toast";
import { cn, formatDuration } from "../../lib/utils";
import { Button, Field, Select, Skeleton } from "../../components/ui";
import { EmptyState, QueryError } from "../../components/states";
import { DialectInput } from "./TranslateTab";

const MAX_SECONDS = 30;

type RecState = "idle" | "recording" | "recorded";

export function RecordTab({ lang, onSubmitted }: { lang: string; onSubmitted: () => void }) {
  const { languages } = useLanguages();
  const { profile, update, name } = useContributor();
  const toast = useToast();
  const prompts = useQuery({
    queryKey: ["prompts", "record", lang],
    queryFn: ({ signal }) => api.prompts({ lang, kind: "record", n: 10 }, signal),
    staleTime: Infinity,
  });
  const [idx, setIdx] = useState(0);
  const [state, setState] = useState<RecState>("idle");
  const [blob, setBlob] = useState<Blob | null>(null);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [seconds, setSeconds] = useState(0);
  const [dialect, setDialect] = useState(profile.dialect);
  const [accent, setAccent] = useState("");
  const [region, setRegion] = useState(profile.region);
  const [gender, setGender] = useState("");
  const [age, setAge] = useState("");

  const recRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunks = useRef<Blob[]>([]);
  const timer = useRef<number | undefined>(undefined);
  const raf = useRef<number | undefined>(undefined);
  const ctxRef = useRef<AudioContext | null>(null);
  const level = useMotionValue(0);
  const ring = useTransform(level, (v) => 1 + Math.min(0.5, v * 3));

  const list = prompts.data ?? [];
  const prompt = list[idx];
  const mime = pickRecorderMime();
  const supported = typeof MediaRecorder !== "undefined" && !!navigator.mediaDevices?.getUserMedia;

  useEffect(() => {
    setIdx(0);
  }, [lang, prompts.dataUpdatedAt]);

  const cleanupStream = () => {
    window.clearInterval(timer.current);
    if (raf.current) cancelAnimationFrame(raf.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    void ctxRef.current?.close().catch(() => undefined);
    ctxRef.current = null;
    level.set(0);
  };

  useEffect(
    () => () => {
      cleanupStream();
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const reset = () => {
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    setBlob(null);
    setBlobUrl(null);
    setSeconds(0);
    setState("idle");
  };

  const start = async () => {
    try {
      reset();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      streamRef.current = stream;
      const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      chunks.current = [];
      rec.ondataavailable = (e) => e.data.size > 0 && chunks.current.push(e.data);
      rec.onstop = () => {
        const b = new Blob(chunks.current, { type: rec.mimeType || mime || "audio/webm" });
        setBlob(b);
        setBlobUrl(URL.createObjectURL(b));
        setState("recorded");
        cleanupStream();
      };
      recRef.current = rec;
      rec.start(250);
      setState("recording");
      const t0 = Date.now();
      timer.current = window.setInterval(() => {
        const s = (Date.now() - t0) / 1000;
        setSeconds(s);
        if (s >= MAX_SECONDS) stop();
      }, 200);
      // Level visual via analyser (motion value, no re-render per frame).
      const Ctor = window.AudioContext;
      if (Ctor) {
        const ctx = new Ctor();
        ctxRef.current = ctx;
        const src = ctx.createMediaStreamSource(stream);
        const an = ctx.createAnalyser();
        an.fftSize = 512;
        src.connect(an);
        const data = new Float32Array(an.fftSize);
        const tick = () => {
          an.getFloatTimeDomainData(data);
          let sum = 0;
          for (const v of data) sum += v * v;
          level.set(Math.sqrt(sum / data.length));
          raf.current = requestAnimationFrame(tick);
        };
        tick();
      }
    } catch (e) {
      cleanupStream();
      setState("idle");
      toast.error(
        "Could not start recording",
        e instanceof DOMException && e.name === "NotAllowedError" ? "Microphone access was blocked." : errorMessage(e),
      );
    }
  };

  const stop = () => {
    if (recRef.current && recRef.current.state !== "inactive") recRef.current.stop();
  };

  const next = () => {
    reset();
    if (idx + 1 < list.length) setIdx(idx + 1);
    else void prompts.refetch().then(() => setIdx(0));
  };

  const upload = useMutation({
    mutationFn: () => {
      if (!blob || !prompt) throw new Error("Nothing to upload");
      const ext = blob.type.includes("ogg") ? "ogg" : blob.type.includes("mp4") ? "m4a" : "webm";
      return api.contributeAudio({
        audio: blob,
        filename: `recording.${ext}`,
        text: prompt.text,
        lang,
        dialect: dialect.trim() || undefined,
        accent: accent.trim() || undefined,
        region: region.trim() || undefined,
        gender: gender || undefined,
        age_range: age || undefined,
        license: profile.license,
        contributor: name,
        voice_cloning_opt_in: profile.voiceCloningOptIn,
      });
    },
    onSuccess: () => {
      toast.success("Recording saved. Thank you.", "Reviewers will listen to it next.");
      update({ dialect: dialect.trim(), region: region.trim() });
      onSubmitted();
      next();
    },
    onError: (e) => toast.error("Upload failed", isOffline(e) ? "The engine is offline." : errorMessage(e)),
  });

  // Space toggles recording when not typing.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(t.tagName) || t.isContentEditable) return;
      if (e.code === "Space" && prompt) {
        e.preventDefault();
        if (state === "recording") stop();
        else if (state === "idle") void start();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!supported) {
    return (
      <EmptyState
        icon={<Mic className="size-5" strokeWidth={1.6} />}
        title="Recording is not available in this browser"
        description="Use a recent Chrome, Edge, Firefox or Safari over HTTPS or localhost."
      />
    );
  }
  if (prompts.isError) return <QueryError error={prompts.error} onRetry={() => void prompts.refetch()} />;

  return (
    <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
      <div className="shell">
        <div className="core flex flex-col p-6 md:p-8">
          <div className="flex items-center justify-between text-[13px] text-subtle">
            <span>Read this aloud in {langLabel(lang, languages)}</span>
            {list.length > 0 && (
              <span>
                {idx + 1} of {list.length}
              </span>
            )}
          </div>
          <div className="mt-5 min-h-[6rem]">
            {prompts.isPending ? (
              <div className="space-y-3">
                <Skeleton className="h-9 w-4/5" />
                <Skeleton className="h-9 w-1/2" />
              </div>
            ) : prompt ? (
              <p lang={languages.find((l) => l.name === lang)?.iso639_1 ?? undefined} className="font-display text-[1.8rem] font-medium leading-snug tracking-[-0.02em] md:text-[2.2rem]">
                {prompt.text}
              </p>
            ) : (
              <EmptyState title="No sentences to record right now" description="Check back later, or translate some sentences first." />
            )}
          </div>

          {prompt && (
            <div className="mt-10 flex flex-col items-center gap-5">
              <div className="relative grid place-items-center">
                <motion.span
                  aria-hidden
                  style={{ scale: ring }}
                  className={cn("absolute size-24 rounded-full", state === "recording" ? "bg-accent-soft" : "bg-transparent")}
                />
                <button
                  type="button"
                  onClick={() => (state === "recording" ? stop() : void start())}
                  className={cn(
                    "relative grid size-20 place-items-center rounded-full transition-[background-color,transform] duration-200 active:scale-95",
                    state === "recording" ? "bg-danger text-bg" : "bg-accent text-accent-fg hover:bg-accent-hover",
                  )}
                  aria-label={state === "recording" ? "Stop recording" : state === "recorded" ? "Record again" : "Start recording"}
                  aria-pressed={state === "recording"}
                >
                  {state === "recording" ? <Square className="size-7" strokeWidth={2} /> : <Mic className="size-8" strokeWidth={1.6} />}
                </button>
              </div>
              <p className="font-mono text-sm tabular-nums text-muted" aria-live="polite">
                {state === "recording"
                  ? `${formatDuration(seconds)} / 00:${MAX_SECONDS}`
                  : state === "recorded"
                    ? `Recorded ${formatDuration(seconds)}`
                    : "Tap or press Space to record"}
              </p>

              {state === "recorded" && blobUrl && (
                <div className="flex w-full flex-col items-center gap-4">
                  <audio controls src={blobUrl} className="w-full max-w-md" />
                  <div className="flex flex-wrap justify-center gap-2">
                    <Button
                      onClick={() => upload.mutate()}
                      loading={upload.isPending}
                      icon={<Upload className="size-4" strokeWidth={1.75} />}
                    >
                      Submit recording
                    </Button>
                    <Button variant="secondary" onClick={reset} icon={<RotateCcw className="size-4" strokeWidth={1.75} />}>
                      Re-record
                    </Button>
                  </div>
                </div>
              )}
              {state === "idle" && (
                <Button variant="ghost" size="sm" onClick={next} icon={<SkipForward className="size-4" strokeWidth={1.75} />}>
                  Skip sentence
                </Button>
              )}
            </div>
          )}
        </div>
      </div>

      <aside className="flex flex-col gap-5 rounded-[1.75rem] border border-line p-6">
        <p className="text-[14px] leading-relaxed text-muted">
          These details help the models understand every way {langLabel(lang, languages)} is spoken. All are optional.
        </p>
        <Field label="Dialect" htmlFor="rec-dialect" optional>
          <DialectInput id="rec-dialect" lang={lang} value={dialect} onChange={setDialect} />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Accent" htmlFor="rec-accent" optional>
            <input id="rec-accent" className="field-input" maxLength={60} value={accent} onChange={(e) => setAccent(e.target.value)} />
          </Field>
          <Field label="Region" htmlFor="rec-region" optional>
            <input id="rec-region" className="field-input" maxLength={60} value={region} onChange={(e) => setRegion(e.target.value)} />
          </Field>
          <Field label="Gender" htmlFor="rec-gender" optional>
            <Select id="rec-gender" value={gender} onChange={(e) => setGender(e.target.value)}>
              <option value="">Prefer not to say</option>
              <option value="female">Female</option>
              <option value="male">Male</option>
              <option value="other">Other</option>
            </Select>
          </Field>
          <Field label="Age range" htmlFor="rec-age" optional>
            <Select id="rec-age" value={age} onChange={(e) => setAge(e.target.value)}>
              <option value="">Prefer not to say</option>
              <option value="18-24">18-24</option>
              <option value="25-34">25-34</option>
              <option value="35-44">35-44</option>
              <option value="45-54">45-54</option>
              <option value="55-64">55-64</option>
              <option value="65+">65+</option>
            </Select>
          </Field>
        </div>
        <p className="mt-auto rounded-2xl bg-sunken p-4 text-[13px] leading-relaxed text-muted">
          Record somewhere quiet and speak naturally. Your voice is {profile.voiceCloningOptIn ? "allowed" : "not allowed"} to train
          synthetic voices; change that in your contributor settings.
        </p>
      </aside>
    </div>
  );
}
