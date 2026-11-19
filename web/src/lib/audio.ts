/** Audio helpers: base64 WAV decoding, gapless queued playback, mic capture via AudioWorklet. */

type AudioCtxCtor = typeof AudioContext;

function getAudioContextCtor(): AudioCtxCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as { AudioContext?: AudioCtxCtor; webkitAudioContext?: AudioCtxCtor };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

export function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

/**
 * Plays decoded clips back-to-back on one AudioContext timeline so sentences
 * never overlap. Decoding is serialised so clips keep arrival order. If the
 * backlog grows past `maxLagSeconds` new clips are dropped to stay near-live.
 */
export class WavQueuePlayer {
  private ctx: AudioContext | null = null;
  private gain: GainNode | null = null;
  private nextTime = 0;
  private chain: Promise<void> = Promise.resolve();
  private sources = new Set<AudioBufferSourceNode>();
  private volume = 1;
  constructor(private readonly maxLagSeconds = 12) {}

  get supported(): boolean {
    return getAudioContextCtor() !== null;
  }

  /** Must be called from a user gesture on iOS/Safari. */
  async unlock(): Promise<boolean> {
    const Ctor = getAudioContextCtor();
    if (!Ctor) return false;
    if (!this.ctx) {
      this.ctx = new Ctor();
      this.gain = this.ctx.createGain();
      this.gain.gain.value = this.volume;
      this.gain.connect(this.ctx.destination);
    }
    if (this.ctx.state === "suspended") {
      try {
        await this.ctx.resume();
      } catch {
        return false;
      }
    }
    // Play a 1-sample silent buffer to fully unlock iOS output.
    const b = this.ctx.createBuffer(1, 1, 22050);
    const s = this.ctx.createBufferSource();
    s.buffer = b;
    s.connect(this.ctx.destination);
    s.start(0);
    return this.ctx.state === "running";
  }

  setVolume(v: number) {
    this.volume = v;
    if (this.gain) this.gain.gain.value = v;
  }

  enqueue(data: ArrayBuffer): void {
    this.chain = this.chain.then(() => this.schedule(data)).catch(() => undefined);
  }

  enqueueBase64(b64: string): void {
    try {
      this.enqueue(base64ToArrayBuffer(b64));
    } catch {
      /* malformed base64: skip */
    }
  }

  private async schedule(data: ArrayBuffer): Promise<void> {
    const ctx = this.ctx;
    if (!ctx || !this.gain) return;
    const buffer = await ctx.decodeAudioData(data.slice(0));
    const now = ctx.currentTime;
    if (this.nextTime < now + 0.05) this.nextTime = now + 0.05;
    if (this.nextTime - now > this.maxLagSeconds) return; // too far behind: drop to stay live
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.gain);
    src.start(this.nextTime);
    this.nextTime += buffer.duration + 0.12; // small breath between sentences
    this.sources.add(src);
    src.onended = () => this.sources.delete(src);
  }

  /** Stop everything queued, keep the context alive. */
  clear(): void {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
    }
    this.sources.clear();
    this.nextTime = 0;
  }

  async close(): Promise<void> {
    this.clear();
    if (this.ctx) {
      try {
        await this.ctx.close();
      } catch {
        /* ignore */
      }
    }
    this.ctx = null;
    this.gain = null;
  }
}

let currentAudio: HTMLAudioElement | null = null;

/** One-shot playback of a Blob (e.g. /api/v1/speech result). Stops any previous clip. */
export function playBlob(blob: Blob): Promise<void> {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  currentAudio = audio;
  return new Promise((resolve, reject) => {
    audio.onended = () => {
      URL.revokeObjectURL(url);
      resolve();
    };
    audio.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not play audio"));
    };
    audio.play().catch((e: unknown) => {
      URL.revokeObjectURL(url);
      reject(e instanceof Error ? e : new Error("Playback blocked"));
    });
  });
}

export function stopBlobPlayback() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
}

// ─── Microphone capture (AudioWorklet → PCM16 LE @ 16 kHz) ─────────────────

export interface MicCaptureHandlers {
  onPcm: (buf: ArrayBuffer) => void;
  onLevel?: (rms: number, peak: number) => void;
}

export class MicCapture {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private sink: GainNode | null = null;
  inputSampleRate = 0;
  deviceLabel = "";

  static get supported(): boolean {
    return (
      typeof navigator !== "undefined" &&
      !!navigator.mediaDevices?.getUserMedia &&
      typeof AudioWorkletNode !== "undefined"
    );
  }

  async start(handlers: MicCaptureHandlers, deviceId?: string): Promise<void> {
    const Ctor = getAudioContextCtor();
    if (!Ctor || !MicCapture.supported) throw new Error("This browser cannot capture audio with AudioWorklet.");
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    this.deviceLabel = this.stream.getAudioTracks()[0]?.label ?? "";
    this.ctx = new Ctor({ latencyHint: "interactive" });
    this.inputSampleRate = this.ctx.sampleRate;
    await this.ctx.audioWorklet.addModule("/pcm-worklet.js");
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "pcm-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
      channelCountMode: "explicit",
    });
    this.node.port.onmessage = (e: MessageEvent<{ type: string; buffer?: ArrayBuffer; rms?: number; peak?: number }>) => {
      const d = e.data;
      if (d.type === "pcm" && d.buffer) handlers.onPcm(d.buffer);
      else if (d.type === "level" && handlers.onLevel) handlers.onLevel(d.rms ?? 0, d.peak ?? 0);
    };
    // Keep the node pulled by the graph without making the mic audible.
    this.sink = this.ctx.createGain();
    this.sink.gain.value = 0;
    this.source.connect(this.node);
    this.node.connect(this.sink);
    this.sink.connect(this.ctx.destination);
    if (this.ctx.state === "suspended") await this.ctx.resume();
  }

  flush() {
    this.node?.port.postMessage({ type: "flush" });
  }

  async stop(): Promise<void> {
    try {
      this.flush();
    } catch {
      /* ignore */
    }
    this.source?.disconnect();
    this.node?.disconnect();
    this.sink?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    if (this.ctx && this.ctx.state !== "closed") {
      try {
        await this.ctx.close();
      } catch {
        /* ignore */
      }
    }
    this.ctx = null;
    this.stream = null;
    this.node = null;
    this.source = null;
    this.sink = null;
  }
}

export async function listMicrophones(): Promise<MediaDeviceInfo[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const all = await navigator.mediaDevices.enumerateDevices();
  return all.filter((d) => d.kind === "audioinput");
}

/** Best MediaRecorder mime type for contribution recordings. */
export function pickRecorderMime(): string {
  if (typeof MediaRecorder === "undefined") return "";
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  return candidates.find((m) => MediaRecorder.isTypeSupported(m)) ?? "";
}
