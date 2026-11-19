/**
 * Typed client for the AfriLingua v2 engine.
 * Mirrors docs/api-contract.md exactly. All calls go through same-origin paths
 * (`/api/v1`, `/v1`) so they work both behind the Vite dev proxy and when the
 * FastAPI server serves the built SPA.
 */

// ─── Shared types ──────────────────────────────────────────────────────────

export type LangName = string;

export interface LanguageSpec {
  name: LangName;
  nllb_code: string;
  display_name: string;
  native_name: string;
  iso639_1: string | null;
  iso639_3: string | null;
  mms_asr_code: string | null;
  mms_tts_model: string | null;
  whisper_lang: string | null;
  aliases: string[];
  supports_asr: boolean;
  supports_tts: boolean;
  builtin: boolean;
}

export interface Health {
  status: "ok" | string;
  version: string;
}

export interface RuntimePlan {
  device: "cpu" | "cuda" | string;
  compute_type: "int8" | "float16" | "float32" | "int8_float16" | string;
  torch_dtype?: string;
  profile: "cpu-lite" | "cpu-standard" | "gpu" | string;
  threads?: number;
  model_nllb?: string;
  model_whisper?: string;
  model_mms_asr?: string;
  [key: string]: unknown;
}

export interface HardwareInfo {
  cpu_count?: number;
  ram_gb?: number;
  cuda_available?: boolean;
  cuda_built?: boolean;
  gpu_name?: string;
  vram_gb?: number;
  reason?: string;
  [key: string]: unknown;
}

/**
 * `models` comes from ModelManager.status(), which the contract leaves open.
 * We accept either a map keyed by model id or an array of entries and
 * normalise in the UI (see normaliseModels).
 */
export interface ModelEntry {
  name?: string;
  kind?: string;
  task?: string;
  backend?: string;
  loaded?: boolean;
  load_seconds?: number | null;
  load_s?: number | null;
  device?: string;
  compute_type?: string;
  [key: string]: unknown;
}

export interface EngineStatus {
  runtime: RuntimePlan;
  hardware: HardwareInfo;
  models: Record<string, ModelEntry | boolean> | ModelEntry[];
  languages: LanguageSpec[];
  translation_memory: boolean | Record<string, unknown>;
}

export type Origin = "passthrough" | "cache" | "tm_exact" | "mt" | "mt_tm_assisted";

export interface TMMatch {
  id: string | number;
  source: string;
  target: string;
  score: number;
  approved: boolean;
  origin: string;
  /** "exact" = a translation of this text; "wording" / "meaning" = suggestions only. */
  kind?: "exact" | "wording" | "meaning" | "";
}

export interface TranslationResult {
  text: string;
  source_text: string;
  src_lang: LangName;
  tgt_lang: LangName;
  origin: Origin;
  latency_ms: number;
  detected: boolean | string | null;
  matches: TMMatch[];
}

export interface TranslateRequest {
  text?: string;
  texts?: string[];
  src_lang?: LangName | "auto";
  tgt_lang: LangName;
  mode?: "live" | "document";
  use_tm?: boolean;
}

export interface Transcript {
  text: string;
  language: LangName;
  segments: { start: number; end: number; text: string }[];
  confidence: number;
  duration_s: number;
}

export type JobStatus = "queued" | "running" | "done" | "error";

export interface DocumentJob {
  id: string;
  status: JobStatus;
  progress: number;
  filename: string;
  error?: string | null;
}

export interface TMSearchResponse {
  matches: TMMatch[];
  took_ms: number;
  /** "meaning" when similar-meaning search was used, "fast" for wording only. */
  mode?: "fast" | "meaning";
}

export interface TMEntryIn {
  source: string;
  target: string;
  src_lang: LangName;
  tgt_lang: LangName;
  approved?: boolean;
  origin?: string;
}

export interface TMStats {
  entries: number;
  approved: number;
  by_pair: Record<string, number>;
}

export type PromptKind = "translate" | "record";

export interface Prompt {
  id: string | number;
  text: string;
  lang: LangName;
  domain: string | null;
}

export type ContributionKind = "translation" | "correction" | "glossary" | "transcription";
export type License = "CC-BY-4.0" | "CC0-1.0";
export type Domain = "general" | "medical" | "legal" | "religious" | "education" | "government" | "media";

export interface ContributionIn {
  kind: ContributionKind;
  src_lang: LangName;
  tgt_lang: LangName;
  source_text: string;
  target_text: string;
  original_mt?: string;
  domain?: Domain;
  dialect?: string;
  region?: string;
  contributor?: string;
  license: License;
  consent: true;
}

export type ContributionStatus = "pending" | "approved" | "rejected";

export interface Contribution extends Partial<Omit<ContributionIn, "consent">> {
  id: string | number;
  kind: ContributionKind;
  status: ContributionStatus;
  votes_up: number;
  votes_down: number;
  quality_score: number | null;
  created_at: string;
  has_audio: boolean;
  consent?: boolean;
  /** Audio contributions carry their prompt in `text` + `lang`. */
  text?: string;
  lang?: LangName;
  accent?: string;
  gender?: string;
  age_range?: string;
}

export interface AudioContributionIn {
  audio: Blob;
  filename?: string;
  text: string;
  lang: LangName;
  dialect?: string;
  accent?: string;
  region?: string;
  gender?: string;
  age_range?: string;
  license: License;
  contributor?: string;
  /** Not in the contract yet: explicit, separate opt-in for synthetic-voice training. */
  voice_cloning_opt_in?: boolean;
}

export interface ContributionStats {
  total: number;
  approved: number;
  pending: number;
  by_lang: Record<LangName, { text: number; audio: number; approved: number }>;
  contributors: number;
  hours_audio: number;
  /** Optional, not in contract yet: top contributors for a leaderboard. */
  leaderboard?: { contributor: string; count: number; approved?: number }[];
}

export interface RoomCreate {
  title: string;
  src_lang: LangName;
  targets: LangName[];
  pin?: string;
}

export interface RoomCreated {
  id: string;
  host_token: string;
  join_url: string;
}

export interface RoomInfo {
  id: string;
  title: string;
  src_lang: LangName;
  targets: LangName[];
  listeners: number;
  live: boolean;
}

export type RoomEvent =
  | { type: "status"; live: boolean; listeners: number }
  | { type: "partial"; seq: number; lang: LangName; text: string }
  | { type: "final"; seq: number; lang: LangName; text: string; ts?: number }
  | { type: "translation"; seq: number; lang: LangName; text: string; origin?: Origin | string; latency_ms?: number }
  | { type: "audio"; seq: number; lang: LangName; format: "wav" | string; data: string }
  | { type: "error"; code: string; message: string };

// ─── Errors ────────────────────────────────────────────────────────────────

export type ErrorCode =
  | "unsupported_language"
  | "invalid_input"
  | "capability_unavailable"
  | "model_load_failed"
  | "model_not_downloaded"
  | "not_found"
  | "unauthorized"
  | "payload_too_large"
  | "offline"
  | "http_error"
  | "bad_response";

export class ApiError extends Error {
  readonly code: ErrorCode | string;
  readonly status: number;
  constructor(code: ErrorCode | string, message: string, status = 0) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
  get offline(): boolean {
    return this.code === "offline";
  }
}

export function isOffline(err: unknown): boolean {
  return err instanceof ApiError && err.offline;
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.offline) return "The AfriLingua engine is not reachable.";
    return err.message;
  }
  if (err instanceof Error) return err.message;
  return "Something went wrong.";
}

// ─── Auth (optional API key) ───────────────────────────────────────────────

const KEY_STORAGE = "afrilingua.apiKey";

export function getApiKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

export function setApiKey(key: string): void {
  try {
    if (key) localStorage.setItem(KEY_STORAGE, key);
    else localStorage.removeItem(KEY_STORAGE);
  } catch {
    /* storage unavailable */
  }
}

function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return key ? { Authorization: `Bearer ${key}` } : {};
}

// ─── Core fetch ────────────────────────────────────────────────────────────

export const API_BASE = "/api/v1";

type Query = Record<string, string | number | boolean | undefined | null>;

function withQuery(path: string, query?: Query): string {
  if (!query) return path;
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v === undefined || v === null || v === "") continue;
    qs.set(k, String(v));
  }
  const s = qs.toString();
  return s ? `${path}?${s}` : path;
}

interface RequestOpts {
  method?: string;
  query?: Query;
  json?: unknown;
  form?: FormData;
  signal?: AbortSignal;
  accept?: "json" | "blob" | "none";
}

async function request<T>(path: string, opts: RequestOpts = {}): Promise<T> {
  const { method = "GET", query, json, form, signal, accept = "json" } = opts;
  const headers: Record<string, string> = { ...authHeaders() };
  let body: BodyInit | undefined;
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  } else if (form) {
    body = form;
  }
  if (accept === "json") headers.Accept = "application/json";

  let res: Response;
  try {
    res = await fetch(withQuery(path, query), { method, headers, body, signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError("offline", "Network request failed", 0);
  }

  if (!res.ok) {
    // The dev proxy answers 502/504 (and sometimes 500 with an empty body) when the engine is down.
    let payload: unknown = null;
    try {
      payload = await res.json();
    } catch {
      /* non-JSON error body */
    }
    const err = (payload as { error?: { code?: string; message?: string } } | null)?.error;
    if (err?.code) throw new ApiError(err.code, err.message ?? err.code, res.status);
    const detail = (payload as { detail?: unknown } | null)?.detail;
    if ((res.status === 502 || res.status === 503 || res.status === 504 || res.status === 500) && !payload) {
      throw new ApiError("offline", "Engine unreachable", res.status);
    }
    throw new ApiError(
      res.status === 404 ? "not_found" : res.status === 401 ? "unauthorized" : "http_error",
      typeof detail === "string" ? detail : `Request failed (${res.status})`,
      res.status,
    );
  }

  if (accept === "none" || res.status === 204) return undefined as T;
  if (accept === "blob") return (await res.blob()) as T;
  const ct = res.headers.get("content-type") ?? "";
  if (!ct.includes("json")) {
    // SPA fallback (index.html) means the API route does not exist on this server.
    throw new ApiError("offline", "Engine API not available on this server", res.status);
  }
  try {
    return (await res.json()) as T;
  } catch {
    throw new ApiError("bad_response", "Malformed response from engine", res.status);
  }
}

// ─── Endpoints ─────────────────────────────────────────────────────────────

export const api = {
  // System
  health: (signal?: AbortSignal) => request<Health>(`${API_BASE}/health`, { signal }),
  status: (signal?: AbortSignal) => request<EngineStatus>(`${API_BASE}/status`, { signal }),
  languages: (signal?: AbortSignal) =>
    request<{ languages: LanguageSpec[] }>(`${API_BASE}/languages`, { signal }).then((r) => r.languages),

  // Translation & speech
  translate: (body: TranslateRequest, signal?: AbortSignal) =>
    request<{ results: TranslationResult[] }>(`${API_BASE}/translate`, { method: "POST", json: body, signal }),
  detect: (text: string, signal?: AbortSignal) =>
    request<{ language: LangName; confidence: number }>(`${API_BASE}/detect`, {
      method: "POST",
      json: { text },
      signal,
    }),
  transcribe: (file: Blob, lang: LangName, filename = "audio.webm") => {
    const form = new FormData();
    form.append("file", file, filename);
    form.append("lang", lang);
    return request<Transcript>(`${API_BASE}/transcribe`, { method: "POST", form });
  },
  speech: (text: string, lang: LangName, signal?: AbortSignal) =>
    request<Blob>(`${API_BASE}/speech`, {
      method: "POST",
      json: { text, lang, format: "wav" },
      accept: "blob",
      signal,
    }),

  // Documents
  uploadDocument: (file: File, tgt_lang: LangName, src_lang?: LangName) => {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("tgt_lang", tgt_lang);
    if (src_lang && src_lang !== "auto") form.append("src_lang", src_lang);
    return request<{ id: string; status: JobStatus }>(`${API_BASE}/documents`, { method: "POST", form });
  },
  document: (id: string, signal?: AbortSignal) =>
    request<DocumentJob>(`${API_BASE}/documents/${encodeURIComponent(id)}`, { signal }),
  documentDownloadUrl: (id: string) => `${API_BASE}/documents/${encodeURIComponent(id)}/download`,
  downloadDocument: (id: string) =>
    request<Blob>(`${API_BASE}/documents/${encodeURIComponent(id)}/download`, { accept: "blob" }),

  // Translation memory
  tmSearch: (
    params: { q: string; src_lang: LangName; tgt_lang: LangName; k?: number },
    signal?: AbortSignal,
  ) => request<TMSearchResponse>(`${API_BASE}/tm/search`, { query: { k: 5, ...params }, signal }),
  tmAdd: (entry: TMEntryIn) =>
    request<{ id: string | number }>(`${API_BASE}/tm/entries`, { method: "POST", json: entry }),
  tmApprove: (id: string | number) =>
    request<{ id: string | number; approved: true }>(`${API_BASE}/tm/entries/${encodeURIComponent(String(id))}/approve`, {
      method: "POST",
    }),
  tmDelete: (id: string | number) =>
    request<void>(`${API_BASE}/tm/entries/${encodeURIComponent(String(id))}`, { method: "DELETE", accept: "none" }),
  tmImport: (file: File, src_lang: LangName, tgt_lang: LangName) => {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("src_lang", src_lang);
    form.append("tgt_lang", tgt_lang);
    return request<{ imported: number }>(`${API_BASE}/tm/import`, { method: "POST", form });
  },
  tmStats: (signal?: AbortSignal) => request<TMStats>(`${API_BASE}/tm/stats`, { signal }),

  // Contributions
  prompts: (params: { lang: LangName; kind: PromptKind; n?: number }, signal?: AbortSignal) =>
    request<{ prompts: Prompt[] }>(`${API_BASE}/contributions/prompts`, { query: { n: 10, ...params }, signal }).then(
      (r) => r.prompts,
    ),
  contribute: (body: ContributionIn) =>
    request<Contribution>(`${API_BASE}/contributions`, { method: "POST", json: body }),
  contributeAudio: (input: AudioContributionIn) => {
    const form = new FormData();
    form.append("audio", input.audio, input.filename ?? "recording.webm");
    form.append("text", input.text);
    form.append("lang", input.lang);
    for (const key of ["dialect", "accent", "region", "gender", "age_range", "contributor"] as const) {
      const v = input[key];
      if (v) form.append(key, v);
    }
    form.append("consent", "true");
    form.append("license", input.license);
    form.append("voice_cloning_opt_in", input.voice_cloning_opt_in ? "true" : "false");
    return request<Contribution>(`${API_BASE}/contributions/audio`, { method: "POST", form });
  },
  contributions: (
    params: { status?: ContributionStatus; lang?: LangName; kind?: ContributionKind; limit?: number; offset?: number } = {},
    signal?: AbortSignal,
  ) =>
    request<{ items: Contribution[]; total: number }>(`${API_BASE}/contributions`, {
      query: { limit: 50, offset: 0, ...params },
      signal,
    }),
  reviewQueue: (params: { reviewer: string; lang?: LangName }, signal?: AbortSignal) =>
    request<{ items: Contribution[] }>(`${API_BASE}/contributions/review-queue`, { query: params, signal }).then(
      (r) => r.items,
    ),
  review: (id: string | number, body: { vote: "approve" | "reject"; reviewer: string; comment?: string }) =>
    request<Contribution>(`${API_BASE}/contributions/${encodeURIComponent(String(id))}/review`, {
      method: "POST",
      json: body,
    }),
  contributionAudioUrl: (id: string | number) => `${API_BASE}/contributions/${encodeURIComponent(String(id))}/audio`,
  contributionAudio: (id: string | number) =>
    request<Blob>(`${API_BASE}/contributions/${encodeURIComponent(String(id))}/audio`, { accept: "blob" }),
  contributionStats: (signal?: AbortSignal) => request<ContributionStats>(`${API_BASE}/contributions/stats`, { signal }),
  contributionsExportUrl: (params: { lang?: LangName; kind?: ContributionKind; format?: "jsonl" | "hf" } = {}) =>
    withQuery(`${API_BASE}/contributions/export`, { format: "jsonl", ...params }),

  // Live rooms
  createRoom: (body: RoomCreate) => request<RoomCreated>(`${API_BASE}/rooms`, { method: "POST", json: body }),
  room: (id: string, signal?: AbortSignal) => request<RoomInfo>(`${API_BASE}/rooms/${encodeURIComponent(id)}`, { signal }),
};

// ─── WebSocket URLs ────────────────────────────────────────────────────────

function wsOrigin(): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}`;
}

function withKey(params: URLSearchParams): URLSearchParams {
  const key = getApiKey();
  if (key) params.set("key", key);
  return params;
}

export function hostSocketUrl(roomId: string, token: string): string {
  const qs = withKey(new URLSearchParams({ token }));
  return `${wsOrigin()}${API_BASE}/rooms/${encodeURIComponent(roomId)}/host?${qs}`;
}

export function listenSocketUrl(roomId: string, opts: { lang: LangName; audio: boolean; pin?: string }): string {
  const qs = new URLSearchParams({ lang: opts.lang, audio: opts.audio ? "1" : "0" });
  if (opts.pin) qs.set("pin", opts.pin);
  withKey(qs);
  return `${wsOrigin()}${API_BASE}/rooms/${encodeURIComponent(roomId)}/listen?${qs}`;
}

export function parseRoomEvent(raw: unknown): RoomEvent | null {
  if (typeof raw !== "string") return null;
  try {
    const ev = JSON.parse(raw) as { type?: unknown };
    if (!ev || typeof ev.type !== "string") return null;
    return ev as RoomEvent;
  } catch {
    return null;
  }
}

// ─── Helpers for loosely specified shapes ─────────────────────────────────

export interface NormalisedModel {
  id: string;
  kind: string;
  loaded: boolean;
  loadSeconds: number | null;
  detail: string;
}

export function normaliseModels(models: EngineStatus["models"] | undefined): NormalisedModel[] {
  if (!models) return [];
  const entries: [string, ModelEntry | boolean][] = Array.isArray(models)
    ? models.map((m, i) => [String(m.name ?? `model-${i}`), m])
    : Object.entries(models);
  return entries.map(([id, raw]) => {
    if (typeof raw === "boolean") return { id, kind: guessKind(id), loaded: raw, loadSeconds: null, detail: "" };
    const loadSeconds =
      typeof raw.load_seconds === "number" ? raw.load_seconds : typeof raw.load_s === "number" ? raw.load_s : null;
    const detailParts = [raw.backend, raw.device, raw.compute_type].filter(Boolean) as string[];
    return {
      id: String(raw.name ?? id),
      kind: String(raw.kind ?? raw.task ?? guessKind(id)),
      loaded: Boolean(raw.loaded),
      loadSeconds,
      detail: detailParts.join(" / "),
    };
  });
}

function guessKind(id: string): string {
  const s = id.toLowerCase();
  if (s.includes("nllb") || s.includes("mt") || s.includes("translat")) return "translation";
  if (s.includes("whisper") || s.includes("asr")) return "speech recognition";
  if (s.includes("tts") || s.includes("speech")) return "speech synthesis";
  if (s.includes("detect") || s.includes("lid")) return "language id";
  return "model";
}
