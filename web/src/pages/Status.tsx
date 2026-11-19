import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Cpu, Database, HardDrive, Key, RefreshCw, Server } from "lucide-react";
import { api, getApiKey, normaliseModels, setApiKey } from "../lib/api";
import { useHealth } from "../lib/hooks";
import { useToast } from "../lib/toast";
import { formatNumber } from "../lib/utils";
import { Button, Field, PageHeader, Pill, Skeleton } from "../components/ui";
import { QueryError } from "../components/states";

const PROFILE_COPY: Record<string, string> = {
  "cpu-lite": "Smallest models, tuned for laptops with limited memory.",
  "cpu-standard": "Balanced int8 models for a typical 4 to 8 core laptop.",
  gpu: "Larger models accelerated on the GPU.",
};

export default function Status() {
  const { online, version } = useHealth();
  const status = useQuery({
    queryKey: ["status"],
    queryFn: ({ signal }) => api.status(signal),
    refetchInterval: 15_000,
    retry: false,
  });
  const tm = useQuery({ queryKey: ["tm", "stats"], queryFn: ({ signal }) => api.tmStats(signal), retry: false, enabled: status.isSuccess });

  const s = status.data;
  const models = normaliseModels(s?.models);
  const rt = s?.runtime;
  const hw = s?.hardware;

  return (
    <div className="mx-auto max-w-[1240px] px-5 py-12 md:px-8 md:py-16">
      <PageHeader
        title="Engine status"
        description="What is running on this machine, which models are loaded, and what each language can do."
        actions={
          <>
            <Pill tone={online ? "ok" : "danger"}>{online ? `Online${version ? `, v${version}` : ""}` : "Offline"}</Pill>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void status.refetch()}
              loading={status.isFetching}
              icon={<RefreshCw className="size-4" strokeWidth={1.75} />}
            >
              Refresh
            </Button>
          </>
        }
      />

      {status.isError ? (
        <QueryError error={status.error} onRetry={() => void status.refetch()} />
      ) : (
        <div className="grid gap-6">
          {/* Runtime + hardware */}
          <div className="grid gap-px overflow-hidden rounded-2xl bg-line ring-1 ring-line md:grid-cols-4">
            {[
              { icon: Server, label: "Runtime profile", value: rt?.profile, sub: rt ? PROFILE_COPY[rt.profile] ?? "" : "" },
              { icon: Cpu, label: "Device", value: rt ? rt.device.toUpperCase() : undefined, sub: hw?.gpu_name || (hw?.cpu_count ? `${hw.cpu_count} CPU cores` : "") },
              { icon: HardDrive, label: "Compute type", value: rt?.compute_type, sub: rt?.threads ? `${rt.threads} threads` : "" },
              {
                icon: Database,
                label: "Memory",
                value: hw?.ram_gb ? `${hw.ram_gb.toFixed(1)} GB` : rt ? "-" : undefined,
                sub: hw?.vram_gb ? `${hw.vram_gb.toFixed(1)} GB VRAM` : hw?.cuda_available === false ? "No CUDA GPU" : "",
              },
            ].map(({ icon: Icon, label, value, sub }) => (
              <div key={label} className="bg-raised p-6">
                <p className="flex items-center gap-2 text-[14px] text-subtle">
                  <Icon className="size-4" strokeWidth={1.6} aria-hidden /> {label}
                </p>
                <p className="mt-2 font-display text-2xl font-semibold tracking-tight">
                  {value ?? <Skeleton className="h-7 w-24" />}
                </p>
                {sub && <p className="mt-1 text-[14px] leading-snug text-muted">{sub}</p>}
              </div>
            ))}
          </div>
          {hw?.reason && <p className="-mt-2 text-[14px] text-subtle">Hardware note: {hw.reason}</p>}

          <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
            {/* Models */}
            <section className="rounded-2xl border border-line p-6" aria-labelledby="models-title">
              <h2 id="models-title" className="text-lg font-semibold tracking-tight">
                Models
              </h2>
              <p className="mt-1 text-[15px] text-muted">Models load on first use and stay in memory.</p>
              {status.isPending ? (
                <div className="mt-5 space-y-3">
                  {[0, 1, 2].map((i) => (
                    <Skeleton key={i} className="h-12" />
                  ))}
                </div>
              ) : models.length === 0 ? (
                <p className="mt-5 text-[16px] text-muted">No models registered yet.</p>
              ) : (
                <ul className="mt-5 divide-y divide-line">
                  {models.map((m) => (
                    <li key={m.id} className="flex flex-wrap items-center gap-3 py-3">
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-mono text-[15px] text-fg">{m.id}</p>
                        <p className="text-[14px] text-subtle">
                          {m.kind}
                          {m.detail && ` / ${m.detail}`}
                        </p>
                      </div>
                      {m.loadSeconds !== null && <span className="font-mono text-[13px] text-muted">loaded in {m.loadSeconds.toFixed(1)} s</span>}
                      <Pill tone={m.loaded ? "ok" : "neutral"}>{m.loaded ? "Loaded" : "Not loaded"}</Pill>
                    </li>
                  ))}
                </ul>
              )}
              {rt && (
                <dl className="mt-5 grid gap-2 rounded-2xl bg-sunken p-4 font-mono text-[13px] text-muted sm:grid-cols-[auto_1fr] sm:gap-x-4">
                  {[
                    ["translation", rt.model_nllb],
                    ["speech (whisper)", rt.model_whisper],
                    ["speech (mms)", rt.model_mms_asr],
                  ]
                    .filter(([, v]) => v)
                    .map(([k, v]) => (
                      <div key={k} className="contents">
                        <dt className="text-subtle">{k}</dt>
                        <dd className="break-all text-fg">{v}</dd>
                      </div>
                    ))}
                </dl>
              )}
            </section>

            <div className="flex flex-col gap-6">
              <section className="rounded-2xl border border-line p-6" aria-labelledby="tm-status">
                <h2 id="tm-status" className="text-lg font-semibold tracking-tight">
                  Translation memory
                </h2>
                {s && !s.translation_memory ? (
                  <p className="mt-2 text-[16px] text-muted">Disabled. The engine is running without a memory database.</p>
                ) : (
                  <dl className="mt-4 grid grid-cols-2 gap-4">
                    <div>
                      <dt className="text-[14px] text-subtle">Entries</dt>
                      <dd className="font-display text-2xl font-semibold">{tm.data ? formatNumber(tm.data.entries) : "-"}</dd>
                    </div>
                    <div>
                      <dt className="text-[14px] text-subtle">Approved</dt>
                      <dd className="font-display text-2xl font-semibold">{tm.data ? formatNumber(tm.data.approved) : "-"}</dd>
                    </div>
                  </dl>
                )}
              </section>
              <ApiKeyCard />
            </div>
          </div>

          {/* Languages */}
          <section className="rounded-2xl border border-line p-6" aria-labelledby="lang-cap">
            <h2 id="lang-cap" className="text-lg font-semibold tracking-tight">
              Languages and capabilities
            </h2>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[36rem] text-left text-[15px]">
                <thead className="text-[13px] text-subtle">
                  <tr className="border-b border-line">
                    <th className="py-2 pr-4 font-medium">Language</th>
                    <th className="py-2 pr-4 font-medium">Code</th>
                    <th className="py-2 pr-4 font-medium">Text</th>
                    <th className="py-2 pr-4 font-medium">Speech in</th>
                    <th className="py-2 font-medium">Voice out</th>
                  </tr>
                </thead>
                <tbody>
                  {(s?.languages ?? []).map((l) => (
                    <tr key={l.name} className="border-b border-line last:border-0">
                      <td className="py-3 pr-4">
                        <span className="font-medium">{l.display_name}</span>
                        {l.native_name && l.native_name !== l.display_name && <span className="ml-2 text-subtle">{l.native_name}</span>}
                      </td>
                      <td className="py-3 pr-4 font-mono text-[13px] text-muted">{l.nllb_code}</td>
                      <td className="py-3 pr-4">
                        <Pill tone="ok">Yes</Pill>
                      </td>
                      <td className="py-3 pr-4">
                        <Pill tone={l.supports_asr ? "ok" : "neutral"}>
                          {l.supports_asr ? (l.whisper_lang ? "Whisper" : "MMS") : "No"}
                        </Pill>
                      </td>
                      <td className="py-3">
                        <Pill tone={l.supports_tts ? "ok" : "neutral"}>{l.supports_tts ? "Yes" : "No"}</Pill>
                      </td>
                    </tr>
                  ))}
                  {status.isPending && (
                    <tr>
                      <td colSpan={5} className="py-3">
                        <Skeleton className="h-8" />
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}

      {status.isError && (
        <div className="mt-6 max-w-md">
          <ApiKeyCard />
        </div>
      )}
    </div>
  );
}

function ApiKeyCard() {
  const toast = useToast();
  const [key, setKey] = useState(getApiKey);
  return (
    <form
      className="rounded-2xl border border-line p-6"
      onSubmit={(e) => {
        e.preventDefault();
        setApiKey(key.trim());
        toast.success(key.trim() ? "API key saved on this device" : "API key removed");
      }}
    >
      <h2 className="flex items-center gap-2 text-lg font-semibold tracking-tight">
        <Key className="size-4 text-accent-text" strokeWidth={1.6} /> API key
      </h2>
      <Field
        className="mt-3"
        label="Key"
        htmlFor="api-key"
        optional
        help="Only needed when the server sets API_KEYS. Stored in this browser only."
      >
        <input
          id="api-key"
          type="password"
          autoComplete="off"
          className="field-input font-mono"
          value={key}
          onChange={(e) => setKey(e.target.value)}
        />
      </Field>
      <Button type="submit" size="sm" variant="secondary" className="mt-4">
        Save
      </Button>
    </form>
  );
}
