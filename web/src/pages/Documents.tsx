import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { CircleCheck, CircleX, Download, FileText, RotateCcw, Upload, X } from "lucide-react";
import { api, errorMessage, isOffline, type DocumentJob } from "../lib/api";
import { langLabel, useLanguages } from "../lib/languages";
import { useToast } from "../lib/toast";
import { cn, downloadBlob, formatBytes } from "../lib/utils";
import { Button, IconButton, Meter, PageHeader, Pill } from "../components/ui";
import { LanguageSelect } from "../components/LanguageSelect";
import { EngineOffline } from "../components/states";

const ACCEPT = [".txt", ".docx", ".pdf"];
const JOBS_KEY = "afrilingua.docJobs";

interface LocalJob {
  id: string;
  filename: string;
  tgt: string;
  src: string;
  createdAt: number;
}

function loadJobs(): LocalJob[] {
  try {
    return JSON.parse(localStorage.getItem(JOBS_KEY) ?? "[]") as LocalJob[];
  } catch {
    return [];
  }
}

function outputName(filename: string, tgt: string): string {
  const dot = filename.lastIndexOf(".");
  const base = dot > 0 ? filename.slice(0, dot) : filename;
  const ext = dot > 0 ? filename.slice(dot + 1).toLowerCase() : "txt";
  const outExt = ext === "pdf" ? "docx" : ext;
  return `${base}.${tgt}.${outExt}`;
}

export default function Documents() {
  const { languages } = useLanguages();
  const toast = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [src, setSrc] = useState("auto");
  const [tgt, setTgt] = useState("yoruba");
  const [drag, setDrag] = useState(false);
  const [jobs, setJobs] = useState<LocalJob[]>(loadJobs);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    try {
      localStorage.setItem(JOBS_KEY, JSON.stringify(jobs.slice(0, 20)));
    } catch {
      /* ignore */
    }
  }, [jobs]);

  const pick = (f: File | undefined | null) => {
    if (!f) return;
    const ok = ACCEPT.some((ext) => f.name.toLowerCase().endsWith(ext));
    if (!ok) {
      toast.error("Unsupported file", "Upload a .txt, .docx or .pdf file.");
      return;
    }
    setFile(f);
  };

  const upload = useMutation({
    mutationFn: () => api.uploadDocument(file!, tgt, src),
    onSuccess: (r) => {
      setJobs((js) => [{ id: r.id, filename: file!.name, tgt, src, createdAt: Date.now() }, ...js]);
      setFile(null);
      toast.info("Translation started", "You can leave this page; jobs are listed below.");
    },
    onError: (e) => toast.error("Upload failed", isOffline(e) ? "The engine is offline." : errorMessage(e)),
  });

  return (
    <div className="mx-auto max-w-[1240px] px-5 py-12 md:px-8 md:py-16">
      <PageHeader
        title="Translate documents"
        description="Upload a Word document, PDF or text file. Formatting is kept for Word files; PDFs come back as Word documents."
      />

      <div className="grid gap-8 lg:grid-cols-[1.3fr_1fr]">
        <form
          className="shell"
          onSubmit={(e) => {
            e.preventDefault();
            if (file) upload.mutate();
          }}
        >
          <div className="core p-6 md:p-8">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                pick(e.dataTransfer.files[0]);
              }}
              className={cn(
                "flex min-h-[14rem] flex-col items-center justify-center rounded-[1.25rem] border-2 border-dashed px-6 py-10 text-center transition-colors",
                drag ? "border-accent bg-accent-soft" : "border-line-strong",
              )}
            >
              {file ? (
                <div className="flex w-full max-w-md items-center gap-4 rounded-2xl bg-sunken p-4 text-left">
                  <FileText className="size-8 shrink-0 text-accent-text" strokeWidth={1.4} />
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-medium">{file.name}</p>
                    <p className="text-[13px] text-subtle">{formatBytes(file.size)}</p>
                  </div>
                  <IconButton label="Remove file" onClick={() => setFile(null)}>
                    <X className="size-4" strokeWidth={1.75} />
                  </IconButton>
                </div>
              ) : (
                <>
                  <span className="grid size-14 place-items-center rounded-full bg-sunken text-muted">
                    <Upload className="size-6" strokeWidth={1.5} />
                  </span>
                  <p className="mt-4 font-display text-lg font-semibold tracking-tight">Drop a document here</p>
                  <p className="mt-1 text-[14px] text-muted">.docx, .pdf or .txt</p>
                  <Button variant="secondary" size="sm" className="mt-4" onClick={() => input.current?.click()}>
                    Choose file
                  </Button>
                </>
              )}
              <input
                ref={input}
                type="file"
                accept={ACCEPT.join(",")}
                className="sr-only"
                tabIndex={-1}
                aria-label="Document to translate"
                onChange={(e) => pick(e.target.files?.[0])}
              />
            </div>

            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              <LanguageSelect label="From" value={src} onChange={setSrc} languages={languages} allowAuto />
              <LanguageSelect label="To" value={tgt} onChange={setTgt} languages={languages} exclude={src !== "auto" ? [src] : undefined} />
            </div>
            <Button type="submit" size="lg" className="mt-6 w-full sm:w-auto" disabled={!file || src === tgt} loading={upload.isPending}>
              Translate document
            </Button>
          </div>
        </form>

        <section aria-labelledby="jobs-title">
          <div className="flex items-baseline justify-between">
            <h2 id="jobs-title" className="text-lg font-semibold tracking-tight">
              Recent jobs
            </h2>
            {jobs.length > 0 && (
              <button type="button" className="text-[13px] text-subtle hover:text-fg" onClick={() => setJobs([])}>
                Clear list
              </button>
            )}
          </div>
          {jobs.length === 0 ? (
            <p className="mt-4 rounded-[1.25rem] border border-dashed border-line-strong p-6 text-[14.5px] text-muted">
              Jobs you start appear here with live progress. Long PDFs can take a few minutes on a laptop CPU.
            </p>
          ) : (
            <ul className="mt-4 flex flex-col gap-3">
              {jobs.map((j) => (
                <JobRow key={j.id} job={j} languagesLabel={(l) => langLabel(l, languages)} onRemove={() => setJobs((js) => js.filter((x) => x.id !== j.id))} />
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}

function JobRow({ job, languagesLabel, onRemove }: { job: LocalJob; languagesLabel: (l: string) => string; onRemove: () => void }) {
  const toast = useToast();
  const q = useQuery({
    queryKey: ["document", job.id],
    queryFn: ({ signal }) => api.document(job.id, signal),
    refetchInterval: (query) => {
      const d = query.state.data as DocumentJob | undefined;
      if (query.state.error) return 5000;
      return d && (d.status === "done" || d.status === "error") ? false : 1200;
    },
    retry: false,
  });
  const [downloading, setDownloading] = useState(false);
  const d = q.data;
  const status = d?.status;

  const download = async () => {
    setDownloading(true);
    try {
      const blob = await api.downloadDocument(job.id);
      downloadBlob(blob, outputName(d?.filename ?? job.filename, job.tgt));
    } catch (e) {
      toast.error("Download failed", errorMessage(e));
    } finally {
      setDownloading(false);
    }
  };

  return (
    <li className="rounded-[1.25rem] border border-line bg-raised p-4">
      <div className="flex items-start gap-3">
        <FileText className="mt-0.5 size-5 shrink-0 text-subtle" strokeWidth={1.5} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-[15px] font-medium">{d?.filename ?? job.filename}</p>
          <p className="text-[13px] text-subtle">
            {job.src === "auto" ? "Detected source" : languagesLabel(job.src)} to {languagesLabel(job.tgt)}
          </p>
        </div>
        {status === "done" ? (
          <Pill tone="ok">
            <CircleCheck className="size-3.5" strokeWidth={2} /> Done
          </Pill>
        ) : status === "error" ? (
          <Pill tone="danger">
            <CircleX className="size-3.5" strokeWidth={2} /> Failed
          </Pill>
        ) : q.isError ? (
          <Pill>{isOffline(q.error) ? "Engine offline" : "Unknown job"}</Pill>
        ) : (
          <Pill tone="indigo">{status === "running" ? `${Math.round((d?.progress ?? 0) * 100)}%` : "Queued"}</Pill>
        )}
      </div>
      {(status === "running" || status === "queued") && (
        <Meter value={d?.progress ?? 0} className="mt-3" label={`Progress for ${job.filename}`} />
      )}
      {status === "error" && d?.error && <p className="mt-2 text-[13px] text-danger">{d.error}</p>}
      <div className="mt-3 flex justify-end gap-1">
        {status === "done" && (
          <Button size="sm" onClick={() => void download()} loading={downloading} icon={<Download className="size-4" strokeWidth={1.75} />}>
            Download
          </Button>
        )}
        {q.isError && (
          <Button size="sm" variant="ghost" onClick={() => void q.refetch()} icon={<RotateCcw className="size-4" strokeWidth={1.75} />}>
            Retry
          </Button>
        )}
        <Button size="sm" variant="ghost" onClick={onRemove}>
          Remove
        </Button>
      </div>
      {q.isError && isOffline(q.error) && (
        <div className="mt-3">
          <EngineOffline compact description="Progress resumes when the engine is back." />
        </div>
      )}
    </li>
  );
}
