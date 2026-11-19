import type { ReactNode } from "react";
import { Copy, Check, PlugZap, RefreshCw } from "lucide-react";
import { useState } from "react";
import { errorMessage, isOffline } from "../lib/api";
import { copyText, cn } from "../lib/utils";
import { AdireField } from "./Brand";
import { Button } from "./ui";

/** Inline command with a copy affordance. */
export function CommandChip({ command, className }: { command: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        if (await copyText(command)) {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1600);
        }
      }}
      className={cn(
        "group inline-flex max-w-full items-center gap-2 rounded-lg border border-line-strong bg-sunken py-1 pl-3 pr-1 font-mono text-[14px] text-fg transition-colors hover:border-fg/40",
        className,
      )}
      aria-label={`Copy command: ${command}`}
    >
      <span className="text-subtle">$</span>
      {command}
      <span className="grid size-7 shrink-0 place-items-center rounded-md bg-raised text-muted group-hover:text-fg">
        {copied ? <Check className="size-3.5 text-ok" strokeWidth={2} /> : <Copy className="size-3.5" strokeWidth={1.75} />}
      </span>
    </button>
  );
}

/**
 * The single, friendly state used everywhere the engine cannot be reached.
 * Never a crash, always a next step.
 */
export function EngineOffline({
  title = "The engine is not running",
  description = "This page talks to the local AfriLingua engine. Start it in a terminal and this page reconnects on its own.",
  onRetry,
  compact = false,
  className,
}: {
  title?: string;
  description?: ReactNode;
  onRetry?: () => void;
  compact?: boolean;
  className?: string;
}) {
  return (
    <div
      role="status"
      className={cn(
        "relative overflow-hidden rounded-xl border border-dashed border-line-strong bg-raised/60",
        compact ? "p-5" : "p-8 md:p-10",
        className,
      )}
    >
      {!compact && (
        <AdireField
          cols={4}
          rows={2}
          size={64}
          className="pointer-events-none absolute -right-6 -top-4 h-36 w-72 text-fg opacity-[0.07]"
        />
      )}
      <div className="relative flex flex-col gap-4 sm:flex-row sm:items-start">
        <span className="grid size-10 shrink-0 place-items-center rounded-lg bg-sunken text-fg ring-1 ring-inset ring-line">
          <PlugZap className="size-5" strokeWidth={1.75} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="font-display text-lg font-semibold tracking-tight text-fg">{title}</p>
          <p className="mt-1 max-w-[60ch] text-[16px] leading-relaxed text-muted">{description}</p>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <span className="text-[15px] text-muted">Start the engine:</span>
            <CommandChip command="afrilingua serve" />
            {onRetry && (
              <Button variant="ghost" size="sm" onClick={onRetry} icon={<RefreshCw className="size-4" strokeWidth={1.75} />}>
                Try again
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Shows EngineOffline for offline errors, otherwise a contextual error block. */
export function QueryError({ error, onRetry, compact }: { error: unknown; onRetry?: () => void; compact?: boolean }) {
  if (isOffline(error)) return <EngineOffline onRetry={onRetry} compact={compact} />;
  return (
    <div role="alert" className="rounded-xl border border-danger/30 bg-danger-soft p-5 text-[15px] text-fg">
      <p className="font-semibold text-danger">That request did not go through.</p>
      <p className="mt-1 text-muted">{errorMessage(error)}</p>
      {onRetry && (
        <Button variant="secondary" size="sm" className="mt-3" onClick={onRetry} icon={<RefreshCw className="size-4" strokeWidth={1.75} />}>
          Retry
        </Button>
      )}
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon?: ReactNode;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center rounded-xl border border-dashed border-line-strong px-6 py-12 text-center", className)}>
      {icon && <span className="mb-4 grid size-12 place-items-center rounded-full bg-sunken text-muted">{icon}</span>}
      <p className="font-display text-lg font-semibold tracking-tight text-fg">{title}</p>
      {description && <p className="mt-1.5 max-w-[46ch] text-[16px] leading-relaxed text-muted">{description}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}
