import {
  forwardRef,
  useEffect,
  useId,
  useRef,
  type AnchorHTMLAttributes,
  type ButtonHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
} from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { ArrowRight, LoaderCircle, X } from "lucide-react";
import type { Origin } from "../lib/api";
import { cn, formatMs } from "../lib/utils";

// ─── Buttons ───────────────────────────────────────────────────────────────

type Variant = "primary" | "secondary" | "ghost" | "danger" | "ink" | "accent";
type Size = "sm" | "md" | "lg";

const base =
  "group relative inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded-lg font-medium transition-[background-color,color,box-shadow,transform,opacity] duration-150 ease-out-strong active:translate-y-px disabled:pointer-events-none disabled:opacity-50";

// Primary actions use the orange accent; "ink" is the quieter monochrome option.
// Green is reserved for live / success states.
const variants: Record<Variant, string> = {
  primary: "bg-accent text-accent-fg font-semibold hover:bg-accent-hover",
  secondary: "bg-raised text-fg ring-1 ring-inset ring-line-strong hover:bg-sunken",
  ghost: "text-fg hover:bg-sunken",
  danger: "bg-danger text-white hover:opacity-90",
  ink: "bg-fg text-bg hover:bg-fg/85",
  accent: "bg-accent text-accent-fg hover:bg-accent-hover",
};

const sizes: Record<Size, string> = {
  sm: "h-9 px-3.5 text-[14px]",
  md: "h-11 px-5 text-[16px]",
  lg: "h-13 px-6 text-[17px]",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  icon?: ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "primary", size = "md", loading, icon, className, children, disabled, type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      disabled={disabled || loading}
      className={cn(base, variants[variant], sizes[size], className)}
      {...rest}
    >
      {loading ? <LoaderCircle aria-hidden className="size-4 animate-spin" strokeWidth={2} /> : icon}
      {children}
    </button>
  );
});

/** Call-to-action link: a large button with a trailing arrow that nudges on hover. */
export function CtaLink({
  to,
  children,
  variant = "primary",
  className,
  ...rest
}: { to: string; variant?: "primary" | "secondary" } & Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href">) {
  return (
    <a
      href={to}
      className={cn(base, sizes.lg, variant === "primary" ? variants.primary : variants.secondary, className)}
      {...rest}
    >
      <span>{children}</span>
      <ArrowRight
        aria-hidden
        className="size-4 transition-transform duration-200 ease-out-strong group-hover:translate-x-0.5"
        strokeWidth={1.75}
      />
    </a>
  );
}

export function IconButton({
  label,
  className,
  children,
  ...rest
}: { label: string } & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cn(
        "inline-grid size-9 place-items-center rounded-lg text-muted transition-[background-color,color,transform] duration-150 ease-out-strong hover:bg-sunken hover:text-fg active:translate-y-px disabled:opacity-40",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

// ─── Surfaces ──────────────────────────────────────────────────────────────

export function Panel({ className, coreClassName, children }: { className?: string; coreClassName?: string; children: ReactNode }) {
  return (
    <div className={cn("shell", className)}>
      <div className={cn("core h-full", coreClassName)}>{children}</div>
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cn("animate-pulse rounded-lg bg-sunken", className)} />;
}

// ─── Form fields ───────────────────────────────────────────────────────────

export function Field({
  label,
  help,
  error,
  children,
  htmlFor,
  className,
  optional,
}: {
  label: ReactNode;
  help?: ReactNode;
  error?: ReactNode;
  children: ReactNode;
  htmlFor?: string;
  className?: string;
  optional?: boolean;
}) {
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <label htmlFor={htmlFor} className="field-label">
        {label}
        {optional && <span className="ml-1.5 font-normal text-subtle">optional</span>}
      </label>
      {children}
      {error ? (
        <p className="text-[14px] font-medium text-danger" role="alert">
          {error}
        </p>
      ) : help ? (
        <p className="field-help">{help}</p>
      ) : null}
    </div>
  );
}

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(function Select(
  { className, children, ...rest },
  ref,
) {
  return (
    <div className="relative">
      <select
        ref={ref}
        className={cn("field-input appearance-none pr-10", className)}
        {...rest}
      >
        {children}
      </select>
      <svg
        aria-hidden
        viewBox="0 0 16 16"
        className="pointer-events-none absolute right-3.5 top-1/2 size-4 -translate-y-1/2 text-subtle"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
      >
        <path d="M4 6l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
});

export function Switch({
  checked,
  onChange,
  label,
  description,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  description?: ReactNode;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="min-w-0">
        <label htmlFor={id} className="text-[15px] font-medium text-fg">
          {label}
        </label>
        {description && <p className="mt-0.5 text-[14px] text-subtle">{description}</p>}
      </div>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative mt-0.5 inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors duration-200 ease-out-strong disabled:opacity-50",
          checked ? "bg-accent" : "bg-line-strong",
        )}
      >
        <span
          className={cn(
            "inline-block size-5 rounded-full bg-raised shadow transition-transform duration-200 ease-out-strong",
            checked ? "translate-x-[22px]" : "translate-x-0.5",
          )}
        />
      </button>
    </div>
  );
}

// ─── Badges ────────────────────────────────────────────────────────────────

const ORIGIN_META: Record<string, { label: string; title: string; className: string }> = {
  tm_exact: { label: "Memory match", title: "Exact match from the translation memory", className: "bg-ok-soft text-ok" },
  mt: { label: "Machine", title: "Neural machine translation (int8)", className: "bg-sunken text-fg ring-1 ring-inset ring-line" },
  mt_tm_assisted: {
    label: "Machine + memory",
    title: "Machine translation guided by close memory matches",
    className: "bg-accent-soft text-accent-text",
  },
  cache: { label: "Cache", title: "Served instantly from the recent-results cache", className: "bg-sunken text-muted" },
  passthrough: { label: "Unchanged", title: "Source and target language are the same", className: "bg-sunken text-muted" },
  community: { label: "Community", title: "Approved by community reviewers", className: "bg-accent-soft text-accent-text" },
};

export function OriginBadge({ origin, className }: { origin: Origin | string | undefined; className?: string }) {
  if (!origin) return null;
  const meta = ORIGIN_META[origin] ?? { label: origin, title: origin, className: "bg-sunken text-muted" };
  return (
    <span
      title={meta.title}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[12px] font-semibold",
        meta.className,
        className,
      )}
    >
      {meta.label}
      <span className="font-mono font-normal opacity-70">{origin}</span>
      <span className="sr-only">: {meta.title}</span>
    </span>
  );
}

export function LatencyBadge({ ms, className }: { ms: number | undefined | null; className?: string }) {
  if (ms === undefined || ms === null) return null;
  return (
    <span className={cn("inline-flex items-center rounded-md bg-sunken px-2 py-0.5 font-mono text-[12px] text-muted", className)}>
      {formatMs(ms)}
    </span>
  );
}

export type PillTone = "neutral" | "ok" | "accent" | "danger" | "warn" | "info";

export function Pill({ children, className, tone = "neutral" }: { children: ReactNode; className?: string; tone?: PillTone }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[13px] font-medium",
        tone === "neutral" && "bg-sunken text-muted",
        tone === "ok" && "bg-ok-soft text-ok",
        tone === "accent" && "bg-accent-soft text-accent-text",
        tone === "danger" && "bg-danger-soft text-danger",
        tone === "warn" && "bg-warn-soft text-warn",
        tone === "info" && "bg-sunken text-fg ring-1 ring-inset ring-line-strong",
        className,
      )}
    >
      {children}
    </span>
  );
}

// ─── Tabs (roving, keyboard accessible) ────────────────────────────────────

export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
  label,
  className,
  layoutId,
}: {
  tabs: { id: T; label: ReactNode; icon?: ReactNode }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  className?: string;
  layoutId: string;
}) {
  const reduce = useReducedMotion();
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (e: React.KeyboardEvent, idx: number) => {
    let next = idx;
    if (e.key === "ArrowRight") next = (idx + 1) % tabs.length;
    else if (e.key === "ArrowLeft") next = (idx - 1 + tabs.length) % tabs.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = tabs.length - 1;
    else return;
    e.preventDefault();
    const t = tabs[next];
    if (t) {
      onChange(t.id);
      refs.current[next]?.focus();
    }
  };
  return (
    <div
      role="tablist"
      aria-label={label}
      className={cn("no-scrollbar inline-flex max-w-full gap-1 overflow-x-auto rounded-[10px] bg-sunken p-1", className)}
    >
      {tabs.map((t, i) => {
        const active = t.id === value;
        return (
          <button
            key={t.id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            role="tab"
            type="button"
            id={`${layoutId}-tab-${t.id}`}
            aria-selected={active}
            aria-controls={`${layoutId}-panel-${t.id}`}
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(t.id)}
            onKeyDown={(e) => onKey(e, i)}
            className={cn(
              "relative inline-flex shrink-0 items-center gap-2 rounded-md px-3.5 py-1.5 text-[15px] font-medium transition-colors duration-200",
              active ? "text-fg" : "text-muted hover:text-fg",
            )}
          >
            {active && (
              <motion.span
                layoutId={reduce ? undefined : layoutId}
                className="absolute inset-0 rounded-md bg-raised shadow-[0_1px_2px_rgb(var(--shadow-tint)/0.08),inset_0_0_0_1px_var(--line)]"
                transition={{ type: "spring", duration: 0.35, bounce: 0.12 }}
              />
            )}
            <span className="relative inline-flex items-center gap-2">
              {t.icon}
              {t.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}

// ─── Modal ─────────────────────────────────────────────────────────────────

export function Modal({
  open,
  onClose,
  title,
  children,
  dismissible = true,
  className,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
  dismissible?: boolean;
  className?: string;
}) {
  const reduce = useReducedMotion();
  const ref = useRef<HTMLDivElement>(null);
  const titleId = useId();

  useEffect(() => {
    if (!open) return;
    const prev = document.activeElement as HTMLElement | null;
    const t = window.setTimeout(() => {
      const el = ref.current?.querySelector<HTMLElement>("[data-autofocus], button, input, select, textarea");
      el?.focus();
    }, 30);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && dismissible) onClose();
      if (e.key === "Tab" && ref.current) {
        const f = ref.current.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        );
        if (f.length === 0) return;
        const first = f[0]!;
        const last = f[f.length - 1]!;
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.clearTimeout(t);
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
      prev?.focus?.();
    };
  }, [open, dismissible, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-[60] flex items-end justify-center sm:items-center sm:p-6">
          <motion.div
            className="absolute inset-0 bg-ink/50 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={dismissible ? onClose : undefined}
          />
          <motion.div
            ref={ref}
            role="dialog"
            aria-modal="true"
            aria-labelledby={titleId}
            initial={reduce ? { opacity: 0 } : { opacity: 0, y: 24, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={reduce ? { opacity: 0 } : { opacity: 0, y: 12, scale: 0.98, transition: { duration: 0.15 } }}
            transition={{ duration: 0.32, ease: [0.32, 0.72, 0, 1] }}
            className={cn(
              "relative max-h-[92dvh] w-full max-w-lg overflow-y-auto rounded-t-2xl border border-line bg-raised p-6 shadow-[0_24px_64px_-16px_rgb(var(--shadow-tint)/0.35)] sm:rounded-2xl sm:p-8",
              className,
            )}
          >
            <div className="mb-4 flex items-start justify-between gap-4">
              <h2 id={titleId} className="text-xl font-semibold tracking-tight text-fg">
                {title}
              </h2>
              {dismissible && (
                <IconButton label="Close dialog" onClick={onClose} className="-mr-2 -mt-1 size-9">
                  <X className="size-4" strokeWidth={1.75} />
                </IconButton>
              )}
            </div>
            {children}
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}

// ─── Misc ──────────────────────────────────────────────────────────────────

export function Meter({ value, className, label }: { value: number; className?: string; label?: string }) {
  const v = Math.max(0, Math.min(1, value));
  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(v * 100)}
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-sunken", className)}
    >
      <div
        className="h-full origin-left rounded-full bg-accent transition-transform duration-500 ease-out-strong"
        style={{ transform: `scaleX(${v})` }}
      />
    </div>
  );
}

export function PageHeader({ title, description, actions }: { title: ReactNode; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="mb-8 flex flex-col gap-5 md:mb-10 md:flex-row md:items-end md:justify-between">
      <div className="max-w-2xl">
        <h1 className="text-3xl font-semibold tracking-[-0.035em] text-fg md:text-[2.5rem] md:leading-[1.05]">{title}</h1>
        {description && <p className="mt-3 text-[17px] leading-relaxed text-muted md:text-[18px]">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}
