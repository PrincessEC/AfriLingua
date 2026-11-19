import {
  forwardRef,
  useEffect,
  useId,
  useRef,
  type ButtonHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
} from "react";
import { Link, type LinkProps } from "react-router";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { ArrowRight, LoaderCircle, X } from "lucide-react";
import type { Origin } from "../lib/api";
import { cn, formatMs } from "../lib/utils";

// ─── Buttons ───────────────────────────────────────────────────────────────

type Variant = "primary" | "secondary" | "ghost" | "danger" | "ink";
type Size = "sm" | "md" | "lg";

const base =
  "group relative inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded-full font-medium transition-[background-color,color,box-shadow,transform,opacity] duration-200 ease-out-strong active:scale-[0.97] disabled:pointer-events-none disabled:opacity-50";

const variants: Record<Variant, string> = {
  primary: "bg-accent text-accent-fg hover:bg-accent-hover shadow-[inset_0_1px_0_rgb(255_255_255/0.18),0_8px_20px_-10px_var(--accent)]",
  secondary: "bg-raised text-fg ring-1 ring-line-strong hover:bg-sunken",
  ghost: "text-fg hover:bg-sunken",
  danger: "bg-danger text-bg hover:opacity-90",
  ink: "bg-fg text-bg hover:opacity-90",
};

const sizes: Record<Size, string> = {
  sm: "h-9 px-3.5 text-sm",
  md: "h-11 px-5 text-[15px]",
  lg: "h-13 pl-6 pr-6 text-base",
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

/** CTA link with the nested trailing-arrow island. */
export function CtaLink({
  to,
  children,
  variant = "primary",
  className,
  ...rest
}: { variant?: "primary" | "secondary" } & LinkProps) {
  return (
    <Link
      to={to}
      className={cn(
        base,
        "h-13 pl-6 pr-1.5 text-base",
        variant === "primary" ? variants.primary : variants.secondary,
        className,
      )}
      {...rest}
    >
      <span>{children}</span>
      <span
        aria-hidden
        className={cn(
          "ml-2 grid size-10 place-items-center rounded-full transition-transform duration-300 ease-out-strong group-hover:translate-x-0.5 group-hover:-translate-y-px group-hover:scale-105",
          variant === "primary" ? "bg-black/15 dark:bg-black/10" : "bg-sunken",
        )}
      >
        <ArrowRight className="size-4" strokeWidth={1.75} />
      </span>
    </Link>
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
        "inline-grid size-10 place-items-center rounded-full text-muted transition-[background-color,color,transform] duration-200 ease-out-strong hover:bg-sunken hover:text-fg active:scale-95 disabled:opacity-40",
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
        <p className="text-[13px] font-medium text-danger" role="alert">
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
        <label htmlFor={id} className="text-sm font-medium text-fg">
          {label}
        </label>
        {description && <p className="mt-0.5 text-[13px] text-subtle">{description}</p>}
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
  mt: { label: "Machine", title: "Neural machine translation (int8)", className: "bg-indigo-soft text-indigo-text" },
  mt_tm_assisted: {
    label: "Machine + memory",
    title: "Machine translation guided by close memory matches",
    className: "bg-indigo-soft text-indigo-text",
  },
  cache: { label: "Cache", title: "Served instantly from the recent-results cache", className: "bg-ochre-soft text-fg" },
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
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-semibold",
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
    <span className={cn("inline-flex items-center rounded-full bg-sunken px-2.5 py-0.5 font-mono text-[11px] text-muted", className)}>
      {formatMs(ms)}
    </span>
  );
}

export function Pill({ children, className, tone = "neutral" }: { children: ReactNode; className?: string; tone?: "neutral" | "ok" | "accent" | "danger" | "indigo" }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium",
        tone === "neutral" && "bg-sunken text-muted",
        tone === "ok" && "bg-ok-soft text-ok",
        tone === "accent" && "bg-accent-soft text-accent-text",
        tone === "danger" && "bg-danger-soft text-danger",
        tone === "indigo" && "bg-indigo-soft text-indigo-text",
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
      className={cn("no-scrollbar inline-flex max-w-full gap-1 overflow-x-auto rounded-full bg-sunken p-1", className)}
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
              "relative inline-flex shrink-0 items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-colors duration-200",
              active ? "text-fg" : "text-muted hover:text-fg",
            )}
          >
            {active && (
              <motion.span
                layoutId={reduce ? undefined : layoutId}
                className="absolute inset-0 rounded-full bg-raised shadow-[0_1px_2px_rgb(var(--shadow-tint)/0.08),inset_0_0_0_1px_var(--line)]"
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
              "relative max-h-[92dvh] w-full max-w-lg overflow-y-auto rounded-t-[1.75rem] border border-line bg-raised p-6 shadow-2xl sm:rounded-[1.75rem] sm:p-8",
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
        <h1 className="text-3xl font-semibold tracking-[-0.03em] text-fg md:text-[2.6rem] md:leading-[1.05]">{title}</h1>
        {description && <p className="mt-3 text-base leading-relaxed text-muted md:text-[17px]">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}
