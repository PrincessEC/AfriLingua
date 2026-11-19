import { useId } from "react";
import { cn } from "../lib/utils";

/**
 * AfriLingua logo. The mark is a single-storey "a" with a dot beneath it:
 * the sub-dot that Yoruba and Igbo use (ẹ, ọ, ụ) and that most keyboards and
 * speech tools still get wrong. Colours follow the theme tokens.
 */
export function LogoMark({ className, title }: { className?: string; title?: string }) {
  return (
    <svg viewBox="0 0 40 40" className={cn("shrink-0", className)} role={title ? "img" : undefined} aria-hidden={title ? undefined : true}>
      {title && <title>{title}</title>}
      <rect width="40" height="40" rx="11" fill="var(--fg)" />
      <circle cx="18.2" cy="18.6" r="6.6" fill="none" stroke="var(--bg)" strokeWidth="3.6" />
      <path d="M25 11.4v14.4" stroke="var(--bg)" strokeWidth="3.6" strokeLinecap="round" />
      <circle cx="20" cy="32.2" r="2.6" fill="var(--accent)" />
    </svg>
  );
}

export function Logo({ className, compact = false }: { className?: string; compact?: boolean }) {
  return (
    <span className={cn("inline-flex items-center gap-2.5", className)}>
      <LogoMark className="size-8" />
      {!compact && (
        <span className="font-display text-[1.2rem] font-semibold leading-none tracking-[-0.03em] text-fg">
          Afri<span className="text-accent-text">Lingua</span>
        </span>
      )}
    </span>
  );
}

// ─── Textile-inspired linework ─────────────────────────────────────────────
// Drawn as thin strokes in currentColor so they sit quietly on any surface.
// References: adire eleko resist-dye tiles (repeating squares, each with its
// own motif), and akwete / kente weave bands. Abstracted, not copied.

function Tile({ kind, x, y, s }: { kind: number; x: number; y: number; s: number }) {
  const c = s / 2;
  const t = `translate(${x} ${y})`;
  switch (kind % 6) {
    case 0: // concentric rings (Olokun-like)
      return (
        <g transform={t}>
          {[0.38, 0.27, 0.16].map((r) => (
            <circle key={r} cx={c} cy={c} r={s * r} fill="none" />
          ))}
          <circle cx={c} cy={c} r={s * 0.04} fill="currentColor" stroke="none" />
        </g>
      );
    case 1: // parallel ripples
      return (
        <g transform={t}>
          {[0.25, 0.42, 0.59, 0.76].map((p) => (
            <path
              key={p}
              d={`M ${s * 0.12} ${s * p} q ${s * 0.19} ${-s * 0.09} ${s * 0.38} 0 t ${s * 0.38} 0`}
              fill="none"
            />
          ))}
        </g>
      );
    case 2: // dotted lattice
      return (
        <g transform={t} stroke="none" fill="currentColor">
          {[0.22, 0.4, 0.58, 0.76].flatMap((px) =>
            [0.22, 0.4, 0.58, 0.76].map((py) => <circle key={`${px}-${py}`} cx={s * px} cy={s * py} r={s * 0.022} />),
          )}
        </g>
      );
    case 3: // quartered cross with seeds
      return (
        <g transform={t}>
          <path d={`M ${c} ${s * 0.14} V ${s * 0.86} M ${s * 0.14} ${c} H ${s * 0.86}`} fill="none" />
          {[
            [0.32, 0.32],
            [0.68, 0.32],
            [0.32, 0.68],
            [0.68, 0.68],
          ].map(([px, py]) => (
            <ellipse key={`${px}${py}`} cx={s * px!} cy={s * py!} rx={s * 0.07} ry={s * 0.035} fill="none" transform={`rotate(45 ${s * px!} ${s * py!})`} />
          ))}
        </g>
      );
    case 4: // stacked chevrons
      return (
        <g transform={t}>
          {[0.3, 0.46, 0.62].map((p) => (
            <path key={p} d={`M ${s * 0.18} ${s * (p + 0.12)} L ${c} ${s * p} L ${s * 0.82} ${s * (p + 0.12)}`} fill="none" />
          ))}
        </g>
      );
    default: // comb / leaf
      return (
        <g transform={t}>
          <path d={`M ${s * 0.2} ${s * 0.8} L ${s * 0.8} ${s * 0.2}`} fill="none" />
          {[0.3, 0.42, 0.54, 0.66].map((p) => (
            <path key={p} d={`M ${s * p} ${s * (1 - p)} l ${s * 0.1} ${s * 0.1} M ${s * p} ${s * (1 - p)} l ${-s * 0.1} ${-s * 0.1}`} fill="none" />
          ))}
        </g>
      );
  }
}

/** A field of adire-style tiles separated by fine "resist" lines. */
export function AdireField({
  cols = 6,
  rows = 4,
  size = 72,
  className,
  strokeWidth = 1,
  offset = 0,
}: {
  cols?: number;
  rows?: number;
  size?: number;
  className?: string;
  strokeWidth?: number;
  offset?: number;
}) {
  const w = cols * size;
  const h = rows * size;
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className={className}
      aria-hidden
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      preserveAspectRatio="xMidYMid slice"
    >
      {Array.from({ length: rows }).flatMap((_, r) =>
        Array.from({ length: cols }).map((__, ci) => (
          <Tile key={`${r}-${ci}`} kind={(r * 2 + ci + offset) % 6} x={ci * size} y={r * size} s={size} />
        )),
      )}
      {Array.from({ length: cols - 1 }).map((_, i) => (
        <line key={`v${i}`} x1={(i + 1) * size} y1={0} x2={(i + 1) * size} y2={h} strokeDasharray="1 5" />
      ))}
      {Array.from({ length: rows - 1 }).map((_, i) => (
        <line key={`h${i}`} x1={0} y1={(i + 1) * size} x2={w} y2={(i + 1) * size} strokeDasharray="1 5" />
      ))}
    </svg>
  );
}

/** Narrow weave band, used as a divider. Colours: indigo, clay, ochre, paper. */
export function WeaveBand({ className }: { className?: string }) {
  const id = useId().replace(/:/g, "");
  return (
    <svg className={cn("block h-2.5 w-full", className)} aria-hidden preserveAspectRatio="none" viewBox="0 0 240 10">
      <defs>
        <pattern id={`weave-${id}`} width="60" height="10" patternUnits="userSpaceOnUse">
          <rect width="60" height="10" fill="var(--indigo)" opacity="0.9" />
          <rect x="6" width="10" height="10" fill="var(--accent)" />
          <rect x="18" width="3" height="10" fill="var(--ochre)" />
          <rect x="23" width="3" height="10" fill="var(--bg)" opacity="0.85" />
          <rect x="34" y="3" width="14" height="4" fill="var(--ochre)" />
          <rect x="52" width="2" height="10" fill="var(--bg)" opacity="0.6" />
        </pattern>
      </defs>
      <rect width="240" height="10" fill={`url(#weave-${id})`} />
    </svg>
  );
}

/** Concentric dashed rings, used behind hero and community art. */
export function Rings({ className, count = 7 }: { className?: string; count?: number }) {
  return (
    <svg viewBox="0 0 400 400" className={className} aria-hidden fill="none" stroke="currentColor">
      {Array.from({ length: count }).map((_, i) => (
        <circle
          key={i}
          cx="200"
          cy="200"
          r={30 + i * 26}
          strokeWidth={i % 3 === 0 ? 1.2 : 0.8}
          strokeDasharray={i % 2 === 0 ? "2 6" : undefined}
        />
      ))}
    </svg>
  );
}
