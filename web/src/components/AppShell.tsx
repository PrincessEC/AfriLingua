import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Monitor, Moon, Sun } from "lucide-react";
import { useHealth } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import { cn } from "../lib/utils";
import { Logo, WeaveBand } from "./Brand";

export const NAV_LINKS = [
  { to: "/live", label: "Live" },
  { to: "/translate", label: "Translate" },
  { to: "/contribute", label: "Contribute" },
  { to: "/memory", label: "Memory" },
  { to: "/documents", label: "Documents" },
] as const;

export function EngineIndicator({ className, showLabel = true }: { className?: string; showLabel?: boolean }) {
  const { online, checking, version } = useHealth();
  const label = checking ? "Checking engine" : online ? "Engine online" : "Engine offline";
  return (
    <Link
      to="/status"
      className={cn(
        "inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-[13px] font-medium text-muted ring-1 ring-line transition-colors hover:text-fg hover:ring-line-strong",
        className,
      )}
      title={online && version ? `AfriLingua engine ${version}` : label}
      aria-label={`${label}. Open engine status.`}
    >
      <span className="relative flex size-2">
        {online && <span className="absolute inset-0 animate-ping rounded-full bg-ok opacity-50 motion-reduce:hidden" />}
        <span
          className={cn(
            "relative inline-flex size-2 rounded-full",
            checking ? "bg-subtle" : online ? "bg-ok" : "bg-danger",
          )}
        />
      </span>
      {showLabel && <span>{checking ? "Checking" : online ? "Online" : "Offline"}</span>}
    </Link>
  );
}

export function ThemeToggle({ className }: { className?: string }) {
  const { pref, cycle } = useTheme();
  const Icon = pref === "system" ? Monitor : pref === "light" ? Sun : Moon;
  const next = pref === "system" ? "light" : pref === "light" ? "dark" : "system";
  return (
    <button
      type="button"
      onClick={cycle}
      className={cn(
        "inline-grid size-9 place-items-center rounded-full text-muted transition-[background-color,color,transform] duration-200 ease-out-strong hover:bg-sunken hover:text-fg active:scale-95",
        className,
      )}
      aria-label={`Theme: ${pref}. Switch to ${next}.`}
      title={`Theme: ${pref}`}
    >
      <Icon className="size-[18px]" strokeWidth={1.6} />
    </button>
  );
}

function Hamburger({ open }: { open: boolean }) {
  return (
    <span className="relative block h-3.5 w-5" aria-hidden>
      <span
        className={cn(
          "absolute left-0 h-[1.5px] w-5 rounded bg-current transition-transform duration-300 ease-out-strong",
          open ? "top-1.5 rotate-45" : "top-0.5",
        )}
      />
      <span
        className={cn(
          "absolute left-0 h-[1.5px] w-5 rounded bg-current transition-transform duration-300 ease-out-strong",
          open ? "top-1.5 -rotate-45" : "top-2.5",
        )}
      />
    </span>
  );
}

function MobileMenu({ open, onClose }: { open: boolean; onClose: () => void }) {
  const reduce = useReducedMotion();
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    const o = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = o;
    };
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          id="mobile-menu"
          className="fixed inset-0 z-40 bg-bg/90 backdrop-blur-2xl lg:hidden"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0, transition: { duration: 0.18 } }}
          transition={{ duration: 0.25 }}
        >
          <nav aria-label="Mobile" className="flex h-full flex-col justify-between px-6 pb-10 pt-28">
            <ul className="flex flex-col gap-1">
              {[{ to: "/", label: "Home" }, ...NAV_LINKS, { to: "/status", label: "Engine status" }].map((l, i) => (
                <li key={l.to} className="overflow-hidden">
                  <motion.div
                    initial={reduce ? { opacity: 0 } : { y: 48, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: reduce ? 0 : 0.04 + i * 0.045, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                  >
                    <NavLink
                      to={l.to}
                      end={l.to === "/"}
                      onClick={onClose}
                      className={({ isActive }) =>
                        cn(
                          "block py-2 font-display text-[2.1rem] font-semibold tracking-[-0.03em]",
                          isActive ? "text-accent-text" : "text-fg",
                        )
                      }
                    >
                      {l.label}
                    </NavLink>
                  </motion.div>
                </li>
              ))}
            </ul>
            <div className="flex items-center justify-between">
              <EngineIndicator />
              <ThemeToggle />
            </div>
          </nav>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export function TopNav() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => setOpen(false), [loc.pathname]);
  return (
    <>
      <a
        href="#main"
        className="sr-only z-[80] rounded-full bg-fg px-4 py-2 text-bg focus:not-sr-only focus:fixed focus:left-4 focus:top-4"
      >
        Skip to content
      </a>
      <header className="sticky top-0 z-50 px-3 pt-3 sm:px-5">
        <div className="mx-auto flex h-14 max-w-[1240px] items-center justify-between gap-3 rounded-full border border-line bg-bg/75 pl-4 pr-2 shadow-[0_8px_30px_-18px_rgb(var(--shadow-tint)/0.35)] backdrop-blur-xl supports-[backdrop-filter]:bg-bg/65 sm:pl-5">
          <Link to="/" aria-label="AfriLingua home" className="rounded-full">
            <Logo />
          </Link>
          <nav aria-label="Primary" className="hidden lg:block">
            <ul className="flex items-center gap-0.5">
              {NAV_LINKS.map((l) => (
                <li key={l.to}>
                  <NavLink
                    to={l.to}
                    className={({ isActive }) =>
                      cn(
                        "rounded-full px-3.5 py-2 text-[14.5px] font-medium transition-colors duration-200",
                        isActive ? "bg-sunken text-fg" : "text-muted hover:text-fg",
                      )
                    }
                  >
                    {l.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          <div className="flex items-center gap-1.5">
            <EngineIndicator className="hidden sm:inline-flex" />
            <ThemeToggle className="hidden lg:inline-grid" />
            <button
              type="button"
              className="relative z-50 inline-grid size-10 place-items-center rounded-full text-fg hover:bg-sunken lg:hidden"
              aria-expanded={open}
              aria-controls="mobile-menu"
              aria-label={open ? "Close menu" : "Open menu"}
              onClick={() => setOpen((v) => !v)}
            >
              <Hamburger open={open} />
            </button>
          </div>
        </div>
      </header>
      <MobileMenu open={open} onClose={() => setOpen(false)} />
    </>
  );
}

export function Footer() {
  return (
    <footer className="relative mt-24 border-t border-line">
      <WeaveBand className="opacity-90" />
      <div className="mx-auto grid max-w-[1240px] gap-10 px-5 py-14 md:grid-cols-[1.4fr_1fr_1fr_1fr] md:px-8">
        <div className="max-w-xs">
          <Logo />
          <p className="mt-4 text-sm leading-relaxed text-muted">
            Open, offline speech and translation for African languages. Built to run on the laptop you already have.
          </p>
        </div>
        <FooterCol
          title="Use it"
          links={[
            { to: "/live", label: "Live interpretation" },
            { to: "/translate", label: "Translate text" },
            { to: "/documents", label: "Translate documents" },
          ]}
        />
        <FooterCol
          title="Community"
          links={[
            { to: "/contribute", label: "Contribute" },
            { to: "/memory", label: "Translation memory" },
            { to: "/contribute?tab=stats", label: "Leaderboard" },
          ]}
        />
        <FooterCol
          title="Developers"
          links={[
            { to: "/#developers", label: "MCP and API" },
            { to: "/status", label: "Engine status" },
          ]}
        />
      </div>
      <div className="mx-auto flex max-w-[1240px] flex-col gap-2 px-5 pb-10 text-[13px] text-subtle sm:flex-row sm:items-center sm:justify-between md:px-8">
        <p>MIT licensed code. Community data under CC-BY-4.0 or CC0.</p>
        <p>
          <span lang="yo">Ẹ ṣé o.</span> <span lang="ig">Daalụ.</span> <span lang="ha">Na gode.</span>
        </p>
      </div>
    </footer>
  );
}

function FooterCol({ title, links }: { title: string; links: { to: string; label: string }[] }) {
  return (
    <div>
      <p className="text-sm font-semibold text-fg">{title}</p>
      <ul className="mt-3 space-y-2">
        {links.map((l) => (
          <li key={l.to}>
            <Link to={l.to} className="text-sm text-muted transition-colors hover:text-fg">
              {l.label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Scrolls to top on path change, or to the hash target if present. */
function ScrollManager() {
  const { pathname, hash } = useLocation();
  useEffect(() => {
    if (hash) {
      const el = document.getElementById(hash.slice(1));
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }
    }
    window.scrollTo({ top: 0 });
  }, [pathname, hash]);
  return null;
}

export function AppShell() {
  return (
    <div className="grain relative flex min-h-[100dvh] flex-col">
      <ScrollManager />
      <TopNav />
      <main id="main" className="relative z-[2] flex-1">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}
