import { useEffect, useState, type ReactNode } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Monitor, Moon, Sun } from "lucide-react";
import { useHealth } from "../lib/hooks";
import { useHash, usePathname } from "../lib/nav";
import { useTheme } from "../lib/theme";
import { cn } from "../lib/utils";
import { GithubMark, Logo } from "./Brand";

export const REPO_URL = "https://github.com/fegig/AfriLingua";

export const NAV_LINKS = [
  { to: "/translate", label: "Translate" },
  { to: "/documents", label: "Documents" },
  { to: "/live", label: "Live" },
  { to: "/memory", label: "Memory" },
  { to: "/contribute", label: "Contribute" },
] as const;

/** Active-link check for nav items: exact match for "/", otherwise path or sub-path. */
function isActivePath(pathname: string, to: string): boolean {
  const target = to.split("#")[0] || "/";
  return target === "/" ? pathname === "/" : pathname === target || pathname.startsWith(target + "/");
}

export function EngineIndicator({ className, showLabel = true }: { className?: string; showLabel?: boolean }) {
  const { online, checking, version } = useHealth();
  const label = checking ? "Checking engine" : online ? "Engine online" : "Engine offline";
  return (
    <a
      href="/status"
      className={cn(
        "inline-flex h-8 items-center gap-2 rounded-lg px-2.5 text-[14px] font-medium text-muted ring-1 ring-inset ring-line transition-colors hover:text-fg hover:ring-line-strong",
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
    </a>
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
        "inline-grid size-8 place-items-center rounded-lg text-muted transition-[background-color,color] duration-150 ease-out-strong hover:bg-sunken hover:text-fg",
        className,
      )}
      aria-label={`Theme: ${pref}. Switch to ${next}.`}
      title={`Theme: ${pref}`}
    >
      <Icon className="size-4" strokeWidth={1.75} />
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
  const pathname = usePathname();
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
                    <a
                      href={l.to}
                      onClick={onClose}
                      aria-current={isActivePath(pathname, l.to) ? "page" : undefined}
                      className={cn(
                        "block py-2 font-display text-[2.1rem] font-semibold tracking-[-0.03em]",
                        isActivePath(pathname, l.to) ? "text-accent-text" : "text-fg",
                      )}
                    >
                      {l.label}
                    </a>
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
  const pathname = usePathname();
  useEffect(() => setOpen(false), [pathname]);
  return (
    <>
      <a
        href="#main"
        className="sr-only z-[80] rounded-lg bg-fg px-4 py-2 text-bg focus:not-sr-only focus:fixed focus:left-4 focus:top-4"
      >
        Skip to content
      </a>
      <header className="sticky top-0 z-50 border-b border-line bg-bg/80 backdrop-blur-xl supports-[backdrop-filter]:bg-bg/70">
        <div className="mx-auto flex h-14 max-w-[1240px] items-center justify-between gap-4 px-5 md:px-8">
          <div className="flex items-center gap-8">
            <a href="/" aria-label="AfriLingua home" className="rounded-md">
              <Logo />
            </a>
            <nav aria-label="Primary" className="hidden lg:block">
              <ul className="flex items-center gap-1">
                {NAV_LINKS.map((l) => (
                  <li key={l.to}>
                    <a
                      href={l.to}
                      aria-current={isActivePath(pathname, l.to) ? "page" : undefined}
                      className={cn(
                        "relative block rounded-md px-2.5 py-1.5 text-[15px] font-medium transition-colors duration-150",
                        isActivePath(pathname, l.to) ? "text-fg" : "text-muted hover:text-fg",
                      )}
                    >
                      {l.label}
                      {isActivePath(pathname, l.to) && (
                        <span aria-hidden className="absolute inset-x-2.5 -bottom-[13px] h-px bg-fg" />
                      )}
                    </a>
                  </li>
                ))}
              </ul>
            </nav>
          </div>
          <div className="flex items-center gap-1.5">
            <EngineIndicator className="hidden sm:inline-flex" />
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="hidden size-8 place-items-center rounded-lg text-muted transition-colors hover:bg-sunken hover:text-fg lg:inline-grid"
              aria-label="AfriLingua on GitHub"
              title="GitHub"
            >
              <GithubMark className="size-4" />
            </a>
            <ThemeToggle className="hidden lg:inline-grid" />
            <button
              type="button"
              className="relative z-50 inline-grid size-9 place-items-center rounded-lg text-fg hover:bg-sunken lg:hidden"
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
      <div className="mx-auto grid max-w-[1240px] gap-10 px-5 py-14 md:grid-cols-[1.6fr_1fr_1fr_1fr] md:px-8">
        <div className="max-w-xs">
          <Logo />
          <p className="mt-4 text-[15px] leading-relaxed text-muted">
            Fast, offline translation for African languages.
          </p>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="mt-5 inline-flex items-center gap-2 text-[15px] font-medium text-fg underline-offset-4 hover:underline"
          >
            <GithubMark className="size-4" /> Star on GitHub
          </a>
        </div>
        <FooterCol
          title="Product"
          links={[
            { to: "/translate", label: "Translate text" },
            { to: "/documents", label: "Translate documents" },
            { to: "/live", label: "Live interpretation" },
          ]}
        />
        <FooterCol
          title="Developers"
          links={[
            { to: "/#developers", label: "Python, API and MCP" },
            { to: "/#speed", label: "Speed and accuracy" },
            { to: "/status", label: "Engine status" },
          ]}
        />
        <FooterCol
          title="Community"
          links={[
            { to: "/contribute", label: "Contribute" },
            { to: "/#train", label: "Train on your data" },
            { to: "/memory", label: "Translation memory" },
          ]}
        />
      </div>
      <div className="mx-auto flex max-w-[1240px] flex-col gap-2 border-t border-line px-5 py-6 text-[14px] text-subtle sm:flex-row sm:items-center sm:justify-between md:px-8">
        <p>MIT licensed.</p>
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
      <p className="text-[15px] font-semibold text-fg">{title}</p>
      <ul className="mt-3 space-y-2">
        {links.map((l) => (
          <li key={l.to}>
            <a href={l.to} className="text-[15px] text-muted transition-colors hover:text-fg">
              {l.label}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Scrolls to top on path change, or to the hash target if present. */
function ScrollManager() {
  const pathname = usePathname();
  const hash = useHash();
  useEffect(() => {
    // Vike resets scroll / jumps to the hash after render; run after it so smooth scrolling wins.
    const t = window.setTimeout(() => {
      if (hash) {
        const el = document.getElementById(hash);
        if (el) {
          el.scrollIntoView({ behavior: "smooth", block: "start" });
          return;
        }
      }
      window.scrollTo({ top: 0 });
    }, 0);
    return () => window.clearTimeout(t);
  }, [pathname, hash]);
  return null;
}

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="relative flex min-h-[100dvh] flex-col">
      <ScrollManager />
      <TopNav />
      <main id="main" className="relative z-[2] flex-1">
        {children}
      </main>
      <Footer />
    </div>
  );
}
