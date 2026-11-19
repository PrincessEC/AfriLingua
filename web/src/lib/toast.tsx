import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { CircleCheck, CircleX, Info, X } from "lucide-react";
import { cn } from "./utils";

type ToastKind = "success" | "error" | "info";
interface Toast {
  id: number;
  kind: ToastKind;
  title: string;
  description?: string;
}

interface ToastApi {
  push: (t: Omit<Toast, "id">) => void;
  success: (title: string, description?: string) => void;
  error: (title: string, description?: string) => void;
  info: (title: string, description?: string) => void;
}

const Ctx = createContext<ToastApi | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const idRef = useRef(0);
  const reduce = useReducedMotion();

  const dismiss = useCallback((id: number) => setToasts((ts) => ts.filter((t) => t.id !== id)), []);

  const push = useCallback(
    (t: Omit<Toast, "id">) => {
      const id = ++idRef.current;
      setToasts((ts) => [...ts.slice(-3), { ...t, id }]);
      window.setTimeout(() => dismiss(id), t.kind === "error" ? 7000 : 4200);
    },
    [dismiss],
  );

  const api = useMemo<ToastApi>(
    () => ({
      push,
      success: (title, description) => push({ kind: "success", title, description }),
      error: (title, description) => push({ kind: "error", title, description }),
      info: (title, description) => push({ kind: "info", title, description }),
    }),
    [push],
  );

  return (
    <Ctx.Provider value={api}>
      {children}
      <div
        aria-live="polite"
        aria-relevant="additions"
        className="pointer-events-none fixed inset-x-0 bottom-0 z-[70] flex flex-col items-center gap-2 p-4 sm:items-end sm:p-6"
      >
        <AnimatePresence initial={false}>
          {toasts.map((t) => {
            const Icon = t.kind === "success" ? CircleCheck : t.kind === "error" ? CircleX : Info;
            return (
              <motion.div
                key={t.id}
                layout={!reduce}
                initial={reduce ? { opacity: 0 } : { opacity: 0, y: 16, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={reduce ? { opacity: 0 } : { opacity: 0, y: 8, scale: 0.97, transition: { duration: 0.15 } }}
                transition={{ type: "spring", duration: 0.4, bounce: 0.15 }}
                role={t.kind === "error" ? "alert" : "status"}
                className="pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-2xl border border-line bg-raised/95 p-3.5 pr-2.5 shadow-[0_18px_40px_-16px_rgb(var(--shadow-tint)/0.35)] backdrop-blur"
              >
                <Icon
                  aria-hidden
                  strokeWidth={1.75}
                  className={cn(
                    "mt-0.5 size-5 shrink-0",
                    t.kind === "success" && "text-ok",
                    t.kind === "error" && "text-danger",
                    t.kind === "info" && "text-fg",
                  )}
                />
                <div className="min-w-0 flex-1">
                  <p className="text-[15px] font-semibold text-fg">{t.title}</p>
                  {t.description && <p className="mt-0.5 text-[14px] leading-snug text-muted">{t.description}</p>}
                </div>
                <button
                  type="button"
                  onClick={() => dismiss(t.id)}
                  className="rounded-full p-1 text-subtle transition-colors hover:bg-sunken hover:text-fg"
                  aria-label="Dismiss notification"
                >
                  <X className="size-4" strokeWidth={1.75} />
                </button>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </Ctx.Provider>
  );
}

export function useToast(): ToastApi {
  const v = useContext(Ctx);
  if (!v) throw new Error("useToast must be used inside ToastProvider");
  return v;
}
