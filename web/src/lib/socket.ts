import { useEffect, useRef, useState } from "react";
import { parseRoomEvent, type RoomEvent } from "./api";

export type SocketState = "idle" | "connecting" | "open" | "reconnecting" | "closed";

interface Options {
  /** Stop retrying after a server-initiated normal close (code 1000) */
  stopOnNormalClose?: boolean;
  maxDelayMs?: number;
}

/**
 * Listener socket with exponential backoff + jitter. Reconnects when the page
 * comes back online or becomes visible. Pass `url = null` to disconnect.
 */
export function useReconnectingSocket(url: string | null, onEvent: (ev: RoomEvent) => void, opts: Options = {}) {
  const { stopOnNormalClose = false, maxDelayMs = 15_000 } = opts;
  const [state, setState] = useState<SocketState>("idle");
  const [attempt, setAttempt] = useState(0);
  const [closeCode, setCloseCode] = useState<number | null>(null);
  const [closeReason, setCloseReason] = useState("");
  const [nextRetryAt, setNextRetryAt] = useState<number | null>(null);
  const handler = useRef(onEvent);
  handler.current = onEvent;

  useEffect(() => {
    if (!url) {
      setState("idle");
      return;
    }
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let tries = 0;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      window.clearTimeout(timer);
      setNextRetryAt(null);
      setState(tries === 0 ? "connecting" : "reconnecting");
      try {
        ws = new WebSocket(url);
      } catch {
        schedule();
        return;
      }
      ws.onopen = () => {
        tries = 0;
        setAttempt(0);
        setCloseCode(null);
        setCloseReason("");
        setState("open");
      };
      ws.onmessage = (e) => {
        const ev = parseRoomEvent(e.data);
        if (ev) handler.current(ev);
      };
      ws.onclose = (e) => {
        ws = null;
        if (disposed) return;
        setCloseCode(e.code);
        setCloseReason(e.reason);
        if (stopOnNormalClose && e.code === 1000) {
          setState("closed");
          return;
        }
        // 4401/4403/4404 style app codes: auth or room missing, do not hammer.
        if (e.code >= 4400 && e.code < 4500) {
          setState("closed");
          return;
        }
        schedule();
      };
      ws.onerror = () => {
        /* onclose follows */
      };
    };

    const schedule = () => {
      if (disposed) return;
      tries += 1;
      setAttempt(tries);
      const base = Math.min(maxDelayMs, 500 * 2 ** Math.min(tries - 1, 6));
      const delay = base / 2 + Math.random() * (base / 2);
      setState("reconnecting");
      setNextRetryAt(Date.now() + delay);
      timer = window.setTimeout(connect, delay);
    };

    const kick = () => {
      if (!ws && !disposed) {
        tries = Math.max(0, tries - 1);
        connect();
      }
    };
    const onVisible = () => {
      if (document.visibilityState === "visible") kick();
    };
    window.addEventListener("online", kick);
    document.addEventListener("visibilitychange", onVisible);

    connect();
    return () => {
      disposed = true;
      window.clearTimeout(timer);
      window.removeEventListener("online", kick);
      document.removeEventListener("visibilitychange", onVisible);
      if (ws) {
        ws.onclose = null;
        ws.close(1000, "client navigated away");
      }
    };
  }, [url, stopOnNormalClose, maxDelayMs]);

  return { state, attempt, nextRetryAt, closeCode, closeReason };
}
