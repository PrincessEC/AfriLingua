import { Component, type ErrorInfo, type ReactNode } from "react";
import { RotateCcw } from "lucide-react";

interface State {
  error: Error | null;
}

/** Last line of defence: a page-level crash shows a recoverable message, not a blank screen. */
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("AfriLingua page error", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    const chunkFailed = /dynamically imported module|Loading chunk|Failed to fetch/i.test(this.state.error.message);
    return (
      <div className="mx-auto max-w-xl px-5 py-24 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">This page hit a snag.</h1>
        <p className="mt-3 text-muted">
          {chunkFailed
            ? "A newer version of AfriLingua is available, or the server restarted. Reload to continue."
            : "Something unexpected happened while rendering. Your data is safe; reloading usually fixes it."}
        </p>
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="mt-6 inline-flex h-11 items-center gap-2 rounded-full bg-accent px-5 font-medium text-accent-fg hover:bg-accent-hover"
        >
          <RotateCcw className="size-4" strokeWidth={1.75} /> Reload
        </button>
      </div>
    );
  }
}
