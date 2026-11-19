import { lazy, Suspense, type ReactNode } from "react";
import { BrowserRouter, Route, Routes } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "./lib/theme";
import { ToastProvider } from "./lib/toast";
import { AppShell } from "./components/AppShell";
import { Skeleton } from "./components/ui";
import { ErrorBoundary } from "./components/ErrorBoundary";
import Landing from "./pages/Landing";

const Translate = lazy(() => import("./pages/Translate"));
const Live = lazy(() => import("./pages/Live"));
const Room = lazy(() => import("./pages/Room"));
const Contribute = lazy(() => import("./pages/Contribute"));
const Memory = lazy(() => import("./pages/Memory"));
const Documents = lazy(() => import("./pages/Documents"));
const Status = lazy(() => import("./pages/Status"));
const NotFound = lazy(() => import("./pages/NotFound"));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: (count, err) => {
        // Never hammer an engine that is down; the health poll handles recovery.
        const code = (err as { code?: string } | null)?.code;
        if (code === "offline" || code === "not_found" || code === "unauthorized") return false;
        return count < 1;
      },
    },
  },
});

function PageFallback() {
  return (
    <div className="mx-auto max-w-[1240px] px-5 py-14 md:px-8" aria-busy="true" aria-label="Loading page">
      <Skeleton className="h-10 w-72" />
      <Skeleton className="mt-4 h-5 w-full max-w-xl" />
      <div className="mt-10 grid gap-4 md:grid-cols-2">
        <Skeleton className="h-64" />
        <Skeleton className="h-64" />
      </div>
    </div>
  );
}

function Page({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary>
      <Suspense fallback={<PageFallback />}>{children}</Suspense>
    </ErrorBoundary>
  );
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <ToastProvider>
          <BrowserRouter>
            <Routes>
              {/* Listener view is immersive: no site chrome. */}
              <Route
                path="/room/:id"
                element={
                  <Page>
                    <Room />
                  </Page>
                }
              />
              <Route element={<AppShell />}>
                <Route index element={<Landing />} />
                <Route path="translate" element={<Page><Translate /></Page>} />
                <Route path="live" element={<Page><Live /></Page>} />
                <Route path="contribute" element={<Page><Contribute /></Page>} />
                <Route path="memory" element={<Page><Memory /></Page>} />
                <Route path="documents" element={<Page><Documents /></Page>} />
                <Route path="status" element={<Page><Status /></Page>} />
                <Route path="*" element={<Page><NotFound /></Page>} />
              </Route>
            </Routes>
          </BrowserRouter>
        </ToastProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
