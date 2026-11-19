import { useState, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { usePageContext } from "vike-react/usePageContext";
import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import "../src/index.css";
import { ThemeProvider } from "../src/lib/theme";
import { ToastProvider } from "../src/lib/toast";
import { ErrorBoundary } from "../src/components/ErrorBoundary";

function makeQueryClient() {
  return new QueryClient({
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
}

export default function Wrapper({ children }: { children: ReactNode }) {
  // One client per mounted app (never shared between server renders).
  const [queryClient] = useState(makeQueryClient);
  const { urlPathname } = usePageContext();
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <ToastProvider>
          {/* Keyed by path so a crash on one page does not stick across navigation. */}
          <ErrorBoundary key={urlPathname}>{children}</ErrorBoundary>
        </ToastProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
