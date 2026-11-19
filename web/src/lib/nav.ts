import { useCallback, useMemo } from "react";
import { navigate } from "vike/client/router";
import { usePageContext } from "vike-react/usePageContext";

/** Current pathname, normalised without a trailing slash ("/" stays "/"). */
export function usePathname(): string {
  const { urlPathname } = usePageContext();
  return urlPathname.length > 1 ? urlPathname.replace(/\/+$/, "") : urlPathname;
}

/** Current URL hash without the leading "#" (empty when none). */
export function useHash(): string {
  const { urlParsed } = usePageContext();
  // Not available during prerender (no hash in HTTP requests); resolved on the client.
  return typeof window === "undefined" ? "" : (urlParsed.hash ?? "");
}

/**
 * Query-string state, shaped like react-router's useSearchParams. Updates replace the
 * current history entry and keep the scroll position.
 */
export function useSearchParams(): [URLSearchParams, (next: URLSearchParams, opts?: { replace?: boolean }) => void] {
  const { urlParsed } = usePageContext();
  const search = urlParsed.search;
  const params = useMemo(() => new URLSearchParams(search as Record<string, string>), [search]);
  const setParams = useCallback((next: URLSearchParams, opts?: { replace?: boolean }) => {
    const qs = next.toString();
    const url = window.location.pathname + (qs ? `?${qs}` : "") + window.location.hash;
    void navigate(url, { overwriteLastHistoryEntry: opts?.replace ?? false, keepScrollPosition: true });
  }, []);
  return [params, setParams];
}
