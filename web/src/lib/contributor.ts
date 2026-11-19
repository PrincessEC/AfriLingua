import { useCallback, useSyncExternalStore } from "react";
import type { License } from "./api";

/**
 * Contributor identity + consent, stored locally only. Nothing here leaves the
 * device except the display name / anonymous id attached to a contribution.
 */
export interface ContributorProfile {
  id: string; // anonymous, random
  displayName: string; // optional, empty = anonymous
  consent: boolean;
  consentAt: string | null;
  license: License;
  voiceCloningOptIn: boolean;
  dialect: string;
  region: string;
}

const KEY = "afrilingua.contributor";

function randomId(): string {
  const bytes = new Uint8Array(4);
  crypto.getRandomValues(bytes);
  return "anon-" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function defaults(): ContributorProfile {
  return {
    id: randomId(),
    displayName: "",
    consent: false,
    consentAt: null,
    license: "CC-BY-4.0",
    voiceCloningOptIn: false,
    dialect: "",
    region: "",
  };
}

let cache: ContributorProfile | null = null;
const listeners = new Set<() => void>();

function read(): ContributorProfile {
  if (cache) return cache;
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      cache = { ...defaults(), ...(JSON.parse(raw) as Partial<ContributorProfile>) };
      return cache;
    }
  } catch {
    /* fall through */
  }
  cache = defaults();
  try {
    localStorage.setItem(KEY, JSON.stringify(cache));
  } catch {
    /* ignore */
  }
  return cache;
}

function write(next: ContributorProfile) {
  cache = next;
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* ignore */
  }
  listeners.forEach((l) => l());
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function useContributor() {
  const profile = useSyncExternalStore(subscribe, read, read);
  const update = useCallback((patch: Partial<ContributorProfile>) => write({ ...read(), ...patch }), []);
  const name = profile.displayName.trim() || profile.id;
  return { profile, update, name };
}
