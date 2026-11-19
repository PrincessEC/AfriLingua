import type { Config } from "vike/types";
import vikeReact from "vike-react/config";
import { PageFallback } from "../src/components/PageFallback";

/**
 * Global Vike config. Production is static: every route is prerendered at build time
 * (no Node at runtime). Pages that need browser-only APIs set `ssr: false` in their own
 * +config.ts: they are prerendered as an HTML shell and rendered on the client.
 */
export default {
  extends: [vikeReact],
  prerender: true,
  lang: "en",
  favicon: "/favicon.svg",
  // vike-react only emits a fixed viewport tag; any other value disables it so +Head can
  // emit one with viewport-fit=cover (the room view uses env(safe-area-inset-*)).
  viewport: false as unknown as "responsive",
  title: "AfriLingua · Fast, accurate AI translation for African languages",
  description:
    "Fast, accurate AI translation for African languages. Speech and text for Igbo, Yoruba, Hausa and more, offline on an ordinary laptop CPU.",
  Loading: { component: PageFallback },
} satisfies Config;
