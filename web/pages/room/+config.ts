import type { Config } from "vike/types";

/**
 * Immersive listener view: no site chrome, client-only. The room id is dynamic, so only
 * one template page (/room/_) is prerendered; the backend serves that file for any
 * /room/<id> and Room.tsx reads the real id from window.location.
 */
export default {
  title: "Live room · AfriLingua",
  description: "Join a live interpretation room and follow along in your language.",
  ssr: false,
} satisfies Config;
