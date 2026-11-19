/** Prerender a single template page for the dynamic route; see +config.ts. */
export function onBeforePrerenderStart() {
  return ["/room/_"];
}
