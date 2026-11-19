import { Skeleton } from "./ui";

/** Skeleton shown in the prerendered shell of client-only pages until they mount. */
export function PageFallback() {
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
