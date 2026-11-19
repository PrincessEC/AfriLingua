import { AdireField } from "../components/Brand";

export default function NotFound() {
  return (
    <div className="relative mx-auto max-w-[1240px] overflow-hidden px-5 py-28 md:px-8">
      <AdireField cols={6} rows={3} size={80} className="pointer-events-none absolute right-0 top-10 hidden h-60 w-[480px] text-fg opacity-[0.07] md:block" />
      <p className="font-mono text-[15px] text-subtle">404</p>
      <h1 className="mt-3 text-4xl font-semibold tracking-[-0.035em] md:text-5xl">This page does not exist.</h1>
      <p className="mt-4 max-w-[48ch] text-lg text-muted">
        The link may be old, or a room may have ended. Check the address, or head back home.
      </p>
      <a
        href="/"
        className="mt-8 inline-flex h-11 items-center rounded-full bg-accent px-5 font-medium text-accent-fg transition-colors hover:bg-accent-hover"
      >
        Back to home
      </a>
    </div>
  );
}
