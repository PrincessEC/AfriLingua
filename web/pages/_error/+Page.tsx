import NotFound from "../../src/pages/NotFound";

/** Vike error page: unknown URLs are prerendered to 404.html and served with status 404. */
export default function Page() {
  return <NotFound />;
}
