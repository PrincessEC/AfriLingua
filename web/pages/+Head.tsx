import { usePageContext } from "vike-react/usePageContext";
import { SITE_NAME, SITE_URL } from "../src/lib/site";

// Applies the persisted theme before first paint (no flash). Must stay inline and synchronous.
const THEME_SCRIPT = `(function(){try{var t=localStorage.getItem("afrilingua.theme")||"system";var d=t==="dark"||(t==="system"&&window.matchMedia("(prefers-color-scheme: dark)").matches);if(d)document.documentElement.classList.add("dark")}catch(e){}})();`;

export default function Head() {
  const { urlPathname, is404 } = usePageContext();
  const path = urlPathname.length > 1 ? urlPathname.replace(/\/+$/, "") : urlPathname;
  const indexable = !is404 && !path.startsWith("/room");
  const canonical = SITE_URL + (path === "/" ? "/" : path);
  return (
    <>
      <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
      <meta name="theme-color" content="#FBFBFC" media="(prefers-color-scheme: light)" />
      <meta name="theme-color" content="#050506" media="(prefers-color-scheme: dark)" />
      <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      {indexable ? <link rel="canonical" href={canonical} /> : <meta name="robots" content="noindex" />}
      {indexable && <meta property="og:url" content={canonical} />}
      <meta property="og:type" content="website" />
      <meta property="og:site_name" content={SITE_NAME} />
      <meta property="og:locale" content="en" />
      <meta name="twitter:card" content="summary" />
    </>
  );
}
