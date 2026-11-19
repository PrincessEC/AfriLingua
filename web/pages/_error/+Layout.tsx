import type { ReactNode } from "react";
import { AppShell } from "../../src/components/AppShell";

/** Site chrome (nav + footer) for every page except the immersive room view. */
export default function Layout({ children }: { children: ReactNode }) {
  return <AppShell>{children}</AppShell>;
}
