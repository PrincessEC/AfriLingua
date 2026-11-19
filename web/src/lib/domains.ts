import type { Domain } from "./api";

export const DOMAINS: { id: Domain; label: string }[] = [
  { id: "general", label: "General" },
  { id: "medical", label: "Medical" },
  { id: "legal", label: "Legal" },
  { id: "religious", label: "Religious" },
  { id: "education", label: "Education" },
  { id: "government", label: "Government" },
  { id: "media", label: "Media" },
];
