import { useQuery } from "@tanstack/react-query";
import { api, type LanguageSpec } from "./api";

/** Built-in registry, mirrored from afrilingua/core/languages.py, used when the engine is offline. */
export const FALLBACK_LANGUAGES: LanguageSpec[] = [
  {
    name: "english",
    nllb_code: "eng_Latn",
    display_name: "English",
    native_name: "English",
    iso639_1: "en",
    iso639_3: "eng",
    mms_asr_code: null,
    mms_tts_model: "facebook/mms-tts-eng",
    whisper_lang: "en",
    aliases: ["en", "eng"],
    supports_asr: true,
    supports_tts: true,
    builtin: true,
  },
  {
    name: "yoruba",
    nllb_code: "yor_Latn",
    display_name: "Yoruba",
    native_name: "Èdè Yorùbá",
    iso639_1: "yo",
    iso639_3: "yor",
    mms_asr_code: "yor",
    mms_tts_model: "facebook/mms-tts-yor",
    whisper_lang: null,
    aliases: ["yo", "yor"],
    supports_asr: true,
    supports_tts: true,
    builtin: true,
  },
  {
    name: "igbo",
    nllb_code: "ibo_Latn",
    display_name: "Igbo",
    native_name: "Asụsụ Igbo",
    iso639_1: "ig",
    iso639_3: "ibo",
    mms_asr_code: "ibo",
    mms_tts_model: "rnjema-unima/mms-tts-ibo-baseline",
    whisper_lang: null,
    aliases: ["ig", "ibo"],
    supports_asr: true,
    supports_tts: true,
    builtin: true,
  },
  {
    name: "hausa",
    nllb_code: "hau_Latn",
    display_name: "Hausa",
    native_name: "Harshen Hausa",
    iso639_1: "ha",
    iso639_3: "hau",
    mms_asr_code: "hau",
    mms_tts_model: "facebook/mms-tts-hau",
    whisper_lang: null,
    aliases: ["ha", "hau"],
    supports_asr: true,
    supports_tts: true,
    builtin: true,
  },
];

/** A greeting per language, used for display flourishes. */
export const GREETINGS: Record<string, string> = {
  english: "Good morning",
  yoruba: "Ẹ káàárọ̀",
  igbo: "Ụtụtụ ọma",
  hausa: "Barka da safiya",
};

export function useLanguages() {
  const q = useQuery({
    queryKey: ["languages"],
    queryFn: ({ signal }) => api.languages(signal),
    staleTime: 5 * 60_000,
    retry: false,
  });
  const languages = q.data && q.data.length > 0 ? q.data : FALLBACK_LANGUAGES;
  return { languages, live: Boolean(q.data && q.data.length > 0), isLoading: q.isLoading, error: q.error };
}

export function langLabel(name: string | undefined | null, languages: LanguageSpec[] = FALLBACK_LANGUAGES): string {
  if (!name) return "";
  if (name === "auto") return "Detect language";
  const spec = languages.find((l) => l.name === name || l.aliases.includes(name));
  if (spec) return spec.display_name;
  return name.charAt(0).toUpperCase() + name.slice(1);
}

export function langShort(name: string, languages: LanguageSpec[] = FALLBACK_LANGUAGES): string {
  const spec = languages.find((l) => l.name === name);
  return (spec?.iso639_1 ?? spec?.iso639_3 ?? name.slice(0, 2)).toUpperCase();
}
