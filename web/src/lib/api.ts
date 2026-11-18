export const API_BASE = '/api/v1';

export async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error?.message || `API Error: ${response.statusText}`);
  }

  return response.json();
}

export interface TranslationResult {
  text: string;
  source_text: string;
  src_lang: string;
  tgt_lang: string;
  origin: 'passthrough' | 'cache' | 'tm_exact' | 'mt' | 'mt_tm_assisted';
  latency_ms: number;
}

export async function translateText(
  text: string,
  tgt_lang: string,
  src_lang: string = 'auto'
): Promise<TranslationResult[]> {
  const data = await fetchApi<{ results: TranslationResult[] }>('/translate', {
    method: 'POST',
    body: JSON.stringify({ text, tgt_lang, src_lang }),
  });
  return data.results;
}

export async function submitContribution(payload: any) {
  return await fetchApi('/contributions', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
