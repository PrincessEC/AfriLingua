import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { translateText, TranslationResult } from '../lib/api';
import { Loader2, AlertCircle } from 'lucide-react';

export function TryItWidget() {
  const [text, setText] = useState('');
  const [tgtLang, setTgtLang] = useState('yoruba');
  const [enabled, setEnabled] = useState(false);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['translate', text, tgtLang],
    queryFn: () => translateText(text, tgtLang),
    enabled: enabled,
  });

  const handleTranslate = () => {
    setEnabled(true);
    refetch();
  };

  return (
    <div className="bg-white p-6 rounded-2xl shadow-xl border border-stone-100">
      <h3 className="text-lg font-bold mb-4">Try AfriLingua v2</h3>
      <textarea
        className="w-full p-3 border border-stone-200 rounded-lg mb-4"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text..."
      />
      <div className="flex gap-2 mb-4">
        <select value={tgtLang} onChange={(e) => setTgtLang(e.target.value)} className="p-2 border rounded-lg">
          <option value="yoruba">Yoruba</option>
          <option value="igbo">Igbo</option>
          <option value="hausa">Hausa</option>
        </select>
        <button onClick={handleTranslate} className="bg-indigo-900 text-white px-4 py-2 rounded-lg hover:bg-indigo-800">
          Translate
        </button>
      </div>
      
      {isLoading && <Loader2 className="animate-spin" />}
      {error && <div className="text-red-500 flex items-center gap-2"><AlertCircle /> {error.message}</div>}
      {data && data.length > 0 && (
        <div className="bg-stone-50 p-4 rounded-lg mt-4">
          <p className="text-stone-800">{data[0].text}</p>
          <span className="text-xs text-stone-500 uppercase mt-2 block">{data[0].origin}</span>
        </div>
      )}
    </div>
  );
}
