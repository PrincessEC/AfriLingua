import { useState } from 'react';
import { translateText } from '../lib/api';

export function TranslatePage() {
  const [source, setSource] = useState('');
  const [target, setTarget] = useState('');
  const [tgtLang, setTgtLang] = useState('yoruba');

  const handleTranslate = async () => {
    const results = await translateText(source, tgtLang);
    if (results.length > 0) setTarget(results[0].text);
  };

  return (
    <div className="max-w-4xl mx-auto p-8">
      <h2 className="text-3xl font-bold mb-6">Translate Text</h2>
      <div className="grid grid-cols-2 gap-4">
        <textarea 
          className="p-4 border rounded-xl"
          value={source} 
          onChange={(e) => setSource(e.target.value)} 
          placeholder="Source text..."
        />
        <div className="p-4 border rounded-xl bg-stone-50">{target}</div>
      </div>
      <button onClick={handleTranslate} className="mt-4 bg-indigo-900 text-white px-6 py-2 rounded-full">Translate</button>
    </div>
  );
}
