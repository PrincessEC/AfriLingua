import { useState } from 'react';
import { submitContribution } from '../lib/api';

export function ContributePage() {
  const [text, setText] = useState('');
  
  const handleSubmit = async (kind: string) => {
    await submitContribution({ kind, src_lang: 'english', tgt_lang: 'yoruba', source_text: 'hello', target_text: text, license: 'CC-BY-4.0', consent: true });
    alert('Submitted!');
  };

  return (
    <div className="max-w-4xl mx-auto p-8">
      <h2 className="text-3xl font-bold mb-6">Community Hub</h2>
      <textarea className="w-full p-4 border rounded-xl mb-4" value={text} onChange={(e) => setText(e.target.value)} placeholder="Enter translation..." />
      <button onClick={() => handleSubmit('translation')} className="bg-indigo-900 text-white px-6 py-2 rounded-full">Submit Translation</button>
    </div>
  );
}
