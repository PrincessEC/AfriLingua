import { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';

export function ListenerView() {
  const { id } = useParams();
  const [captions, setCaptions] = useState<string[]>([]);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket(`ws://${window.location.host}/api/v1/rooms/${id}/listen?lang=yoruba&audio=1`);
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'final') {
        setCaptions((prev) => [...prev, data.text]);
      }
    };
    return () => ws.current?.close();
  }, [id]);

  return (
    <div className="bg-stone-950 text-white min-h-screen p-8">
      <h2 className="text-2xl font-bold mb-4">Live Captions</h2>
      <div className="space-y-2">
        {captions.map((c, i) => <p key={i} className="text-xl">{c}</p>)}
      </div>
    </div>
  );
}
