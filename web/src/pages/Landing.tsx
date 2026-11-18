import { TryItWidget } from '../components/TryItWidget';
import { motion } from 'motion/react';

export function LandingPage() {
  return (
    <div className="min-h-screen bg-stone-50 p-8">
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-6xl mx-auto flex justify-between items-center mb-16"
      >
        <h1 className="text-3xl font-bold text-indigo-950">AfriLingua</h1>
        <nav className="flex gap-4">
          <a href="/live" className="text-stone-600 hover:text-indigo-900">Live</a>
          <a href="/contribute" className="text-stone-600 hover:text-indigo-900">Contribute</a>
        </nav>
      </motion.header>

      <main className="max-w-6xl mx-auto grid md:grid-cols-2 gap-12 items-center">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <h2 className="text-6xl font-bold text-indigo-950 mb-6">Real-time interpretation for African languages.</h2>
          <p className="text-xl text-stone-600 mb-8">Offline. On any laptop. Empowering your community with fast, accurate translation.</p>
          <div className="flex gap-4">
            <button className="bg-indigo-900 text-white px-8 py-3 rounded-full font-bold hover:bg-indigo-800">Start Interpreting</button>
            <button className="border border-stone-300 px-8 py-3 rounded-full font-bold hover:bg-stone-100">Contribute</button>
          </div>
        </motion.div>
        <TryItWidget />
      </main>
    </div>
  );
}
