import React from 'react';
import { RiskLevel } from '../types';
import { Flame } from 'lucide-react';

interface RiskGaugeProps {
  score: number;
  level: RiskLevel;
  fusionMethod: string;
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({ score, level, fusionMethod }) => {
  const pct = Math.max(0, Math.min(100, score));
  const tone =
    pct < 30 ? { bar: 'bg-[#22ff88]', text: 'text-[#22ff88]', glow: 'shadow-[0_0_18px_rgba(34,255,136,0.35)]' }
    : pct < 70 ? { bar: 'bg-amber-400', text: 'text-amber-300', glow: 'shadow-[0_0_18px_rgba(251,191,36,0.35)]' }
    : { bar: 'bg-red-500', text: 'text-red-400', glow: 'shadow-[0_0_18px_rgba(239,68,68,0.35)]' };

  return (
    <div className={`bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl ${tone.glow}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Flame className={`w-4 h-4 ${tone.text}`} />
          <h3 className="font-sans font-bold text-sm text-white uppercase tracking-wider">System Risk</h3>
        </div>
        <span className="text-[9px] font-mono text-slate-500 uppercase">fusion: {fusionMethod}</span>
      </div>

      <div className="flex items-end justify-between mb-2">
        <span className={`text-5xl font-black ${tone.text}`}>{pct}<span className="text-xl text-slate-500">%</span></span>
        <span className={`text-[10px] font-mono font-black uppercase tracking-widest ${tone.text}`}>{level}</span>
      </div>

      <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
        <div className={`h-full ${tone.bar} transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
};
