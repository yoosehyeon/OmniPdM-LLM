import React from 'react';
import { LucideIcon } from 'lucide-react';

type ColorRef = 'indigo' | 'green' | 'red' | 'amber' | 'slate';
type TrendType = 'up' | 'down' | 'neutral';

interface KpiCardProps {
  title: string;
  value: string;
  subtitle: string;
  icon: LucideIcon;
  subValue?: string;
  trendType?: TrendType;
  colorRef?: ColorRef;
}

const COLOR_MAP: Record<ColorRef, { ring: string; icon: string; sub: string }> = {
  indigo: { ring: 'border-indigo-500/30', icon: 'text-indigo-400 bg-indigo-500/10', sub: 'text-indigo-300' },
  green: { ring: 'border-[#22ff88]/30', icon: 'text-[#22ff88] bg-[#22ff88]/10', sub: 'text-[#22ff88]' },
  red: { ring: 'border-red-500/30', icon: 'text-red-400 bg-red-500/10', sub: 'text-red-300' },
  amber: { ring: 'border-amber-500/30', icon: 'text-amber-400 bg-amber-500/10', sub: 'text-amber-300' },
  slate: { ring: 'border-slate-700', icon: 'text-slate-300 bg-slate-800/60', sub: 'text-slate-300' },
};

export const KpiCard: React.FC<KpiCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  subValue,
  trendType = 'neutral',
  colorRef = 'slate',
}) => {
  const c = COLOR_MAP[colorRef];
  const trendColor =
    trendType === 'up' ? 'text-red-400' : trendType === 'down' ? 'text-[#22ff88]' : 'text-slate-400';

  return (
    <div className={`bg-[#111827] border ${c.ring} rounded-xl p-4 shadow-2xl`}>
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-[10px] font-mono uppercase tracking-widest text-slate-500">{title}</p>
          <p className="text-2xl font-black text-white leading-tight">{value}</p>
          <p className="text-[10.5px] text-slate-400">{subtitle}</p>
        </div>
        <div className={`p-2 rounded-lg ${c.icon}`}>
          <Icon className="w-4 h-4" />
        </div>
      </div>
      {subValue && (
        <div className="mt-3 pt-2 border-t border-slate-800 flex justify-end">
          <span className={`text-[10px] font-mono font-bold ${trendColor}`}>{subValue}</span>
        </div>
      )}
    </div>
  );
};
