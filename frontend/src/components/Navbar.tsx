import React from 'react';
import { Activity, BarChart3, Sliders, Layers, ShieldAlert, LogOut } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const TABS = [
  { id: 'home', label: 'Home', icon: Activity },
  { id: 'analysis', label: 'Analysis', icon: BarChart3 },
  { id: 'diagnostics', label: 'Diagnostics', icon: ShieldAlert },
  { id: 'simulator', label: 'What-If', icon: Sliders },
  { id: 'reports', label: 'Reports', icon: Layers },
];

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  return (
    <nav className="sticky top-0 z-30 bg-[#0a0f1c]/95 backdrop-blur border-b border-slate-800">
      <div className="max-w-[1600px] mx-auto px-6 h-14 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-xl text-[#22ff88] font-black tracking-tight" aria-hidden>∞</span>
          <span className="font-sans font-black text-white tracking-tight">
            Omni<span className="text-[#22ff88]">PdM</span>
          </span>
        </div>

        <ul className="flex items-center gap-1">
          {TABS.map(({ id, label, icon: Icon }) => {
            const active = activeTab === id;
            return (
              <li key={id}>
                <button
                  onClick={() => setActiveTab(id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-bold tracking-wide transition-all ${
                    active
                      ? 'bg-[#22ff88]/15 text-[#22ff88] border border-[#22ff88]/30'
                      : 'text-slate-400 hover:text-slate-200 border border-transparent'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {label}
                </button>
              </li>
            );
          })}
        </ul>

        <a
          href="/logout"
          className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-mono text-slate-400 hover:text-red-400 transition-colors"
        >
          <LogOut className="w-3.5 h-3.5" />
          Logout
        </a>
      </div>
    </nav>
  );
};
