import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ReferenceLine } from 'recharts';
import { KpiCard } from '../components/KpiCard';
import { Activity, ShieldAlert, Zap, Cpu, History, Flame, TableProperties } from 'lucide-react';
import { ReportItem } from '../types';

interface HomeProps {
  setActiveTab: (tab: string) => void;
  history: ReportItem[];
  selectedDataset: string;
  selectedModel: string;
}

export const Home: React.FC<HomeProps> = ({
  setActiveTab,
  history,
  selectedDataset,
  selectedModel
}) => {
  // Mock performance metrics for BiLSTM, DLinear, and iTransformer
  const modelPerformance = [
    { name: 'BiLSTM + Attention', RMSE: 11.4, NASALoss: 1.82, PRAUC: 0.94 },
    { name: 'DLinear Model', RMSE: 14.8, NASALoss: 2.95, PRAUC: 0.82 },
    { name: 'iTransformer', RMSE: 9.2, NASALoss: 1.15, PRAUC: 0.98 }
  ];

  // Correlation heatmap grid representation (simplifying with styled table grids for maximum styling control and accuracy)
  const sensorsList = ["T30 Temp", "Ps30 Static P", "BPR Ratio", "T24 LPC Temp", "Rotational Speed", "Vibration", "Torque", "Leakage"];
  const correlationsMatrixColors: { [key: string]: string } = {
    "T30 Temp-T30 Temp": "bg-red-500 text-white", "T30 Temp-Ps30 Static P": "bg-orange-500/80 text-orange-200", "T30 Temp-BPR Ratio": "bg-orange-500/60 text-orange-200", "T30 Temp-T24 LPC Temp": "bg-rose-500 text-rose-100", "T30 Temp-Rotational Speed": "bg-blue-500/40 text-blue-200", "T30 Temp-Vibration": "bg-emerald-500/30 text-emerald-300", "T30 Temp-Torque": "bg-slate-800 text-slate-400", "T30 Temp-Leakage": "bg-slate-900 text-slate-500",
    "Ps30 Static P-T30 Temp": "bg-orange-500/80 text-orange-200", "Ps30 Static P-Ps30 Static P": "bg-red-500 text-white", "Ps30 Static P-BPR Ratio": "bg-yellow-500/40 text-yellow-100", "Ps30 Static P-T24 LPC Temp": "bg-orange-500/60 text-orange-200", "Ps30 Static P-Rotational Speed": "bg-blue-500/60 text-blue-200", "Ps30 Static P-Vibration": "bg-slate-800 text-slate-400", "Ps30 Static P-Torque": "bg-slate-900 text-slate-500", "Ps30 Static P-Leakage": "bg-slate-900 text-slate-500",
    "BPR Ratio-T30 Temp": "bg-orange-500/60 text-orange-200", "BPR Ratio-Ps30 Static P": "bg-yellow-500/40 text-yellow-100", "BPR Ratio-BPR Ratio": "bg-red-500 text-white", "BPR Ratio-T24 LPC Temp": "bg-orange-500/50 text-orange-200", "BPR Ratio-Rotational Speed": "bg-slate-800 text-slate-400", "BPR Ratio-Vibration": "bg-slate-900 text-slate-500", "BPR Ratio-Torque": "bg-slate-900 text-slate-500", "BPR Ratio-Leakage": "bg-slate-900 text-slate-500",
    "T24 LPC Temp-T30 Temp": "bg-rose-500 text-rose-100", "T24 LPC Temp-Ps30 Static P": "bg-orange-500/60 text-orange-200", "T24 LPC Temp-BPR Ratio": "bg-orange-500/50 text-orange-200", "T24 LPC Temp-T24 LPC Temp": "bg-red-500 text-white", "T24 LPC Temp-Rotational Speed": "bg-blue-500/50 text-blue-100", "T24 LPC Temp-Vibration": "bg-emerald-500/30 text-emerald-300", "T24 LPC Temp-Torque": "bg-slate-800 text-slate-400", "T24 LPC Temp-Leakage": "bg-slate-900 text-slate-500",
    "Rotational Speed-T30 Temp": "bg-blue-500/40 text-blue-200", "Rotational Speed-Ps30 Static P": "bg-blue-500/60 text-blue-200", "Rotational Speed-BPR Ratio": "bg-slate-800 text-slate-400", "Rotational Speed-T24 LPC Temp": "bg-blue-500/50 text-blue-100", "Rotational Speed-Rotational Speed": "bg-red-500 text-white", "Rotational Speed-Vibration": "bg-orange-500/50 text-orange-200", "Rotational Speed-Torque": "bg-orange-500/80 text-orange-100", "Rotational Speed-Leakage": "bg-emerald-500/40 text-emerald-200",
    "Vibration-T30 Temp": "bg-emerald-500/30 text-emerald-300", "Vibration-Ps30 Static P": "bg-slate-800 text-slate-400", "Vibration-BPR Ratio": "bg-slate-900 text-slate-500", "Vibration-T24 LPC Temp": "bg-emerald-500/30 text-emerald-300", "Vibration-Rotational Speed": "bg-orange-500/50 text-orange-200", "Vibration-Vibration": "bg-red-500 text-white", "Vibration-Torque": "bg-orange-500/60 text-orange-100", "Vibration-Leakage": "bg-slate-800 text-slate-400",
    "Torque-T30 Temp": "bg-slate-800 text-slate-400", "Torque-Ps30 Static P": "bg-slate-900 text-slate-500", "Torque-BPR Ratio": "bg-slate-900 text-slate-500", "Torque-T24 LPC Temp": "bg-slate-800 text-slate-400", "Torque-Rotational Speed": "bg-orange-500/80 text-orange-100", "Torque-Vibration": "bg-orange-500/60 text-orange-100", "Torque-Torque": "bg-red-500 text-white", "Torque-Leakage": "bg-slate-800 text-slate-400",
    "Leakage-T30 Temp": "bg-slate-900 text-slate-500", "Leakage-Ps30 Static P": "bg-slate-900 text-slate-500", "Leakage-BPR Ratio": "bg-slate-900 text-slate-500", "Leakage-T24 LPC Temp": "bg-slate-900 text-slate-500", "Leakage-Rotational Speed": "bg-emerald-500/40 text-emerald-200", "Leakage-Vibration": "bg-slate-800 text-slate-400", "Leakage-Torque": "bg-slate-800 text-slate-400", "Leakage-Leakage": "bg-red-500 text-white"
  };

  const getHeatmapClass = (s1: string, s2: string) => {
    return correlationsMatrixColors[`${s1}-${s2}`] || "bg-slate-900 text-slate-500";
  };

  const getCorrelationValue = (s1: string, s2: string) => {
    if (s1 === s2) return "1.00";
    const key = `${s1}-${s2}`;
    if (key.includes("LPC Temp-T30 Temp") || key.includes("T30 Temp-LPC Temp")) return "0.88";
    if (key.includes("Static P-T30 Temp") || key.includes("T30 Temp-Static P")) return "0.74";
    if (key.includes("Speed-Torque") || key.includes("Torque-Speed")) return "-0.81";
    if (key.includes("Vibration-Speed")) return "0.65";
    if (key.includes("BPR-T30 Temp")) return "0.58";
    return (Math.sin((s1.length + s2.length) * 1.5) * 0.4).toFixed(2);
  };

  return (
    <div className="space-y-6">
      {/* Hero Welcome Unit */}
      <div className="relative bg-gradient-to-r from-slate-900 via-slate-900 to-[#1e293b] border border-slate-800 rounded-2xl p-6 overflow-hidden shadow-2xl">
        <div className="absolute right-0 top-0 w-80 h-80 bg-[#22ff88]/5 rounded-full filter blur-[80px]" />
        <div className="absolute left-1/3 bottom-0 w-60 h-60 bg-indigo-500/5 rounded-full filter blur-[60px]" />

        <div className="max-w-3xl space-y-2">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 bg-[#22ff88]/10 border border-[#22ff88]/30 text-[#22ff88] rounded text-[10px] font-mono tracking-wider font-extrabold uppercase animate-pulse">
            <Activity className="w-3 h-3" />
            Live industrial health status
          </div>
          <h1 className="text-3xl font-sans font-black tracking-tight text-white">
            Predictive Reliability Command Hub <span className="text-[#22ff88]">OmniPdM</span>
          </h1>
          <p className="text-xs text-slate-400 leading-relaxed max-w-2xl">
            Welcome back, <strong className="text-indigo-300">yoosehyeon98</strong>. OmniPdM v6.0 processes online sensor telemetry, applies Bidirectional LSTMs and Inverted Transformers, generates temporal self-attention maps, and yields guardrail-vetted maintenance actions with AI explanation grounding.
          </p>
        </div>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          title="Avg NASA Loss Metric"
          value="1.64 RMSE"
          subtitle="All model seeds mean margin"
          icon={Cpu}
          subValue="^ 0.12%"
          trendType="neutral"
          colorRef="indigo"
        />
        <KpiCard
          title="False Alarm Rate"
          value="0.04 %"
          subtitle="Filtered via dynamic threshold"
          icon={ShieldAlert}
          subValue="v 0.8%"
          trendType="down"
          colorRef="green"
        />
        <KpiCard
          title="Deep Learning Latency"
          value="340 ms"
          subtitle="CPU optimized edge prediction"
          icon={Zap}
          subValue="FAST INF"
          trendType="neutral"
          colorRef="slate"
        />
        <KpiCard
          title="Current System Risk"
          value={`${history[0]?.riskScore ?? 45}%`}
          subtitle={`Index for ${history[0]?.dataset ?? 'Gas Turbines'}`}
          icon={Flame}
          subValue={history[0]?.riskLevel ?? 'Warning'}
          trendType={history[0]?.riskLevel === 'Critical' ? 'up' : 'neutral'}
          colorRef={history[0]?.riskLevel === 'Critical' ? 'red' : history[0]?.riskLevel === 'Warning' ? 'amber' : 'green'}
        />
      </div>

      {/* Content Blocks Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Model Accuracy Chart: 7 Columns */}
        <div className="lg:col-span-7 bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col justify-between">
          <div className="mb-4">
            <h3 className="font-sans font-bold text-sm text-white">Deep Engine Generalization Comparison</h3>
            <p className="text-[11px] text-slate-400">Comparing RMSE error rate (lower is better) and PR-AUC precision (higher is better) across 15 seeds</p>
          </div>

          <div className="h-64 mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelPerformance} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.3} />
                <XAxis dataKey="name" stroke="#64748b" fontSize={10} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={10} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b', borderRadius: '8px' }}
                  labelStyle={{ color: '#ffffff', fontWeight: 'bold', fontSize: '11px' }}
                  itemStyle={{ fontSize: '11px' }}
                />
                <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
                <Bar dataKey="RMSE" name="RMSE Error" fill="#6366f1" radius={[4, 4, 0, 0]} />
                <Bar dataKey="NASALoss" name="NASA Loss Metric" fill="#eab308" radius={[4, 4, 0, 0]} />
                <Bar dataKey="PRAUC" name="PR-AUC Precision" fill="#22ff88" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 flex justify-between items-center text-[10.5px] text-slate-400">
            <span>Model variants synchronized with MLflow logging parameters.</span>
            <button
              onClick={() => setActiveTab('analysis')}
              className="font-mono text-[9px] font-bold text-[#22ff88] uppercase tracking-wider hover:underline"
            >
              Analyze Live &rarr;
            </button>
          </div>
        </div>

        {/* Sensor Heatmap Matrix: 5 Columns */}
        <div className="lg:col-span-5 bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col justify-between">
          <div>
            <h3 className="font-sans font-bold text-sm text-white">Multi-Sensor Covariance Matrix</h3>
            <p className="text-[11px] text-slate-400">Heatmap tracking correlation ratios between physical turbine/actuator parameters (C-MAPSS FD001)</p>
          </div>

          <div className="overflow-x-auto mt-4">
            <table className="w-full text-left font-mono border-collapse" style={{ minWidth: '320px' }}>
              <thead>
                <tr>
                  <th className="p-0.5 text-[8.5px] text-slate-500 border border-slate-800/80"></th>
                  {sensorsList.map((s, idx) => (
                    <th key={idx} className="p-0.5 text-[8.5px] text-slate-400 font-bold border border-slate-800/80 text-center truncate select-none max-w-[45px]" title={s}>
                      {s.split(' ')[0]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sensorsList.map((rowSensor, rIdx) => (
                  <tr key={rIdx}>
                    <td className="p-1 text-[8.5px] text-slate-400 font-bold border border-slate-800/80 whitespace-nowrap truncate max-w-[70px]" title={rowSensor}>
                      {rowSensor}
                    </td>
                    {sensorsList.map((colSensor, cIdx) => {
                      const bgClass = getHeatmapClass(rowSensor, colSensor);
                      const coeff = getCorrelationValue(rowSensor, colSensor);
                      return (
                        <td
                          key={cIdx}
                          className={`p-1 text-[8px] font-black border border-slate-800/80 text-center transition-all duration-300 select-none ${bgClass}`}
                        >
                          {coeff}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="text-[9.5px] text-slate-500 font-sans mt-3.5 leading-normal">
            * Strong positive correlation (red) dictates simultaneous thermal friction buildup. Blue highlights negative shear torque.
          </p>
        </div>
      </div>

      {/* Recent Analyses Log table */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl">
        <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-2">
          <div className="flex items-center gap-2">
            <History className="w-4 h-4 text-[#22ff88]" />
            <h3 className="font-sans font-bold text-sm text-white">Recent Maintenance Runs on Command</h3>
          </div>
          <button
            onClick={() => setActiveTab('reports')}
            className="text-[10px] bg-slate-850 hover:bg-slate-800 text-[#22ff88] font-bold py-1 px-2.5 rounded border border-slate-700 font-sans transition-colors"
          >
            All Reports Log
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left font-sans text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-medium">
                <th className="py-2.5 px-3">Log ID</th>
                <th className="py-2.5 px-3">Execution Time</th>
                <th className="py-2.5 px-3">Dataset ID</th>
                <th className="py-2.5 px-3">Model Variant</th>
                <th className="py-2.5 px-3">Risk Level</th>
                <th className="py-2.5 px-3">Decision Comment Snippet</th>
                <th className="py-2.5 px-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {history.slice(0, 3).map((item, idx) => {
                const color =
                  item.riskScore < 30
                    ? 'text-[#22ff88] bg-[#22ff88]/10'
                    : item.riskScore < 75
                    ? 'text-amber-400 bg-amber-500/10'
                    : 'text-red-400 bg-red-400/10';
                return (
                  <tr key={idx} className="border-b border-slate-900 hover:bg-slate-800/20 text-slate-300 transition-colors">
                    <td className="py-3 px-3 font-mono font-bold text-slate-100">{item.id}</td>
                    <td className="py-3 px-3 text-slate-400">{new Date(item.timestamp).toLocaleString('ko-KR')}</td>
                    <td className="py-3 px-3 font-semibold">{item.dataset}</td>
                    <td className="py-3 px-3 font-mono">{item.model}</td>
                    <td className="py-3 px-3">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${color}`}>
                        {item.riskScore}% ({item.riskLevel})
                      </span>
                    </td>
                    <td className="py-3 px-3 max-w-xs truncate text-slate-400" title={item.llmSummary}>
                      {item.llmSummary}
                    </td>
                    <td className="py-3 px-3 text-right">
                      <button
                        onClick={() => {
                          setActiveTab('analysis');
                        }}
                        className="text-indigo-400 hover:text-indigo-300 font-bold text-[11px] underline"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
