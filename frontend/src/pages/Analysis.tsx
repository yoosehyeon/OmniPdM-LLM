import React, { useState } from 'react';
import { ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Area } from 'recharts';
import { RiskGauge } from '../components/RiskGauge';
import { LlmCard } from '../components/LlmCard';
import { AnalysisResult, DatasetConfig } from '../types';
import { Sliders, HelpCircle, Flame, Sparkles, Network } from 'lucide-react';

interface AnalysisProps {
  datasets: DatasetConfig[];
  dataset: string;
  setDataset: (val: string) => void;
  model: string;
  setModel: (val: string) => void;
  sequenceLength: number;
  setSequenceLength: (val: number) => void;
  riskFusionMethod: string;
  setRiskFusionMethod: (val: string) => void;
  llmEnabled: boolean;
  setLlmEnabled: (val: boolean) => void;
  sensorValues: { [key: string]: number };
  setSensorValues: React.Dispatch<React.SetStateAction<{ [key: string]: number }>>;
  analysisResult: AnalysisResult | null;
  loading: boolean;
  onAnalyze: () => void;
}

export const Analysis: React.FC<AnalysisProps> = ({
  datasets,
  dataset,
  setDataset,
  model,
  setModel,
  sequenceLength,
  setSequenceLength,
  riskFusionMethod,
  setRiskFusionMethod,
  llmEnabled,
  setLlmEnabled,
  sensorValues,
  setSensorValues,
  analysisResult,
  loading,
  onAnalyze
}) => {
  const [activeXaiTab, setActiveXaiTab] = useState<'importance' | 'attention' | 'temporal'>('importance');

  // datasets 가 비어 있으면 (fetch 전) sensors 도 빈 배열 — 슬라이더 미렌더.
  const currentDatasetConfig: DatasetConfig | undefined =
    datasets.find(d => d.id === dataset) ?? datasets[0];

  const handleSliderChange = (sensorName: string, value: number) => {
    setSensorValues(prev => ({
      ...prev,
      [sensorName]: value
    }));
  };

  const chartData = analysisResult?.rulTrend || [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      <div className="lg:col-span-3 space-y-6">
        <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl">
          <div className="flex items-center gap-2 mb-4 border-b border-slate-800 pb-2">
            <Sliders className="w-4 h-4 text-[#22ff88]" />
            <h3 className="font-sans font-bold text-sm text-white uppercase tracking-wider">Parameters & Sensors</h3>
          </div>

          <div className="space-y-4 text-xs">
            <div className="flex flex-col gap-1.5">
              <label className="text-slate-400 font-medium">Input Sequence Length</label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="10"
                  max="120"
                  value={sequenceLength}
                  onChange={(e) => setSequenceLength(parseInt(e.target.value))}
                  className="w-full accent-[#22ff88] bg-[#1e293b] h-1.5 rounded cursor-pointer"
                />
                <span className="font-mono text-white bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700 min-w-[35px] text-center font-bold">
                  {sequenceLength}
                </span>
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-slate-400 font-medium flex items-center gap-1">
                Risk Fusion Method
                <span className="cursor-help text-slate-500" title="weighted = linear combination, max = bottleneck, noisy_or = probabilistic cascade failure model">
                  <HelpCircle className="w-3 h-3" />
                </span>
              </label>
              <select
                value={riskFusionMethod}
                onChange={(e) => setRiskFusionMethod(e.target.value)}
                className="w-full px-3 py-1.5 bg-[#1e293b] border border-slate-700 focus:border-[#22ff88]/50 text-slate-200 rounded outline-none font-medium h-9"
              >
                <option value="weighted">Weighted sum (Linear)</option>
                <option value="noisy_or">Noisy-OR model</option>
                <option value="max">Maximum bottleneck (Worst-case)</option>
              </select>
            </div>

            <div className="flex items-center justify-between py-2 border-t border-b border-slate-850">
              <div className="flex flex-col">
                <span className="text-slate-200 font-semibold flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-400 animate-pulse" />
                  Korean AI Analyst
                </span>
                <span className="text-[9px] text-slate-500 font-mono text-left">Generate detailed expert report</span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={llmEnabled}
                  onChange={(e) => setLlmEnabled(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-slate-800 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-slate-300 after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-500 peer-checked:after:bg-white" />
              </label>
            </div>

            <div className="space-y-4 pt-2">
              <span className="text-[10px] font-mono text-slate-500 font-black tracking-widest uppercase block border-b border-slate-800 pb-1">
                Real-Time Sensor Overrides
              </span>

              {(currentDatasetConfig?.sensors ?? []).map((sensor) => {
                const currentVal = sensorValues[sensor.name] ?? sensor.defaultValue;
                return (
                  <div key={sensor.name} className="space-y-1">
                    <div className="flex justify-between font-mono text-[10px]">
                      <span className="text-slate-300 font-bold truncate max-w-[130px] block" title={sensor.name}>
                        {sensor.name}
                      </span>
                      <span className="text-indigo-400 font-black">
                        {currentVal.toFixed(1)} <span className="text-slate-500 text-[9px] font-normal">{sensor.unit}</span>
                      </span>
                    </div>
                    <input
                      type="range"
                      min={sensor.min}
                      max={sensor.max}
                      step={(sensor.max - sensor.min) / 100}
                      value={currentVal}
                      onChange={(e) => handleSliderChange(sensor.name, parseFloat(e.target.value))}
                      className="w-full accent-[#22ff88] bg-[#1e293b] h-1 rounded cursor-pointer"
                    />
                    <div className="flex justify-between text-[9px] text-slate-600 font-mono">
                      <span>{sensor.min}</span>
                      <span>{sensor.max}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <button
          onClick={onAnalyze}
          disabled={loading}
          className={`w-full py-2.5 font-sans font-black text-xs uppercase tracking-widest rounded-lg flex items-center justify-center gap-1.5 transition-all ${
            loading
              ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
              : 'bg-[#22ff88] text-slate-950 hover:bg-[#1ee077] hover:shadow-[0_0_12px_rgba(34,255,136,0.25)]'
          }`}
        >
          {loading ? 'Crunching engine numbers...' : 'Evaluate Diagnostics'}
        </button>
      </div>

      <div className="lg:col-span-6 space-y-6">
        {analysisResult ? (
          <>
            <RiskGauge
              score={analysisResult.riskScore}
              level={analysisResult.riskLevel}
              fusionMethod={riskFusionMethod}
            />

            <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl">
              <div className="mb-4">
                <h4 className="font-sans font-bold text-sm text-white">Remaining Useful Life (RUL) Forecasting Trajectory</h4>
                <p className="text-[11px] text-slate-400">Time-series forecasting mapping current degradation profile to structural failure limit (Cycles)</p>
              </div>

              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.3} />
                    <XAxis dataKey="time" stroke="#64748b" fontSize={10} tickLine={false} label={{ value: 'Prediction Cycles', position: 'insideBottom', offset: -5, fontSize: 9, fill: '#64748b' }} />
                    <YAxis stroke="#64748b" fontSize={10} tickLine={false} label={{ value: 'RUL Target', angle: -90, position: 'insideLeft', offset: 10, fontSize: 9, fill: '#64748b' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b', borderRadius: '8px' }}
                      labelStyle={{ color: '#ffffff', fontWeight: 'bold', fontSize: '11px' }}
                      itemStyle={{ fontSize: '11px' }}
                    />
                    <Line type="monotone" dataKey="predicted" name="Predicted RUL" stroke="#c084fc" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                    <Line type="monotone" dataKey="actual" name="Actual RUL" stroke="#22ff88" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                    <Line type="monotone" dataKey="lower" name="Confidence Bounds (-)" stroke="#fda4af" strokeWidth={1} dot={false} className="opacity-40" />
                    <Line type="monotone" dataKey="upper" name="Confidence Bounds (+)" stroke="#fda4af" strokeWidth={1} dot={false} className="opacity-40" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl">
              <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-1">
                <h4 className="font-sans font-bold text-sm text-white">XAI Explanatory Transparency</h4>

                <div className="flex items-center gap-1.5 p-0.5 bg-slate-950 border border-slate-800 rounded">
                  {(['importance', 'attention', 'temporal'] as const).map((t) => (
                    <button
                      key={t}
                      onClick={() => setActiveXaiTab(t)}
                      className={`px-2 py-1 text-[10px] font-mono rounded select-none font-bold tracking-tight uppercase ${
                        activeXaiTab === t
                          ? 'bg-[#22ff88] text-[#0a0f1c] font-black shadow-[0_0_10px_rgba(34,255,136,0.3)]'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {t === 'importance' ? 'Sensors' : t === 'attention' ? 'Attention' : 'Summary'}
                    </button>
                  ))}
                </div>
              </div>

              {activeXaiTab === 'importance' && (
                <div className="space-y-4">
                  <p className="text-[10.5px] text-slate-400 leading-relaxed">
                    Percentage influence contribution of top physical parameters towards current failure index score, calculated via temporal Integrated Gradients on DLinear/BiLSTM channels.
                  </p>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={analysisResult.featureImportance.slice(0, 7)} layout="vertical" margin={{ top: 0, right: 10, left: 35, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.2} horizontal={false} />
                        <XAxis type="number" stroke="#64748b" fontSize={9} tickLine={false} />
                        <YAxis type="category" dataKey="name" stroke="#cbd5e1" fontSize={9} width={90} tickLine={false} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b', borderRadius: '8px' }}
                          itemStyle={{ fontSize: '11px', color: '#c084fc' }}
                        />
                        <Bar dataKey="value" name="Risk Weight %" fill="#c084fc" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {activeXaiTab === 'attention' && (
                <div className="space-y-3">
                  <p className="text-[11px] text-slate-400">
                    Self-Attention alignment matrix (A = Softmax(QK^T / sqrt(d))) capturing sequential relevance over the last 15 historical sensor intervals. Hotter grids indicate temporal failure anchors.
                  </p>
                  <div className="grid grid-cols-15 gap-0.5 border border-slate-800 p-2 bg-slate-950 rounded-lg w-fit mx-auto select-none">
                    {analysisResult.attentionMap.map((cell, idx) => {
                      const hotness = cell.val;
                      let bg = 'bg-slate-900/60';
                      if (hotness > 0.6) bg = 'bg-red-500/90 text-white';
                      else if (hotness > 0.45) bg = 'bg-orange-500/80 text-orange-100';
                      else if (hotness > 0.3) bg = 'bg-[#22ff88]/50 text-slate-200';
                      else if (hotness > 0.15) bg = 'bg-[#22ff88]/15 text-slate-400';
                      return (
                        <div
                          key={idx}
                          className={`w-[18px] h-[18px] text-[7px] font-mono flex items-center justify-center rounded-sm transition-all duration-300 hover:scale-125 hover:z-20 ${bg}`}
                          title={`Time Frame ${cell.row} x ${cell.col}: ${cell.val}`}
                        >
                          {cell.row === cell.col ? '.' : ''}
                        </div>
                      );
                    })}
                  </div>
                  <div className="flex gap-4 justify-center text-[9px] font-mono text-slate-500 pt-1">
                    <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-slate-900/60 rounded" /> Base Attn</span>
                    <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-[#22ff88]/30 rounded" /> Minor Relevance</span>
                    <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-orange-500/70 rounded animate-pulse" /> Moderate Correlation</span>
                    <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-red-400 rounded animate-ping scale-75" /> Peak Overloads</span>
                  </div>
                </div>
              )}

              {activeXaiTab === 'temporal' && (
                <div className="space-y-4">
                  <p className="text-[10.5px] text-slate-400 leading-relaxed">
                    Integrated risk progression curve showing how chemical sensory shifts, mechanical load stress, and system couplings stack over the sequence.
                  </p>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={analysisResult.temporalContribution} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.2} />
                        <XAxis dataKey="time" stroke="#64748b" fontSize={9} tickLine={false} />
                        <YAxis stroke="#64748b" fontSize={9} tickLine={false} />
                        <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b' }} />
                        <Line type="linear" dataKey="sensors" name="Sensor Fluctuations" stroke="#ef4444" strokeWidth={2} dot={false} />
                        <Line type="linear" dataKey="loadFactor" name="Frictional Load Factor" stroke="#eab308" strokeWidth={2} dot={false} />
                        <Line type="linear" dataKey="interaction" name="Component Coupling" stroke="#3b82f6" strokeWidth={1} strokeDasharray="3 3" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="bg-[#111827] border border-dashed border-slate-800 rounded-xl py-16 px-10 text-center flex flex-col items-center justify-center space-y-4 shadow-inner">
            <div className="p-4 bg-slate-800/60 border border-slate-700 rounded-full animate-bounce">
              <Network className="w-8 h-8 text-indigo-400" />
            </div>
            <div className="space-y-1">
              <h4 className="font-sans font-bold text-sm text-white">Diagnostics Awaiting Run Trigger</h4>
              <p className="text-xs text-slate-400 max-w-sm leading-relaxed">
                Click "Run Diagnostics" inside parameters sidebar or top right command panel to initiate predictive failure assessments.
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="lg:col-span-3 space-y-6">
        <LlmCard
          analysisText={analysisResult?.llmAnalysis || ''}
          judgeScore={analysisResult?.judgeScore ?? 90}
          guardrailViolation={analysisResult?.guardrailViolation || 'Awaiting Evaluation'}
          loading={loading}
        />
      </div>
    </div>
  );
};
