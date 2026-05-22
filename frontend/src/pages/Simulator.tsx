import React, { useState, useEffect, useMemo } from 'react';
import { DatasetConfig } from '../types';
import { postWithCsrf } from '../csrf';
import { Sparkles, Sliders, Play, TrendingUp, Cpu, Flame, Check, HelpCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface SimulatorProps {
  datasets: DatasetConfig[];
}

export const Simulator: React.FC<SimulatorProps> = ({ datasets }) => {
  const [selectedDataset, setSelectedDataset] = useState('ai4i');
  const [simulationSensors, setSimulationSensors] = useState<{ [key: string]: number }>({});
  const [fusionMethod, setFusionMethod] = useState<'weighted' | 'noisy_or' | 'max'>('weighted');
  const [simulatedRisk, setSimulatedRisk] = useState(24);
  const [aiRecommendation, setAiRecommendation] = useState('');
  const [aiLoading, setAiLoading] = useState(false);

  // derived state — useMemo 로 안정화 (exhaustive-deps lint 해결).
  const activeDataset = useMemo<DatasetConfig | undefined>(
    () => datasets.find(d => d.id === selectedDataset) ?? datasets[0],
    [datasets, selectedDataset],
  );

  // catalog 도착 후 selectedDataset 이 카탈로그에 없으면 첫 id 로 동기화.
  useEffect(() => {
    if (datasets.length === 0) return;
    if (!datasets.find(d => d.id === selectedDataset)) {
      setSelectedDataset(datasets[0].id);
    }
  }, [datasets, selectedDataset]);

  useEffect(() => {
    if (!activeDataset) return;
    const initialValues: { [key: string]: number } = {};
    activeDataset.sensors.forEach(s => {
      initialValues[s.name] = s.defaultValue;
    });
    setSimulationSensors(initialValues);
    setAiRecommendation('');
  }, [activeDataset]);

  useEffect(() => {
    if (Object.keys(simulationSensors).length === 0) return;

    let computedRisk = 15;

    if (selectedDataset === 'ai4i') {
      const torque = simulationSensors["Torque [Nm]"] || 40;
      const speed = simulationSensors["Rotational Speed [rpm]"] || 1510;
      const wear = simulationSensors["Tool Wear [min]"] || 50;
      const temp = simulationSensors["Process Temp [K]"] || 308;

      const torqueRisk = Math.max(0, (torque - 40) * 1.8);
      const speedRisk = Math.max(0, (speed - 1500) * 0.05);
      const wearRisk = Math.max(0, (wear - 100) * 0.9);
      const tempRisk = Math.max(0, (temp - 300) * 3);

      if (fusionMethod === 'max') {
        computedRisk = Math.max(torqueRisk, speedRisk, wearRisk, tempRisk);
      } else if (fusionMethod === 'noisy_or') {
        const p1 = Math.min(0.85, torqueRisk / 100);
        const p2 = Math.min(0.85, speedRisk / 100);
        const p3 = Math.min(0.85, wearRisk / 100);
        const p4 = Math.min(0.85, tempRisk / 100);
        const total = 1 - (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4);
        computedRisk = Math.round(total * 100);
      } else {
        computedRisk = Math.round(torqueRisk * 0.4 + speedRisk * 0.15 + wearRisk * 0.25 + tempRisk * 0.2);
      }
    } else if (selectedDataset.includes('cmapss')) {
      const t30 = simulationSensors["T30 (HPC Temp)"] || 512;
      const ps30 = simulationSensors["Ps30 (Static Pressure)"] || 47;
      const bpr = simulationSensors["BPR (Bypass Ratio)"] || 8.35;
      const t24 = simulationSensors["T24 (LPC Temp)"] || 641;

      const t30Risk = Math.max(0, (t30 - 510) * 3.5);
      const ps30Risk = Math.max(0, (48.5 - ps30) * 12);
      const bprRisk = Math.max(0, (bpr - 8.3) * 45);
      const t24Risk = Math.max(0, (t24 - 640) * 2.5);

      if (fusionMethod === 'max') {
        computedRisk = Math.max(t30Risk, ps30Risk, bprRisk, t24Risk);
      } else if (fusionMethod === 'noisy_or') {
        const p1 = Math.min(0.85, t30Risk / 100);
        const p2 = Math.min(0.85, ps30Risk / 100);
        const p3 = Math.min(0.85, bprRisk / 100);
        const p4 = Math.min(0.85, t24Risk / 100);
        const total = 1 - (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4);
        computedRisk = Math.round(total * 100);
      } else {
        computedRisk = Math.round(t30Risk * 0.35 + ps30Risk * 0.25 + bprRisk * 0.25 + t24Risk * 0.15);
      }
    } else if (activeDataset) {
      const sumCoef = Object.values(simulationSensors).reduce((acc, curr) => acc + curr, 0);
      computedRisk = Math.min(98, Math.max(8, Math.round((sumCoef / (activeDataset.sensors.length * 150)) * 60)));
    }

    setSimulatedRisk(Math.min(100, Math.max(1, computedRisk)));
  }, [simulationSensors, fusionMethod, selectedDataset, activeDataset]);

  const handleSliderChange = (name: string, val: number) => {
    setSimulationSensors(prev => ({
      ...prev,
      [name]: val
    }));
  };

  const handleGetAiRecommendation = async () => {
    setAiLoading(true);
    setAiRecommendation('');

    try {
      // postWithCsrf 가 X-CSRF-Token 자동 주입 + 403 시 토큰 refresh & 재시도.
      // 백엔드 /api/analyze 는 sensorValues 키를 검증 (services/api/analyze.py).
      const response = await postWithCsrf('/api/analyze', {
        dataset: selectedDataset,
        model: 'bilstm',
        riskFusionMethod: fusionMethod,
        llmEnabled: true,
        sensorValues: simulationSensors,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      // 백엔드 응답 dict 는 report_markdown 에 LLM 코멘트 포함 (services/analyze_service.run).
      const md =
        data?.report_markdown ||
        data?.summary_text ||
        '### No recommendation\n\nBackend returned an empty report.';
      setAiRecommendation(md);
    } catch (err) {
      console.error(err);
      setAiRecommendation('### Network error\n\nFailed to reach the OmniPdM backend.');
    } finally {
      setAiLoading(false);
    }
  };

  const levelColor =
    simulatedRisk < 30
      ? 'border-[#22ff88]/25 text-[#22ff88]'
      : simulatedRisk < 75
      ? 'border-amber-500/25 text-amber-400'
      : 'border-red-500/25 text-red-450';

  return (
    <div className="space-y-6">
      <div className="bg-[#111827] border border-slate-800 p-5 rounded-xl shadow-2xl">
        <h2 className="font-sans font-black text-white text-lg flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-[#22ff88]" />
          What-If Stress Simulation Chamber
        </h2>
        <p className="text-xs text-slate-400 mt-1 max-w-4xl leading-relaxed">
          Manipulate turbine physical features or smart rotary tools underneath highly stressed settings. Observe direct failure index propagation, analyze mathematical variance bounds across weighted sums or cascade cascade-probability fusion methods, and consult AI Co-Pilot recommendations on how to counter catastrophic risk.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-5 bg-[#111827] border border-slate-800 p-5 rounded-xl space-y-5 shadow-2xl">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-2.5">
            <Sliders className="w-4 h-4 text-[#22ff88]" />
            <h3 className="font-sans font-bold text-sm text-white">Simulation Variables</h3>
          </div>

          <div className="space-y-4">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-slate-400 font-bold font-sans">Active Target Machine Domain</label>
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="w-full px-3 py-1.5 bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded outline-none h-9 font-semibold"
              >
                {/* 옵션은 backend catalog (/api/datasets) 가 source of truth. */}
                {datasets.map(ds => (
                  <option key={ds.id} value={ds.id}>{ds.label}</option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-xs text-slate-400 font-bold font-sans flex items-center gap-1">
                Risk Fusion Algorithm
                <span className="cursor-help text-slate-500" title="weighted = linear averages, max = conservative peak failures, noisy_or = independent causal failure cascades">
                  <HelpCircle className="w-3.5 h-3.5" />
                </span>
              </label>
              <div className="grid grid-cols-3 gap-1.5 p-0.5 bg-slate-950 border border-slate-800 rounded">
                {(['weighted', 'noisy_or', 'max'] as const).map(op => (
                  <button
                    key={op}
                    onClick={() => setFusionMethod(op)}
                    className={`py-1 text-[10px] font-mono tracking-tight rounded uppercase transition-colors font-bold ${
                      fusionMethod === op
                        ? 'bg-[#22ff88] text-[#0a0f1c] font-black shadow-[0_0_10px_rgba(34,255,136,0.3)]'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {op === 'weighted' ? 'Weighted' : op === 'noisy_or' ? 'Noisy_OR' : 'Bottleneck_Max'}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-4 pt-3 border-t border-slate-850">
              <span className="text-[10px] font-mono font-bold text-slate-500 uppercase tracking-widest block">
                Adjust Variables
              </span>

              {(activeDataset?.sensors ?? []).map(sensor => {
                const curVal = simulationSensors[sensor.name] ?? sensor.defaultValue;
                return (
                  <div key={sensor.name} className="space-y-1">
                    <div className="flex justify-between font-mono text-[10px]">
                      <span className="text-slate-300 font-bold truncate max-w-[170px] select-none block" title={sensor.name}>
                        {sensor.name}
                      </span>
                      <span className="text-[#22ff88] font-black">
                        {curVal.toFixed(1)} <span className="text-slate-500 text-[9px] font-normal">{sensor.unit}</span>
                      </span>
                    </div>
                    <input
                      type="range"
                      min={sensor.min}
                      max={sensor.max}
                      step={(sensor.max - sensor.min) / 100}
                      value={curVal}
                      onChange={(e) => handleSliderChange(sensor.name, parseFloat(e.target.value))}
                      className="w-full accent-[#22ff88] bg-slate-950 h-1.5 rounded cursor-pointer"
                    />
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="lg:col-span-7 space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="bg-[#111827] border border-slate-800 p-4.5 rounded-xl text-center space-y-2 flex flex-col justify-center items-center shadow-2xl">
              <span className="text-[9px] font-mono tracking-widest text-[#22ff88] font-bold bg-[#22ff88]/15 px-2 py-0.5 rounded border border-[#22ff88]/25">
                BASELINE SCENARIO
              </span>
              <div className="text-3xl font-sans font-black text-slate-300">18%</div>
              <p className="text-[10.5px] text-slate-400">
                Standard machine configuration operating inside designed ambient parameters with no micro-friction.
              </p>
            </div>

            <div className={`bg-[#111827] border p-4.5 rounded-xl text-center space-y-2 flex flex-col justify-center items-center ${levelColor} shadow-2xl`}>
              <span className="text-[9px] font-mono tracking-widest font-black bg-slate-800/80 px-2.5 py-0.5 rounded border border-slate-700 uppercase">
                SIMULATED RISK (STRESSED)
              </span>
              <div className="text-3xl font-sans font-black text-white">{simulatedRisk}%</div>
              <p className="text-[10.5px] text-slate-300">
                {simulatedRisk < 30
                  ? 'Healthy condition maintained under current simulated profile.'
                  : simulatedRisk < 75
                  ? 'Anomalous fatigue accumulation warned. High-wear margin activated.'
                  : 'CRITICAL SHUTDOWN WARNING. Physical parameters exceeding maximum shear limit.'}
              </p>
            </div>
          </div>

          <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800 pb-3 gap-3">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4.5 h-4.5 text-[#22ff88] animate-pulse" />
                <div>
                  <h4 className="font-sans font-bold text-xs text-white">AI What-If Remediator Guidance</h4>
                  <p className="text-[9.5px] font-mono text-[#22ff88]">REAL-TIME SCENARIO EVALUATION</p>
                </div>
              </div>

              <button
                onClick={handleGetAiRecommendation}
                disabled={aiLoading}
                className="flex items-center gap-1.5 px-3.5 py-1.5 bg-[#22ff88] text-[#0a0f1c] font-sans font-black text-[11px] tracking-wider uppercase rounded hover:bg-[#1ee077] transition-all shadow-[0_0_12px_rgba(34,255,136,0.25)] disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed cursor-pointer"
              >
                <Play className="w-3.5 h-3.5" />
                {aiLoading ? 'Reasoning...' : 'Get Scenario Guidance'}
              </button>
            </div>

            {aiLoading ? (
              <div className="py-12 text-center space-y-2">
                <div className="w-10 h-10 border-4 border-[#22ff88]/20 border-t-[#22ff88] rounded-full animate-spin mx-auto" />
                <span className="text-xs font-mono text-slate-400 block animate-pulse">Requesting emergency actions...</span>
              </div>
            ) : aiRecommendation ? (
              <div className="prose prose-invert max-w-none font-sans text-[11px] leading-relaxed text-slate-300 px-4 py-3 bg-slate-950 border border-slate-800 rounded-lg space-y-3 prose-headings:text-[#22ff88]">
                <ReactMarkdown>{aiRecommendation}</ReactMarkdown>
              </div>
            ) : (
              <div className="text-center py-10 text-slate-500 border border-dashed border-slate-800 bg-slate-950 rounded-lg">
                <Cpu className="w-8 h-8 text-[#22ff88]/20 mx-auto mb-2 animate-pulse" />
                <span className="text-[11px] font-sans">
                  Click the button above to request maintenance guidance based on your custom slider configurations.
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
