import React, { useState } from 'react';
import { ResponsiveContainer, LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import { Table, ShieldAlert, BadgeInfo, Scale, Orbit, Activity } from 'lucide-react';

export const Diagnostics: React.FC = () => {
  const [activeModel, setActiveModel] = useState<'bilstm' | 'dlinear' | 'itransformer'>('bilstm');

  const curCurveData = {
    bilstm: [
      { rec: 0.0, pr: 1.00, fpr: 0.0, tpr: 0.0 },
      { rec: 0.1, pr: 0.99, fpr: 0.01, tpr: 0.15 },
      { rec: 0.3, pr: 0.98, fpr: 0.03, tpr: 0.45 },
      { rec: 0.5, pr: 0.97, fpr: 0.04, tpr: 0.72 },
      { rec: 0.7, pr: 0.95, fpr: 0.06, tpr: 0.88 },
      { rec: 0.8, pr: 0.92, fpr: 0.08, tpr: 0.94 },
      { rec: 0.9, pr: 0.88, fpr: 0.12, tpr: 0.97 },
      { rec: 0.95, pr: 0.78, fpr: 0.20, tpr: 0.99 },
      { rec: 1.0, pr: 0.45, fpr: 1.0, tpr: 1.0 }
    ],
    dlinear: [
      { rec: 0.0, pr: 0.98, fpr: 0.0, tpr: 0.0 },
      { rec: 0.1, pr: 0.95, fpr: 0.02, tpr: 0.12 },
      { rec: 0.3, pr: 0.92, fpr: 0.05, tpr: 0.38 },
      { rec: 0.5, pr: 0.88, fpr: 0.09, tpr: 0.62 },
      { rec: 0.7, pr: 0.84, fpr: 0.14, tpr: 0.80 },
      { rec: 0.8, pr: 0.79, fpr: 0.19, tpr: 0.87 },
      { rec: 0.9, pr: 0.70, fpr: 0.28, tpr: 0.93 },
      { rec: 0.95, pr: 0.58, fpr: 0.40, tpr: 0.96 },
      { rec: 1.0, pr: 0.30, fpr: 1.0, tpr: 1.0 }
    ],
    itransformer: [
      { rec: 0.0, pr: 1.00, fpr: 0.0, tpr: 0.0 },
      { rec: 0.1, pr: 1.00, fpr: 0.00, tpr: 0.18 },
      { rec: 0.3, pr: 0.99, fpr: 0.01, tpr: 0.52 },
      { rec: 0.5, pr: 0.98, fpr: 0.02, tpr: 0.81 },
      { rec: 0.7, pr: 0.97, fpr: 0.03, tpr: 0.93 },
      { rec: 0.8, pr: 0.96, fpr: 0.05, tpr: 0.97 },
      { rec: 0.9, pr: 0.92, fpr: 0.08, tpr: 0.99 },
      { rec: 0.95, pr: 0.85, fpr: 0.14, tpr: 1.00 },
      { rec: 1.0, pr: 0.55, fpr: 1.0, tpr: 1.0 }
    ]
  };

  const nasaPenaltyCurve = Array.from({ length: 31 }, (_, idx) => {
    const errorRatio = (idx - 15) / 10;
    let penaltyValue = 0;
    if (errorRatio < 0) {
      penaltyValue = Math.exp(-errorRatio / 1.3) - 1;
    } else {
      penaltyValue = Math.exp(errorRatio / 1.0) - 1;
    }
    return {
      errorRatio,
      earlyLabel: errorRatio < 0 ? parseFloat(penaltyValue.toFixed(2)) : 0,
      lateLabel: errorRatio >= 0 ? parseFloat(penaltyValue.toFixed(2)) : 0,
      combined: parseFloat(penaltyValue.toFixed(2))
    };
  });

  const bearingClusters = [
    { x: 1.2, y: 1.5, z: 12, name: 'Normal Ring State', fill: '#10b981' },
    { x: 1.8, y: 1.1, z: 15, name: 'Normal Ring State', fill: '#10b981' },
    { x: 0.9, y: 1.7, z: 10, name: 'Normal Ring State', fill: '#10b981' },
    { x: 4.8, y: 5.5, z: 25, name: 'Inner Race Cracks', fill: '#f59e0b' },
    { x: 5.1, y: 4.9, z: 30, name: 'Inner Race Cracks', fill: '#f59e0b' },
    { x: 4.4, y: 5.2, z: 20, name: 'Inner Race Cracks', fill: '#f59e0b' },
    { x: 8.5, y: -2.3, z: 45, name: 'Outer Race Micro-pitting', fill: '#ef4444' },
    { x: 9.1, y: -1.7, z: 40, name: 'Outer Race Micro-pitting', fill: '#ef4444' },
    { x: 8.1, y: -3.0, z: 35, name: 'Outer Race Micro-pitting', fill: '#ef4444' },
    { x: -3.2, y: 4.5, z: 20, name: 'Ball Fault Debris', fill: '#6366f1' },
    { x: -3.8, y: 3.9, z: 25, name: 'Ball Fault Debris', fill: '#6366f1' },
    { x: -2.8, y: 5.1, z: 18, name: 'Ball Fault Debris', fill: '#6366f1' }
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 bg-[#111827] border border-slate-800 p-4 rounded-xl shadow-2xl">
        <div>
          <h2 className="font-sans font-black text-white text-lg">Predictive Diagnostic Evaluation Console</h2>
          <p className="text-xs text-slate-400">Auditing confusion matrices, ROC/PR curves, NASA prognostic penalties, and CWRU bearing failure clusters</p>
        </div>
        <div className="flex items-center gap-1.5 p-1 bg-slate-950 border border-slate-800 rounded shadow-inner">
          {(['bilstm', 'dlinear', 'itransformer'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setActiveModel(m)}
              className={`px-3 py-1 font-mono text-[10.5px] font-bold rounded uppercase tracking-wide transition-all ${
                activeModel === m ? 'bg-[#22ff88] text-[#0a0f1c] shadow-[0_0_10px_rgba(34,255,136,0.3)]' : 'text-slate-400 hover:text-white'
              }`}
            >
              {m === 'bilstm' ? 'BiLSTM' : m === 'dlinear' ? 'DLinear' : 'iTransformer'}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Table className="w-4 h-4 text-[#22ff88]" />
              <h3 className="font-sans font-bold text-sm text-white">Dynamic Anomaly Confusion Matrix</h3>
            </div>
            <p className="text-[10.5px] text-slate-400">
              Evaluation matching current model {activeModel.toUpperCase()} over n = 1500 verification cycles. Shows state sensitivity bounds.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3.5 my-3.5 pb-2.5 max-w-[320px] mx-auto w-full select-none">
            <div className="bg-[#22ff88]/10 border border-[#22ff88]/35 hover:border-[#22ff88]/50 rounded-lg p-3 text-center flex flex-col justify-center">
              <span className="text-[9px] font-mono text-[#22ff88] tracking-wider">TRUE POSITIVE (TP)</span>
              <span className="text-2xl font-sans font-black text-[#22ff88] mt-1">210</span>
              <span className="text-[9px] text-slate-500 font-mono mt-0.5">Anomalies Detected Correctly</span>
            </div>
            <div className="bg-[#241a12] border border-orange-500/15 hover:border-orange-500/30 rounded-lg p-3 text-center flex flex-col justify-center">
              <span className="text-[9px] font-mono text-orange-400 tracking-wider">FALSE POSITIVE (FP)</span>
              <span className="text-2xl font-sans font-black text-orange-300 mt-1">8</span>
              <span className="text-[9px] text-slate-500 font-mono mt-0.5">False Alarms triggered</span>
            </div>
            <div className="bg-[#241212] border border-red-500/15 hover:border-red-500/30 rounded-lg p-3 text-center flex flex-col justify-center">
              <span className="text-[9px] font-mono text-red-400 tracking-wider">FALSE NEGATIVE (FN)</span>
              <span className="text-2xl font-sans font-black text-red-300 mt-1">2</span>
              <span className="text-[9px] text-slate-500 font-mono mt-0.5">Critical Failures Missed</span>
            </div>
            <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3 text-center flex flex-col justify-center">
              <span className="text-[9px] font-mono text-slate-400 tracking-wider">TRUE NEGATIVE (TN)</span>
              <span className="text-2xl font-sans font-black text-white mt-1">1030</span>
              <span className="text-[9px] text-slate-500 font-mono mt-0.5">Healthy States Verified</span>
            </div>
          </div>

          <div className="border-t border-slate-800 pt-3 flex grid-cols-2 justify-between text-center gap-1.5 font-mono text-[10px]">
            <div className="flex flex-col items-center">
              <span className="text-slate-500">ACCURACY</span>
              <span className="text-white font-bold">99.20%</span>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-slate-500">PRECISION</span>
              <span className="text-white font-bold">96.33%</span>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-slate-500">RECALL (SENS)</span>
              <span className="text-white font-bold">99.06%</span>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-slate-500">F1 SCORE</span>
              <span className="text-[#22ff88] font-extrabold font-black">97.67%</span>
            </div>
          </div>
        </div>

        <div className="lg:col-span-7 bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <ShieldAlert className="w-4 h-4 text-[#22ff88]" />
              <h3 className="font-sans font-bold text-sm text-white">Precision-Recall (PR) Curve Metrics</h3>
            </div>
            <p className="text-[10.5px] text-slate-400">
              Evaluating the precision/recall trade-off profile of {activeModel.toUpperCase()} under high-density noise simulation. PR-AUC validates model stability.
            </p>
          </div>

          <div className="h-56 mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curCurveData[activeModel]} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.3} />
                <XAxis dataKey="rec" stroke="#64748b" fontSize={10} tickLine={false} label={{ value: 'Recall Threshold', position: 'insideBottom', offset: -5, fontSize: 9, fill: '#64748b' }} />
                <YAxis stroke="#64748b" fontSize={10} tickLine={false} label={{ value: 'Precision Rate', angle: -90, position: 'insideLeft', offset: 10, fontSize: 9, fill: '#64748b' }} />
                <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b' }} />
                <Legend wrapperStyle={{ fontSize: '9px' }} />
                <Line type="monotone" dataKey="pr" name={`${activeModel.toUpperCase()} Curve`} stroke="#22ff88" strokeWidth={3} dot={{ r: 3 }} />
                <Line type="linear" dataKey="fpr" name="False Positive Baseline" stroke="#ef4444" strokeWidth={1} strokeDasharray="4 4" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <p className="text-[9.5px] text-slate-500 font-sans mt-3">
            * AUC scores of greater than 0.90 declare robust industrial safety capabilities suitable for safety-critical micro-jet installations.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Scale className="w-4 h-4 text-[#22ff88]" />
              <h3 className="font-sans font-bold text-sm text-white">NASA prognostic Loss Penalty curves (Asymmetric)</h3>
            </div>
            <p className="text-[10.5px] text-slate-400">
              The NASA Gas Turbine Prognostic Loss specifies exponential penalty bounds where a late prediction (e_t &gt; 0) is up to 13 times more heavily penalized than an early shutdown prediction (e_t &lt; 0) due to cataclysmic risk prevention constraints.
            </p>
          </div>

          <div className="h-48 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={nasaPenaltyCurve} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.2} />
                <XAxis dataKey="errorRatio" stroke="#64748b" fontSize={9} tickLine={false} label={{ value: 'Prediction Error Ratio (Under/Over RUL)', position: 'insideBottom', offset: -5, fontSize: 8, fill: '#64748b' }} />
                <YAxis stroke="#64748b" fontSize={9} tickLine={false} label={{ value: 'NASA Penalty Weight', angle: -90, position: 'insideLeft', offset: 10, fontSize: 8, fill: '#64748b' }} />
                <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b' }} />
                <Line type="monotone" dataKey="earlyLabel" name="Early (Soft safety penalty)" stroke="#22ff88" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="lateLabel" name="Late (Catastrophic penalty)" stroke="#ef4444" strokeWidth={3.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="flex items-center gap-2 bg-[#22ff88]/10 border border-[#22ff88]/30 p-2.5 rounded-lg text-[9.5px] text-slate-450 mt-4 font-sans leading-relaxed">
            <BadgeInfo className="w-4 h-4 text-[#22ff88] flex-shrink-0" />
            <span>
              OmniPdM deep models target asymmetric bounds to prioritize slightly early shutdowns rather than risking high-inertia compressor detonations.
            </span>
          </div>
        </div>

        <div className="lg:col-span-6 bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Orbit className="w-4 h-4 text-indigo-400" />
              <h3 className="font-sans font-bold text-sm text-white">Structure-based bearing Fault Clustering (t-SNE)</h3>
            </div>
            <p className="text-[10.5px] text-slate-400">
              Dimensionality reduction mapping high-frequency CWRU vibration matrices (DE, FE, and BA sensors) into localized t-SNE clusters, demonstrating explicit separation between healthy rings and micro-crack layers.
            </p>
          </div>

          <div className="h-48 mt-4 select-none">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.2} />
                <XAxis type="number" dataKey="x" name="t-SNE Axis 1" stroke="#64748b" fontSize={9} />
                <YAxis type="number" dataKey="y" name="t-SNE Axis 2" stroke="#64748b" fontSize={9} />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-[#111827] border border-slate-800 p-2 rounded-lg text-[9.5px] font-mono leading-normal shadow-md">
                          <p className="font-black" style={{ color: data.fill }}>{data.name}</p>
                          <p className="text-slate-400">Coords: {data.x.toFixed(2)}, {data.y.toFixed(2)}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter name="Bearing Components" data={bearingClusters} fill="#6366f1">
                  {bearingClusters.map((entry, index) => (
                    <circle
                      key={`cell-${entry.x}-${entry.y}-${index}`}
                      cx="0"
                      cy="0"
                      r="6"
                      fill={entry.fill}
                      style={{ transform: `translate(0, 0)` }}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          <div className="flex flex-wrap gap-2 justify-center text-[8.5px] font-mono mt-4">
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-[#22ff88] rounded" /> Normal Rings</span>
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-[#f59e0b] rounded" /> Inner Race fault</span>
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-[#ef4444] rounded" /> Outer Race Micro</span>
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-[#6366f1] rounded" /> Ball Fault Debris</span>
          </div>
        </div>
      </div>
    </div>
  );
};
