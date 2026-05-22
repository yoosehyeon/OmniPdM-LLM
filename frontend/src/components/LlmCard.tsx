import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Sparkles, ShieldCheck, AlertTriangle } from 'lucide-react';

interface LlmCardProps {
  analysisText: string;
  judgeScore: number;
  guardrailViolation: string;
  loading: boolean;
}

export const LlmCard: React.FC<LlmCardProps> = ({
  analysisText,
  judgeScore,
  guardrailViolation,
  loading,
}) => {
  const violated = guardrailViolation && guardrailViolation !== 'None' && guardrailViolation !== 'Awaiting Evaluation';

  return (
    <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl">
      <div className="flex items-center gap-2 mb-3 border-b border-slate-800 pb-2">
        <Sparkles className="w-4 h-4 text-indigo-400" />
        <h3 className="font-sans font-bold text-sm text-white uppercase tracking-wider">AI Analyst</h3>
      </div>

      <div className="flex items-center justify-between mb-3 text-[10px] font-mono">
        <span className="text-slate-400">Judge Score</span>
        <span className={`font-black ${judgeScore >= 85 ? 'text-[#22ff88]' : judgeScore >= 60 ? 'text-amber-300' : 'text-red-400'}`}>
          {judgeScore}
        </span>
      </div>

      <div
        className={`flex items-center gap-1.5 mb-3 text-[10px] font-mono px-2 py-1 rounded ${
          violated
            ? 'bg-red-500/10 text-red-300 border border-red-500/30'
            : 'bg-[#22ff88]/10 text-[#22ff88] border border-[#22ff88]/30'
        }`}
      >
        {violated ? <AlertTriangle className="w-3 h-3" /> : <ShieldCheck className="w-3 h-3" />}
        <span>Guardrail: {guardrailViolation || 'None'}</span>
      </div>

      <div className="text-xs text-slate-300 leading-relaxed prose prose-invert prose-sm max-w-none">
        {loading ? (
          <p className="text-slate-500 italic font-mono">Generating analyst report…</p>
        ) : analysisText ? (
          <ReactMarkdown>{analysisText}</ReactMarkdown>
        ) : (
          <p className="text-slate-500 font-mono text-[11px]">No analyst report yet. Run diagnostics to generate.</p>
        )}
      </div>
    </div>
  );
};
