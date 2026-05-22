import React, { useState } from 'react';
import { ReportItem } from '../types';
import { Table, Search, ShieldCheck, Download, Trash2, Sliders, RefreshCw, Layers } from 'lucide-react';

interface ReportsProps {
  history: ReportItem[];
  setHistory: React.Dispatch<React.SetStateAction<ReportItem[]>>;
  setActiveTab: (tab: string) => void;
  setDataset: (val: string) => void;
}

export const Reports: React.FC<ReportsProps> = ({
  history,
  setHistory,
  setActiveTab,
  setDataset
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedLevel, setSelectedLevel] = useState<string>('ALL');
  const [selectedLogs, setSelectedLogs] = useState<string[]>([]);
  const [comparisonSummary, setComparisonSummary] = useState<string | null>(null);

  const handleToggleSelect = (id: string) => {
    setSelectedLogs(prev =>
      prev.includes(id) ? prev.filter(item => item !== id) : [...prev, id]
    );
  };

  const handleDelete = (id: string) => {
    setHistory(prev => prev.filter(item => item.id !== id));
    setSelectedLogs(prev => prev.filter(item => item !== id));
  };

  const handleBulkCompare = () => {
    if (selectedLogs.length < 2) {
      alert('Please select at least 2 reports to perform bulk comparisons.');
      return;
    }

    const items = history.filter(h => selectedLogs.includes(h.id));
    let summaryText = `### Selected Systems Bulk Comparative Diagnostics\n\n`;
    summaryText += `Comparing **${items.length}** distinct predictive configurations under standard MLflow runs.\n\n`;

    items.forEach((item, index) => {
      summaryText += `#### [${index + 1}] System Run: ${item.id} (${item.dataset})\n`;
      summaryText += `- **Model Variant**: ${item.model}\n`;
      summaryText += `- **Engine Risk Level**: ${item.riskScore}% (${item.riskLevel})\n`;
      summaryText += `- **NASA loss Penalty Score**: ${item.nasaScore} RMSE (Asymmetric margin)\n`;
      summaryText += `- **Operator Grounding**: ${item.operator}\n`;
      summaryText += `- **Diagnostic Summary**: ${item.llmSummary}\n\n`;
    });

    setComparisonSummary(summaryText);
  };

  const handleExportZip = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(history, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "omnipdm_reports_archive_export.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const filteredHistory = history.filter(item => {
    const matchSearch =
      item.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.dataset.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.llmSummary.toLowerCase().includes(searchTerm.toLowerCase());
    const matchLevel = selectedLevel === 'ALL' || item.riskLevel === selectedLevel;
    return matchSearch && matchLevel;
  });

  return (
    <div className="space-y-6">
      <div className="bg-[#111827] border border-slate-800 p-5 rounded-xl flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-2xl">
        <div>
          <h2 className="font-sans font-black text-white text-lg flex items-center gap-2">
            <Layers className="w-5 h-5 text-[#22ff88]" />
            MLflow Registry & Diagnostics Records
          </h2>
          <p className="text-xs text-slate-400 mt-1">Audit previous diagnostics run logs, trigger side-by-side comparative matrices, or synchronize exports with edge databases.</p>
        </div>

        <div className="flex gap-2.5">
          <button
            onClick={handleExportZip}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#1e293b] text-slate-200 border border-slate-700 hover:bg-slate-800 rounded font-sans text-xs font-bold transition-all uppercase"
          >
            <Download className="w-4 h-4 text-[#22ff88]" />
            Export Archive
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-8 bg-[#111827] border border-slate-800 p-5 rounded-xl space-y-4 shadow-2xl">
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search log ID, dataset anomalies..."
                className="w-full bg-slate-950 border border-slate-800 text-xs text-slate-200 pl-9 pr-4 h-9 rounded outline-none focus:border-[#22ff88]/50 transition-colors placeholder:text-slate-500"
              />
            </div>

            <select
              value={selectedLevel}
              onChange={(e) => setSelectedLevel(e.target.value)}
              className="bg-slate-950 border border-slate-800 text-xs text-slate-300 px-3 h-9 rounded outline-none focus:border-[#22ff88]/50 w-full sm:w-40 font-semibold"
            >
              <option value="ALL">All Risk states</option>
              <option value="Healthy">Healthy (Safe)</option>
              <option value="Warning">Warning (Anomalous)</option>
              <option value="Critical">Critical (Shutdown)</option>
            </select>

            <button
              onClick={handleBulkCompare}
              disabled={selectedLogs.length < 2}
              className={`px-4 h-9 rounded text-xs font-sans font-black uppercase tracking-wider transition-all border ${
                selectedLogs.length >= 2
                  ? 'bg-[#22ff88] border-[#22ff88] text-[#0a0f1c] shadow-[0_0_10px_rgba(34,255,136,0.3)] hover:bg-[#1ee077]'
                  : 'bg-slate-900 border-slate-800 text-slate-500 cursor-not-allowed'
              }`}
            >
              Bulk Compare ({selectedLogs.length})
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left font-sans text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 font-medium">
                  <th className="py-2.5 px-2 text-center w-8">
                    <span className="sr-only">Select</span>
                  </th>
                  <th className="py-2.5 px-3">Run ID</th>
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Diagnostic Dataset</th>
                  <th className="py-2.5 px-3 font-mono">Engine</th>
                  <th className="py-2.5 px-3 text-center">Score</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredHistory.map((item) => {
                  const selectState = selectedLogs.includes(item.id);
                  const isCritical = item.riskLevel === 'Critical';
                  const isWarning = item.riskLevel === 'Warning';
                  return (
                    <tr
                      key={item.id}
                      className={`border-b border-slate-900/60 hover:bg-slate-800/10 text-slate-300 transition-colors ${
                        selectState ? 'bg-[#22ff88]/5' : ''
                      }`}
                    >
                      <td className="py-3 px-2 text-center">
                        <input
                          type="checkbox"
                          checked={selectState}
                          onChange={() => handleToggleSelect(item.id)}
                          className="rounded border-slate-700 text-[#22ff88] bg-slate-950 h-3.5 w-3.5 focus:ring-0 cursor-pointer"
                        />
                      </td>
                      <td className="py-3 px-3 font-mono font-bold text-[#e2e8f0]">{item.id}</td>
                      <td className="py-3 px-3 text-slate-400">
                        {new Date(item.timestamp).toLocaleString('ko-KR')}
                      </td>
                      <td className="py-3 px-3 text-slate-200">
                        {item.dataset}
                      </td>
                      <td className="py-3 px-3 font-mono text-xs">{item.model}</td>
                      <td className="py-3 px-3 text-center">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            isCritical
                              ? 'text-red-400 bg-red-400/10'
                              : isWarning
                              ? 'text-amber-400 bg-amber-400/10'
                              : 'text-[#22ff88] bg-[#22ff88]/10'
                          }`}
                        >
                          {item.riskScore}%
                        </span>
                      </td>
                      <td className="py-3 px-3 text-[#22ff88] text-right space-x-3.5">
                        <button
                          onClick={() => {
                            setDataset(item.dataset.toLowerCase().replace('-', '_'));
                            setActiveTab('analysis');
                          }}
                          className="hover:text-[#1ee077] hover:underline font-bold text-[10.5px] uppercase font-sans whitespace-nowrap cursor-pointer"
                        >
                          Load
                        </button>
                        <button
                          onClick={() => handleDelete(item.id)}
                          className="text-red-400 hover:text-red-300 tracking-wide font-sans font-bold hover:underline cursor-pointer"
                          title="Delete Diagnostic Document"
                        >
                          <Trash2 className="w-3.5 h-3.5 inline" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
                {filteredHistory.length === 0 && (
                  <tr>
                    <td colSpan={7} className="text-center py-12 text-slate-500 font-sans">
                      No records match the active database filtering values.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="lg:col-span-4 bg-[#111827] border border-slate-800 p-5 rounded-xl space-y-4 shadow-2xl min-h-[300px]">
          <div className="border-b border-slate-800 pb-2">
            <h3 className="font-sans font-bold text-sm text-white">Compare Analyzer Workspace</h3>
            <p className="text-[10.5px] text-slate-400">Run multiple system comparison analysis grids Side-by-Side.</p>
          </div>

          {comparisonSummary ? (
            <div className="bg-slate-950 border border-slate-800 p-4 rounded-lg overflow-y-auto max-h-[350px] font-sans text-[11px] text-slate-300 leading-relaxed prose prose-invert max-w-none space-y-3 prose-headings:text-[#22ff88]">
              <span className="flex items-center gap-1.5 text-[9px] text-[#22ff88] font-mono tracking-widest font-extrabold uppercase mb-2">
                <ShieldCheck className="w-4 h-4 text-[#22ff88]" />
                Comparative report Compiled
              </span>
              <div className="prose prose-invert">
                {comparisonSummary.split('\n\n').map((paragraph, pIdx) => {
                  if (paragraph.startsWith('###') || paragraph.startsWith('####')) {
                    const cleanP = paragraph.replace(/[#]+/g, '').trim();
                    return <h4 key={pIdx} className="text-[#22ff88] font-bold border-b border-slate-800 pb-1 mt-3 mb-1 text-xs uppercase tracking-wide">{cleanP}</h4>;
                  }
                  if (paragraph.startsWith('-')) {
                    return (
                      <ul key={pIdx} className="list-disc pl-4 space-y-1 my-1 text-[10.5px]">
                        {paragraph.split('\n').map((li, lIdx) => (
                          <li key={lIdx} className="text-slate-300">{li.substring(1).trim()}</li>
                        ))}
                      </ul>
                    );
                  }
                  return <p key={pIdx} className="text-slate-300 text-[10.5px] leading-relaxed my-1">{paragraph}</p>;
                })}
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-16 text-center text-slate-500 border border-dashed border-slate-800 rounded-lg">
              <RefreshCw className="w-8 h-8 text-[#22ff88]/20 mb-2 animate-pulse" />
              <p className="text-[10.5px] font-sans px-4">
                Select 2 or more checkboxes in MLflow logs on left, and click "Bulk Compare" to analyze anomalies side-by-side.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
