import React, { useState, useEffect, useMemo } from 'react';
import { Navbar } from './components/Navbar';
import { Home } from './pages/Home';
import { Analysis } from './pages/Analysis';
import { Diagnostics } from './pages/Diagnostics';
import { Simulator } from './pages/Simulator';
import { Reports } from './pages/Reports';
import { fetchDatasets, SAMPLE_HISTORY } from './data';
import { postWithCsrf } from './csrf';
import { AnalysisResult, DatasetConfig, ReportItem, RiskLevel } from './types';

// POST 호출은 postWithCsrf 로 X-CSRF-Token 자동 주입 + 403 시 토큰 refresh & 재시도.
// 백엔드/네트워크 실패는 catch 가 mock fallback 으로 흡수 (PoC 시연 안정성).

function deriveLevel(score: number): RiskLevel {
  if (score < 30) return 'Stable';
  if (score < 70) return 'Warning';
  return 'Critical';
}

function buildMockResult(score: number): AnalysisResult {
  const trend = Array.from({ length: 30 }, (_, i) => ({
    time: i,
    predicted: Math.max(0, 120 - score * 0.6 - i * 1.8),
    actual: Math.max(0, 125 - i * 1.9),
    lower: Math.max(0, 100 - score * 0.5 - i * 1.8),
    upper: Math.max(0, 140 - score * 0.5 - i * 1.8),
  }));
  const importance = [
    { name: 'T30 Temp', value: 28 },
    { name: 'Ps30 Pressure', value: 22 },
    { name: 'BPR', value: 17 },
    { name: 'T24 LPC Temp', value: 13 },
    { name: 'Vibration', value: 10 },
    { name: 'Torque', value: 7 },
    { name: 'Leakage', value: 3 },
  ];
  const attention = Array.from({ length: 15 * 15 }, (_, idx) => ({
    row: Math.floor(idx / 15),
    col: idx % 15,
    val: Math.random() * 0.8,
  }));
  const temporal = Array.from({ length: 30 }, (_, i) => ({
    time: i,
    sensors: 20 + Math.sin(i / 3) * 15 + i * 0.5,
    loadFactor: 15 + Math.cos(i / 4) * 10 + i * 0.3,
    interaction: 8 + Math.sin(i / 5) * 5,
  }));
  return {
    riskScore: score,
    riskLevel: deriveLevel(score),
    rulTrend: trend,
    featureImportance: importance,
    attentionMap: attention,
    temporalContribution: temporal,
    llmAnalysis: `**진단 요약**\n\n- 위험도 **${score}%** (${deriveLevel(score)})\n- 상위 기여 센서: T30 Temp, Ps30 Pressure\n- 권고: 200 cycle 내 정밀 점검\n\n*PoC mock 응답입니다. /api/analyze 연결 후 실데이터로 교체됩니다.*`,
    judgeScore: 88,
    guardrailViolation: 'None',
  };
}

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('home');
  const [history, setHistory] = useState<ReportItem[]>(SAMPLE_HISTORY);

  // Dataset 카탈로그 — App 마운트 시 백엔드(/api/datasets) 에서 1회 fetch.
  // 빈 배열로 시작 → 도착 전 페이지는 옵셔널 체이닝으로 비활성 상태 렌더.
  const [datasets, setDatasets] = useState<DatasetConfig[]>([]);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);

  // Analysis 페이지 상태. dataset 은 빈 문자열로 시작, datasets 도착 후 첫 항목으로 동기화.
  const [dataset, setDataset] = useState<string>('');
  const [model, setModel] = useState<string>('bilstm');
  const [sequenceLength, setSequenceLength] = useState<number>(60);
  const [riskFusionMethod, setRiskFusionMethod] = useState<string>('weighted');
  const [llmEnabled, setLlmEnabled] = useState<boolean>(true);
  const [sensorValues, setSensorValues] = useState<{ [key: string]: number }>({});
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const loadDatasets = () => {
    setDatasetsError(null);
    fetchDatasets()
      .then(list => {
        setDatasets(list);
        setDataset(prev => prev || list[0]?.id || '');
      })
      .catch(err => {
        console.error(err);
        setDatasetsError('데이터셋 목록을 불러오지 못했습니다.');
      });
  };

  useEffect(() => {
    loadDatasets();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // dataset 변경 시 sensor 기본값으로 초기화.
  useEffect(() => {
    if (!dataset || datasets.length === 0) return;
    const cfg = datasets.find(d => d.id === dataset);
    if (!cfg) return;
    const initial: { [key: string]: number } = {};
    cfg.sensors.forEach(s => { initial[s.name] = s.defaultValue; });
    setSensorValues(initial);
    setAnalysisResult(null);
  }, [dataset, datasets]);

  const onAnalyze = async () => {
    setLoading(true);
    try {
      const resp = await postWithCsrf('/api/analyze', {
        dataset, model, sequenceLength, riskFusionMethod, llmEnabled, sensorValues,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data: AnalysisResult = await resp.json();
      setAnalysisResult(data);
      pushHistory(data);
    } catch {
      // 백엔드 미구현/에러/네트워크 실패 시 PoC mock fallback — 시연 안정성.
      const mockScore = Math.round(40 + Math.random() * 40);
      const mock = buildMockResult(mockScore);
      setAnalysisResult(mock);
      pushHistory(mock);
    } finally {
      setLoading(false);
    }
  };

  const pushHistory = (r: AnalysisResult) => {
    const cfg = datasets.find(d => d.id === dataset);
    const label = cfg?.label ?? dataset ?? 'Unknown';
    setHistory(prev => [{
      id: `RUN-${Date.now().toString(36).toUpperCase()}`,
      timestamp: new Date().toISOString(),
      dataset: label,
      model,
      riskScore: r.riskScore,
      riskLevel: r.riskLevel,
      llmSummary: r.llmAnalysis.split('\n').find(l => l.trim().length > 0) || '',
      nasaScore: Math.round((r.riskScore / 50) * 100) / 100,
      operator: 'yoosehyeon98',
    }, ...prev]);
  };

  const page = useMemo(() => {
    switch (activeTab) {
      case 'analysis':
        return (
          <Analysis
            datasets={datasets}
            dataset={dataset} setDataset={setDataset}
            model={model} setModel={setModel}
            sequenceLength={sequenceLength} setSequenceLength={setSequenceLength}
            riskFusionMethod={riskFusionMethod} setRiskFusionMethod={setRiskFusionMethod}
            llmEnabled={llmEnabled} setLlmEnabled={setLlmEnabled}
            sensorValues={sensorValues} setSensorValues={setSensorValues}
            analysisResult={analysisResult} loading={loading} onAnalyze={onAnalyze}
          />
        );
      case 'diagnostics':
        return <Diagnostics />;
      case 'simulator':
        return <Simulator datasets={datasets} />;
      case 'reports':
        return <Reports history={history} setHistory={setHistory} setActiveTab={setActiveTab} setDataset={setDataset} />;
      case 'home':
      default:
        return <Home setActiveTab={setActiveTab} history={history} selectedDataset={dataset} selectedModel={model} />;
    }
  }, [activeTab, datasets, dataset, model, sequenceLength, riskFusionMethod, llmEnabled, sensorValues, analysisResult, loading, history]);

  return (
    <div className="min-h-screen">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="max-w-[1600px] mx-auto px-6 py-6">
        {datasetsError && (
          <div className="mb-4 flex items-center gap-3 bg-red-500/10 border border-red-500/30 text-red-300 px-4 py-2 rounded">
            <span className="text-sm">{datasetsError}</span>
            <button
              onClick={loadDatasets}
              className="ml-auto text-xs font-mono uppercase tracking-wider text-red-200 hover:text-white underline"
            >
              재시도
            </button>
          </div>
        )}
        {page}
      </main>
    </div>
  );
};
