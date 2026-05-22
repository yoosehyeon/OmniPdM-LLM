import { DatasetConfig, CoreModel, ReportItem } from './types';

// 백엔드 SSOT 와 동기화:
// - DatasetConfig: GET /api/datasets 로 런타임 fetch (services/dataset_catalog.py 가 source of truth).
// - CoreModel / ReportItem: 현재 frontend mock — 후속 API 가 붙으면 동일 패턴으로 교체.

/**
 * Dataset 메타 카탈로그를 백엔드에서 가져온다.
 *
 * 호출 시점: App 마운트 1회. 결과는 props 로 페이지에 전달.
 * 실패 시: throw — App.tsx 에서 catch 해 경고 배너 표시.
 *
 * 인증: /api/datasets 는 가드 면제 (app.py _AUTH_EXEMPT_PREFIXES) 라 무인증 호출 가능.
 * credentials: 'include' 는 후속 endpoint 와 일관성 + 향후 가드 추가 대비.
 */
export async function fetchDatasets(): Promise<DatasetConfig[]> {
  const resp = await fetch('/api/datasets', {
    credentials: 'include',
    headers: { Accept: 'application/json' },
  });
  if (!resp.ok) {
    throw new Error(`GET /api/datasets failed: ${resp.status}`);
  }
  return resp.json() as Promise<DatasetConfig[]>;
}

export const CORE_MODELS: CoreModel[] = [
  { id: 'bilstm', label: 'BiLSTM + Attention', description: 'Bidirectional LSTM with temporal attention' },
  { id: 'dlinear', label: 'DLinear', description: 'Decomposed linear forecast baseline' },
  { id: 'itransformer', label: 'iTransformer', description: 'Inverted Transformer for time-series' },
];

export const SAMPLE_HISTORY: ReportItem[] = [
  {
    id: 'RUN-2026-0518-A1',
    timestamp: '2026-05-18T09:14:00Z',
    dataset: 'C-MAPSS FD001',
    model: 'BiLSTM + Attention',
    riskScore: 78,
    riskLevel: 'Critical',
    llmSummary: 'HPC 온도/정압 동반 상승, 200 cycle 내 정비 권고.',
    nasaScore: 1.62,
    operator: 'yoosehyeon98',
  },
  {
    id: 'RUN-2026-0517-B2',
    timestamp: '2026-05-17T16:02:00Z',
    dataset: 'AI4I 2020',
    model: 'iTransformer',
    riskScore: 42,
    riskLevel: 'Warning',
    llmSummary: '공구 마모 한계 접근, 다음 교대 후 교체 검토.',
    nasaScore: 0.94,
    operator: 'yoosehyeon98',
  },
  {
    id: 'RUN-2026-0517-C3',
    timestamp: '2026-05-17T08:31:00Z',
    dataset: 'CWRU',
    model: 'BiLSTM + Attention',
    riskScore: 15,
    riskLevel: 'Stable',
    llmSummary: 'Bearing 신호 안정, 잔존수명 양호.',
    nasaScore: 0.41,
    operator: 'yoosehyeon98',
  },
];
