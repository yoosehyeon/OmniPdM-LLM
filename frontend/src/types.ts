// 도메인 타입 — pages/* 와 components/* 가 공통으로 참조.
// 백엔드 응답 스키마와 1:1 동기화 필요: services/api/* 응답 모델 변경 시 같이 갱신.

export type RiskLevel = 'Stable' | 'Warning' | 'Critical';

export interface SensorSpec {
  name: string;
  defaultValue: number;
  min: number;
  max: number;
  unit: string;
}

export interface DatasetConfig {
  id: string;
  label: string;
  sensors: SensorSpec[];
}

export interface CoreModel {
  id: string;
  label: string;
  description?: string;
}

export interface RulPoint {
  time: number;
  predicted: number;
  actual?: number;
  lower?: number;
  upper?: number;
}

export interface FeatureImportanceItem {
  name: string;
  value: number;
}

export interface AttentionCell {
  row: number;
  col: number;
  val: number;
}

export interface TemporalContributionPoint {
  time: number;
  sensors: number;
  loadFactor: number;
  interaction: number;
}

export interface AnalysisResult {
  riskScore: number;
  riskLevel: RiskLevel;
  rulTrend: RulPoint[];
  featureImportance: FeatureImportanceItem[];
  attentionMap: AttentionCell[];
  temporalContribution: TemporalContributionPoint[];
  llmAnalysis: string;
  judgeScore: number;
  guardrailViolation: string;
}

export interface ReportItem {
  id: string;
  timestamp: string;
  dataset: string;
  model: string;
  riskScore: number;
  riskLevel: RiskLevel;
  llmSummary: string;
  nasaScore: number;
  operator: string;
}
