export type ModelType = "cnn" | "cnn_dnn" | "multimodal_cnn" | "cnn_rf";

export interface ModelInfo {
  id: ModelType;
  display_name: string;
  description: string;
  requires_gender: boolean;
  supports_gradcam: boolean;
  checkpoint_available: boolean;
}

export interface PredictionResult {
  model_type: string;
  bone_age_months: number;
  confidence: number;
  gender_used?: string | null;
  processing_time_ms: number;
}

export interface GradCamResult {
  model_type: string;
  bone_age_months: number;
  confidence: number;
  heatmap_base64: string;
  overlay_base64: string;
  processing_time_ms: number;
}

export interface MetricValues {
  mae: number;
  mse: number;
  rmse: number;
}

export interface ModelMetrics {
  model_type: string;
  display_name: string;
  metrics: MetricValues;
  num_samples: number;
}

export interface ComparisonData {
  models: ModelMetrics[];
  best_model: string;
  generated_at: string;
}

export interface EvaluationData {
  comparison: ComparisonData;
  plots: Record<string, string>;
  details?: Record<string, unknown>;
}
