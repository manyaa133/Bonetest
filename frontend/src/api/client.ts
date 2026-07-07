import axios from "axios";
import type {
  ComparisonData,
  EvaluationData,
  GradCamResult,
  ModelInfo,
  ModelType,
  PredictionResult,
} from "../types";

const baseURL = import.meta.env.VITE_API_URL || "";

export const api = axios.create({
  baseURL,
  timeout: 120000,
});

export async function fetchHealth(): Promise<{ status: string }> {
  const { data } = await api.get("/api/health");
  return data;
}

export async function fetchModels(): Promise<{
  models: ModelInfo[];
  default_model: string;
}> {
  const { data } = await api.get("/api/models");
  return data;
}

export async function predictBoneAge(
  file: File,
  modelType: ModelType,
  gender?: string
): Promise<PredictionResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_type", modelType);
  if (gender) form.append("gender", gender);

  const { data } = await api.post<PredictionResult>("/api/predict", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function generateGradCam(
  file: File,
  modelType: ModelType,
  gender?: string
): Promise<GradCamResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_type", modelType);
  if (gender) form.append("gender", gender);

  const { data } = await api.post<GradCamResult>("/api/gradcam", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function fetchMetrics(): Promise<EvaluationData> {
  const { data } = await api.get<EvaluationData>("/api/metrics");
  return data;
}

export async function fetchComparison(): Promise<ComparisonData> {
  const { data } = await api.get<ComparisonData>("/api/metrics/comparison");
  return data;
}

export function plotUrl(name: string): string {
  return `${baseURL}/api/metrics/plots/${name}`;
}
