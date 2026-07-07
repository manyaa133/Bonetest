import type { PredictionResult } from "../types";

interface PredictionResultCardProps {
  result: PredictionResult | null;
  loading?: boolean;
  error?: string | null;
}

function yearsMonths(totalMonths: number): string {
  const years = Math.floor(totalMonths / 12);
  const months = Math.round(totalMonths % 12);
  return `${years}y ${months}m`;
}

export default function PredictionResultCard({
  result,
  loading,
  error,
}: PredictionResultCardProps) {
  if (loading) {
    return (
      <div className="card animate-pulse">
        <div className="h-4 w-32 rounded bg-slate-200" />
        <div className="mt-4 h-12 w-48 rounded bg-slate-200" />
        <div className="mt-3 h-3 w-full rounded bg-slate-100" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="card border-red-200 bg-red-50">
        <p className="text-sm font-medium text-red-800">Prediction Error</p>
        <p className="mt-1 text-sm text-red-600">{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card text-center text-slate-500">
        <p className="text-sm">Upload an X-ray and run prediction</p>
      </div>
    );
  }

  const confidencePct = Math.round(result.confidence * 100);

  return (
    <div className="card">
      <p className="text-sm font-medium text-slate-500">Predicted Bone Age</p>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="text-4xl font-bold text-primary-700">
          {result.bone_age_months}
        </span>
        <span className="text-lg text-slate-500">months</span>
      </div>
      <p className="mt-1 text-sm text-slate-600">
        ≈ {yearsMonths(result.bone_age_months)}
      </p>

      <div className="mt-5">
        <div className="mb-1 flex justify-between text-xs">
          <span className="font-medium text-slate-600">Confidence</span>
          <span className="text-slate-500">{confidencePct}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-slate-200">
          <div
            className="h-full rounded-full bg-gradient-to-r from-primary-500 to-medical-accent transition-all"
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-500">
        <span className="rounded-full bg-slate-100 px-2.5 py-1">
          Model: {result.model_type}
        </span>
        <span className="rounded-full bg-slate-100 px-2.5 py-1">
          {result.processing_time_ms.toFixed(0)} ms
        </span>
        {result.gender_used && (
          <span className="rounded-full bg-slate-100 px-2.5 py-1 capitalize">
            Gender: {result.gender_used}
          </span>
        )}
      </div>
    </div>
  );
}
