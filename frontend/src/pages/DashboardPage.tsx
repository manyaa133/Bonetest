import { useEffect, useState } from "react";
import { fetchComparison } from "../api/client";
import ComparisonDashboard from "../components/ComparisonDashboard";
import type { ComparisonData } from "../types";

export default function DashboardPage() {
  const [data, setData] = useState<ComparisonData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchComparison()
      .then(setData)
      .catch(() => setError("Could not load evaluation metrics from API"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">
          Model Comparison Dashboard
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          Validation-set performance across all four bone age regression models
          (MAE, MSE, RMSE).
        </p>
      </div>
      <ComparisonDashboard data={data} loading={loading} error={error} />
    </div>
  );
}
