import type { ComparisonData } from "../types";
import MetricsChart from "./MetricsChart";

interface ComparisonDashboardProps {
  data: ComparisonData | null;
  loading?: boolean;
  error?: string | null;
}

export default function ComparisonDashboard({
  data,
  loading,
  error,
}: ComparisonDashboardProps) {
  if (error) {
    return (
      <div className="card border-red-200 bg-red-50 text-red-700">
        <p className="font-medium">Failed to load metrics</p>
        <p className="mt-1 text-sm">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="card overflow-x-auto">
        <h2 className="mb-4 text-lg font-semibold text-slate-800">
          Model Performance Comparison
        </h2>
        {loading ? (
          <div className="h-32 animate-pulse rounded bg-slate-100" />
        ) : data ? (
          <>
            <table className="w-full min-w-[540px] text-left text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="pb-3 pr-4 font-medium">Model</th>
                  <th className="pb-3 pr-4 font-medium">MAE ↓</th>
                  <th className="pb-3 pr-4 font-medium">MSE ↓</th>
                  <th className="pb-3 pr-4 font-medium">RMSE ↓</th>
                  <th className="pb-3 font-medium">Samples</th>
                </tr>
              </thead>
              <tbody>
                {[...data.models]
                  .sort((a, b) => a.metrics.mae - b.metrics.mae)
                  .map((m) => (
                    <tr
                      key={m.model_type}
                      className={`border-b border-slate-100 ${
                        m.model_type === data.best_model ? "bg-primary-50" : ""
                      }`}
                    >
                      <td className="py-3 pr-4 font-medium text-slate-800">
                        {m.display_name}
                        {m.model_type === data.best_model && (
                          <span className="ml-2 rounded-full bg-primary-100 px-2 py-0.5 text-xs text-primary-700">
                            Best
                          </span>
                        )}
                      </td>
                      <td className="py-3 pr-4 tabular-nums">
                        {m.metrics.mae.toFixed(2)}
                      </td>
                      <td className="py-3 pr-4 tabular-nums">
                        {m.metrics.mse.toFixed(2)}
                      </td>
                      <td className="py-3 pr-4 tabular-nums">
                        {m.metrics.rmse.toFixed(2)}
                      </td>
                      <td className="py-3 tabular-nums">{m.num_samples}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
            <p className="mt-3 text-xs text-slate-500">
              Evaluated: {data.generated_at}
            </p>
          </>
        ) : null}
      </div>

      {data && <MetricsChart models={data.models} />}
    </div>
  );
}
