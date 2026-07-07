import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ModelMetrics } from "../types";

interface MetricsChartProps {
  models: ModelMetrics[];
  loading?: boolean;
}

const COLORS = ["#3b82f6", "#0ea5e9", "#6366f1", "#8b5cf6"];

export default function MetricsChart({ models, loading }: MetricsChartProps) {
  if (loading) {
    return (
      <div className="card h-80 animate-pulse">
        <div className="h-full rounded bg-slate-100" />
      </div>
    );
  }

  const maeData = models.map((m, i) => ({
    name: m.display_name.replace(" + ", "\n+ "),
    value: m.metrics.mae,
    fill: COLORS[i % COLORS.length],
  }));

  const rmseData = models.map((m, i) => ({
    name: m.display_name.replace(" + ", "\n+ "),
    value: m.metrics.rmse,
    fill: COLORS[i % COLORS.length],
  }));

  const mseData = models.map((m, i) => ({
    name: m.display_name.replace(" + ", "\n+ "),
    value: m.metrics.mse,
    fill: COLORS[i % COLORS.length],
  }));

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      {[
        { title: "Mean Absolute Error (months)", data: maeData },
        { title: "Root Mean Squared Error (months)", data: rmseData },
        { title: "Mean Squared Error", data: mseData },
      ].map(({ title, data }) => (
        <div key={title} className="card">
          <h3 className="mb-4 text-sm font-semibold text-slate-800">{title}</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={data} margin={{ top: 5, right: 5, left: -10, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 10 }}
                interval={0}
                angle={-20}
                textAnchor="end"
                height={50}
              />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip
                formatter={(v: number) => [v.toFixed(2), "Value"]}
                contentStyle={{ borderRadius: 8, fontSize: 12 }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ))}
    </div>
  );
}
