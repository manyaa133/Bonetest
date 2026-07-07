import { useState } from "react";
import DashboardPage from "./pages/DashboardPage";
import PredictPage from "./pages/PredictPage";

type Tab = "predict" | "dashboard";

export default function App() {
  const [tab, setTab] = useState<Tab>("predict");

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
          <div>
            <h1 className="text-xl font-bold text-slate-900 sm:text-2xl">
              Bone Age Predictor
            </h1>
            <p className="text-sm text-slate-500">
              Pediatric skeletal maturity from hand X-rays · Deep learning regression
            </p>
          </div>
          <nav className="flex gap-1 rounded-lg bg-slate-100 p-1">
            {(
              [
                ["predict", "Predict"],
                ["dashboard", "Model Comparison"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setTab(id)}
                className={`rounded-md px-4 py-2 text-sm font-medium transition ${
                  tab === id
                    ? "bg-white text-primary-700 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {tab === "predict" ? <PredictPage /> : <DashboardPage />}
      </main>

      <footer className="border-t border-slate-200 py-6 text-center text-xs text-slate-400">
        Bone Age Predictor · For research &amp; educational use · Not for clinical diagnosis
      </footer>
    </div>
  );
}
