interface GradCamViewerProps {
  overlayBase64: string | null;
  heatmapBase64: string | null;
  loading?: boolean;
  error?: string | null;
  showHeatmap: boolean;
  onToggleView: () => void;
}

export default function GradCamViewer({
  overlayBase64,
  heatmapBase64,
  loading,
  error,
  showHeatmap,
  onToggleView,
}: GradCamViewerProps) {
  const src = showHeatmap ? heatmapBase64 : overlayBase64;

  return (
    <div className="card">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-800">
          Grad-CAM Explainability
        </h3>
        {src && (
          <button
            type="button"
            onClick={onToggleView}
            className="text-xs font-medium text-primary-600 hover:text-primary-700"
          >
            {showHeatmap ? "Show overlay" : "Show heatmap only"}
          </button>
        )}
      </div>

      {loading && (
        <div className="flex aspect-square animate-pulse items-center justify-center rounded-lg bg-slate-100">
          <p className="text-sm text-slate-500">Generating heatmap…</p>
        </div>
      )}

      {error && (
        <div className="rounded-lg bg-red-50 p-4 text-sm text-red-600">{error}</div>
      )}

      {!loading && !error && !src && (
        <div className="flex aspect-square items-center justify-center rounded-lg border border-dashed border-slate-200 bg-slate-50">
          <p className="text-sm text-slate-500">
            Run Grad-CAM to see skeletal attention regions
          </p>
        </div>
      )}

      {src && !loading && (
        <div className="overflow-hidden rounded-lg border border-slate-200">
          <img
            src={`data:image/png;base64,${src}`}
            alt="Grad-CAM visualization"
            className="w-full object-contain"
          />
        </div>
      )}

      <p className="mt-2 text-xs text-slate-500">
        Warmer colors indicate regions that most influenced the bone age prediction
        (epiphyses, metaphyses, carpal bones).
      </p>
    </div>
  );
}
