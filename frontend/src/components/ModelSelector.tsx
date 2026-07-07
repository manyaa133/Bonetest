import type { ModelInfo, ModelType } from "../types";

interface ModelSelectorProps {
  models: ModelInfo[];
  selected: ModelType;
  onChange: (model: ModelType) => void;
  gender: string;
  onGenderChange: (gender: string) => void;
  disabled?: boolean;
}

export default function ModelSelector({
  models,
  selected,
  onChange,
  gender,
  onGenderChange,
  disabled,
}: ModelSelectorProps) {
  const current = models.find((m) => m.id === selected);

  return (
    <div className="space-y-4">
      <div>
        <label className="mb-1.5 block text-sm font-medium text-slate-700">
          Model
        </label>
        <select
          value={selected}
          onChange={(e) => onChange(e.target.value as ModelType)}
          disabled={disabled}
          className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm
                     focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-200"
        >
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.display_name}
              {!m.checkpoint_available ? " (no checkpoint)" : ""}
            </option>
          ))}
        </select>
        {current && (
          <p className="mt-1.5 text-xs text-slate-500">{current.description}</p>
        )}
      </div>

      {current?.requires_gender && (
        <div>
          <label className="mb-1.5 block text-sm font-medium text-slate-700">
            Patient Gender
          </label>
          <div className="flex gap-3">
            {(["male", "female"] as const).map((g) => (
              <label
                key={g}
                className={`flex flex-1 cursor-pointer items-center justify-center rounded-lg border px-4 py-2.5 text-sm font-medium capitalize transition
                  ${gender === g ? "border-primary-500 bg-primary-50 text-primary-700" : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50"}`}
              >
                <input
                  type="radio"
                  name="gender"
                  value={g}
                  checked={gender === g}
                  onChange={() => onGenderChange(g)}
                  className="sr-only"
                  disabled={disabled}
                />
                {g}
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
