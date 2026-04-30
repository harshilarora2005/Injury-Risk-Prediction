const STAGES = [
  { key: "queued",      label: "Queued" },
  { key: "model",       label: "Model" },
  { key: "pose",        label: "Pose" },
  { key: "features",    label: "Features" },
  { key: "inference",   label: "Inference" },
  { key: "annotations", label: "Annotations" },
  { key: "overlay",     label: "Overlay" },
  { key: "timeline",    label: "Timeline" },
  { key: "report",      label: "Report" },
  { key: "done",        label: "Done" },
];

export default function StatusStepper({ status }) {
  if (!status) return null;
  const { stage, progress, message, error } = status;
  const activeIdx = Math.max(0, STAGES.findIndex(s => s.key === stage));

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="label-tiny">Pipeline</div>
          <h2 className="text-lg font-semibold">
            {error ? "Failed" : stage === "done" ? "Complete" : "Processing"}
          </h2>
        </div>
        <span className="font-mono text-sm text-muted">{progress}%</span>
      </div>

      <div className="h-2 bg-panel2 rounded overflow-hidden mb-4">
        <div
          className={`h-full transition-all duration-300 ${error ? "bg-risk-high" : "bg-risk-low"}`}
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="grid grid-cols-5 md:grid-cols-10 gap-1.5">
        {STAGES.map((s, i) => {
          const done   = i < activeIdx || stage === "done";
          const active = i === activeIdx && stage !== "done";
          return (
            <div key={s.key} className="text-center">
              <div className={`h-1.5 rounded-full mb-1.5 ${
                done ? "bg-risk-low" : active ? "bg-risk-med animate-pulse" : "bg-border"
              }`} />
              <div className={`text-[10px] uppercase tracking-wider ${
                active ? "text-ink" : done ? "text-muted" : "text-muted/60"
              }`}>{s.label}</div>
            </div>
          );
        })}
      </div>

      <div className={`mt-4 text-sm font-mono ${error ? "text-risk-high" : "text-muted"}`}>
        {error ? `Error: ${error}` : message}
      </div>
    </div>
  );
}
