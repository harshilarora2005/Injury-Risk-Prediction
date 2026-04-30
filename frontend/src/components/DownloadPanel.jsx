import { artifactUrl } from "../api/client";

const FILES = [
  { name: "summary_report.pdf",         label: "Summary report (PDF)",   primary: true },
  { name: "output_skeleton_overlay.mp4",label: "Overlay video (MP4)" },
  { name: "risk_timeline.png",          label: "Risk timeline (PNG)" },
  { name: "movement_annotations.txt",   label: "Annotations (TXT)" },
  { name: "per_window_predictions.csv", label: "Predictions (CSV)" },
];

export default function DownloadPanel({ jobId }) {
  return (
    <div className="card">
      <div className="label-tiny mb-1">Step 7</div>
      <h2 className="text-lg font-semibold mb-4">Downloads</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {FILES.map(f => (
          <a key={f.name}
              href={artifactUrl(jobId, f.name)}
              download
              className={f.primary ? "btn-primary" : "btn"}>
              ↓ {f.label}
          </a>
        ))}
      </div>
    </div>
  );
}
