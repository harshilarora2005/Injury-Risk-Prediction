import { artifactUrl } from "../api/client";

export default function RiskTimeline({ jobId }) {
  const url = artifactUrl(jobId, "risk_timeline.png");
  return (
    <div className="card">
      <div className="label-tiny mb-1">Step 5</div>
      <h2 className="text-lg font-semibold mb-4">Risk score timeline</h2>
      <a href={url} target="_blank" rel="noreferrer" className="block">
        <img src={url} alt="Risk timeline" className="w-full rounded-lg border border-border" />
      </a>
      <p className="mt-3 text-xs text-muted">
        Top: P(High) per window with 0.33 / 0.66 thresholds. Bottom: smoothed L/R knee
        flexion with asymmetry shading and ground-contact ticks.
      </p>
    </div>
  );
}
