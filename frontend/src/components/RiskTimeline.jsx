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
    </div>
  );
}
