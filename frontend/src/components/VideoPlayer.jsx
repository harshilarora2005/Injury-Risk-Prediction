import { artifactUrl } from "../api/client";

export default function VideoPlayer({ jobId, originalUrl }) {
  const overlay = artifactUrl(jobId, "output_skeleton_overlay.mp4");
  console.log(overlay);
  return (
    <div className="card">
      <div className="label-tiny mb-1">Step 4</div>
      <h2 className="text-lg font-semibold mb-4">Skeleton overlay</h2>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {originalUrl && (
          <div>
            <div className="label-tiny mb-2">Original</div>
            <video src={originalUrl} controls className="w-full rounded-lg bg-black" />
          </div>
        )}
        <div>
          <div className="label-tiny mb-2 flex items-center gap-2">
            Annotated
            <span className="chip border-risk-high/40 text-risk-high bg-risk-high/10">Color-coded risk</span>
          </div>
          <video src={overlay} controls className="w-full rounded-lg bg-black shadow-glow" />
        </div>
      </div>
    </div>
  );
}
