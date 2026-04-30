import { useEffect, useState } from "react";
import UploadPanel from "./components/UploadPanel.jsx";
import StatusStepper from "./components/StatusStepper.jsx";
import VideoPlayer from "./components/VideoPlayer.jsx";
import RiskTimeline from "./components/RiskTimeline.jsx";
import AnnotationList from "./components/AnnotationList.jsx";
import RiskSummaryTable from "./components/RiskSummaryTable.jsx";
import DownloadPanel from "./components/DownloadPanel.jsx";
import { getResult, streamJob } from "./api/client.js";

export default function App() {
  const [jobId, setJobId]       = useState(null);
  const [status, setStatus]     = useState(null);
  const [result, setResult]     = useState(null);
  const [originalUrl, setOrigUrl] = useState(null);

  useEffect(() => {
    if (!jobId) return;
    setStatus({ stage: "queued", progress: 0, message: "Connecting…" });
    setResult(null);
    const stop = streamJob(jobId, (snap) => {
      setStatus(snap);
      if (snap.stage === "done" && snap.result) setResult(snap.result);
    }, () => {

      getResult(jobId).then(setResult).catch(() => {});
    });
    return stop;
  }, [jobId]);

  const startJob = (id, file) => {
    setJobId(id);
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    setOrigUrl(file ? URL.createObjectURL(file) : null);
  };

  const reset = () => {
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    setJobId(null); setStatus(null); setResult(null); setOrigUrl(null);
  };

  const isDone = status?.stage === "done" && result;

  return (
    <div className="min-h-full">
      <header className="border-b border-border bg-panel/60 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-risk-high/15 border border-risk-high/40 grid place-items-center">
              <span className="font-mono text-risk-high font-bold">A</span>
            </div>
            <div>
              <div className="text-sm font-semibold leading-tight">ACL Risk Screening</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {jobId && (
              <button className="btn" onClick={reset}>New clip</button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        {!jobId && (
          <UploadPanel onJobStarted={startJob} />
        )}

        {jobId && (
          <>
            <StatusStepper status={status} />
            {isDone && (
              <>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-2 space-y-6">
                    <VideoPlayer jobId={jobId} originalUrl={originalUrl} />
                    <RiskTimeline jobId={jobId} />
                  </div>
                  <div className="space-y-6">
                    <RiskSummaryTable summary={result.risk_summary} />
                    <DownloadPanel jobId={jobId} />
                  </div>
                </div>
                <AnnotationList events={result.annotated_events} />
                <ClipMeta result={result} />
              </>
            )}
          </>
        )}
      </main>

    </div>
  );
}

function ClipMeta({ result }) {
  const items = [
    ["File",       result.filename],
    ["Duration",   `${result.duration_sec.toFixed(2)} s`],
    ["FPS",        result.fps.toFixed(1)],
    ["Frames",     result.total_frames],
    ["Camera",     result.camera_angle],
    ["Job ID",     result.job_id.slice(0, 12) + "…"],
  ];
  return (
    <div className="card">
      <div className="label-tiny mb-3">Clip metadata</div>
      <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
        {items.map(([k, v]) => (
          <div key={k}>
            <div className="text-[11px] text-muted">{k}</div>
            <div className="font-mono text-sm truncate">{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

