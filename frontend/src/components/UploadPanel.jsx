import { useRef, useState } from "react";
import { uploadVideo } from "../api/client";

const ACCEPT = ".mp4,.mov,.avi,.mkv,.webm";

export default function UploadPanel({ onJobStarted, disabled }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);
  const [file, setFile] = useState(null);
  const [angle, setAngle] = useState("sagittal");
  const [uploadPct, setUploadPct] = useState(0);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const pick = (f) => { setErr(null); setFile(f); setUploadPct(0); };

  const onDrop = (e) => {
    e.preventDefault(); setDrag(false);
    const f = e.dataTransfer.files?.[0]; if (f) pick(f);
  };

  const submit = async () => {
    if (!file) return;
    setBusy(true); setErr(null);
    try {
      const { job_id } = await uploadVideo(file, angle, setUploadPct);
      onJobStarted(job_id, file);
    } catch (e) {
      setErr(e?.response?.data?.detail || e.message || "Upload failed");
    } finally { setBusy(false); }
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="label-tiny">Step 1</div>
          <h2 className="text-lg font-semibold">Upload movement clip</h2>
        </div>
        <span className="chip border-border text-muted">MP4 · MOV · WEBM · ≤20s recommended</span>
      </div>

      <div
        onDragOver={(e)=>{e.preventDefault(); setDrag(true);}}
        onDragLeave={()=>setDrag(false)}
        onDrop={onDrop}
        onClick={()=>inputRef.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-colors
          ${drag ? "border-risk-high bg-risk-high/5" : "border-border hover:border-muted/60 bg-panel2/40"}`}
      >
        <input ref={inputRef} type="file" accept={ACCEPT} className="hidden"
          onChange={(e)=> e.target.files?.[0] && pick(e.target.files[0])} />
        {file ? (
          <div>
            <div className="font-medium">{file.name}</div>
            <div className="text-xs text-muted mt-1">
              {(file.size/1024/1024).toFixed(2)} MB · click to change
            </div>
          </div>
        ) : (
          <div>
            <div className="text-sm font-medium">Drop a video here, or click to browse</div>
            <div className="text-xs text-muted mt-1">FPS ≥ 25 · front-facing or sagittal only</div>
          </div>
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="md:col-span-2">
          <div className="label-tiny mb-1">Camera angle</div>
          <div className="flex gap-2">
            {[
              {v:"sagittal", lbl:"Sagittal (side)"},
              {v:"front",    lbl:"Front-facing"},
            ].map(o => (
              <button key={o.v} onClick={()=>setAngle(o.v)}
                className={`btn flex-1 ${angle===o.v ? "border-risk-high text-risk-high bg-risk-high/10" : ""}`}>
                {o.lbl}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-end">
          <button className="btn-primary w-full" disabled={!file || busy || disabled} onClick={submit}>
            {busy ? `Uploading ${uploadPct}%` : "Analyze clip"}
          </button>
        </div>
      </div>

      {uploadPct > 0 && busy && (
        <div className="mt-3 h-1.5 bg-panel2 rounded overflow-hidden">
          <div className="h-full bg-risk-high transition-all" style={{width: `${uploadPct}%`}} />
        </div>
      )}
      {err && <div className="mt-3 text-sm text-risk-high">{err}</div>}
    </div>
  );
}
