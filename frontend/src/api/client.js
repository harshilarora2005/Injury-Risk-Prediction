import axios from "axios";

const api = axios.create({ baseURL: "/api" });

export async function uploadVideo(file, cameraAngle = "sagittal", onProgress) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("camera_angle", cameraAngle);
  const { data } = await api.post("/upload", fd, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100));
    },
  });
  return data; // { job_id, status_url }
}

export async function getJob(jobId) {
  const { data } = await api.get(`/jobs/${jobId}`);
  return data;
}

export async function getResult(jobId) {
  const { data } = await api.get(`/jobs/${jobId}/result`);
  return data;
}

export function artifactUrl(jobId, name) {
  return `/api/jobs/${jobId}/artifacts/${name}`;
}

/**
 * Subscribe to SSE progress for a job.
 * onMessage receives the parsed JobStatus snapshot.
 * Returns a cleanup function.
 */
export function streamJob(jobId, onMessage, onError) {
  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  es.onmessage = (ev) => {
    try { onMessage(JSON.parse(ev.data)); }
    catch (e) { console.warn("SSE parse error", e); }
  };
  es.onerror = (e) => {
    onError?.(e);
    es.close();
  };
  return () => es.close();
}
