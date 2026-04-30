# ACL Risk Screening — Frontend (Phase 7)

React + Vite + Tailwind UI for the FastAPI backend.

## Run

```bash
npm install
npm run dev          # http://localhost:5173
```

Vite proxies `/api` → `http://localhost:8000`, so start the backend first:

```bash
cd ../backend && uvicorn main:app --reload --port 8000
```

## What it does

1. **UploadPanel** — drag-and-drop a clip, pick camera angle, POST `/api/upload`.
2. **StatusStepper** — subscribes to `/api/jobs/{id}/stream` (SSE) and renders the 10-stage pipeline.
3. When the job is `done`, the result panel appears:
   - **VideoPlayer** — original vs `output_skeleton_overlay.mp4` side-by-side.
   - **RiskTimeline** — embeds `risk_timeline.png`.
   - **RiskSummaryTable** — Low/Medium/High counts + peak window.
   - **AnnotationList** — high-risk events with dominant biomechanical signal.
   - **DownloadPanel** — direct links to all 5 artifacts (PDF, MP4, PNG, TXT, CSV).

## Files

```
src/
  App.jsx
  main.jsx
  index.css
  api/client.js              # axios + SSE
  lib/format.js
  components/
    UploadPanel.jsx
    StatusStepper.jsx
    VideoPlayer.jsx
    RiskTimeline.jsx
    AnnotationList.jsx
    RiskSummaryTable.jsx
    DownloadPanel.jsx
```

SCREENING ONLY — NOT A CLINICAL DIAGNOSIS.
