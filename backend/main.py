from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference import load_model
from routes import router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("phase7")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ACL Risk Screening — Phase 7 API",
    version="1.0.0",
    description="Pose extraction → BiLSTM inference → overlay/timeline/report.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


# ── Startup: warm the model ───────────────────────────────────────────────────
@app.on_event("startup")
def _startup():
    try:
        _, meta = load_model()
        log.info(
            f"Model loaded. Features={len(meta['feature_schema'])}, "
            f"window_size={meta['window_size']}, labels={meta['label_map']}"
        )
    except FileNotFoundError as e:
        log.warning(f"Model not loaded at startup: {e}")
    except Exception as e:
        log.exception(f"Unexpected error loading model: {e}")


# ── Local dev entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
