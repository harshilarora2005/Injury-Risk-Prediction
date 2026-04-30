from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference import load_model
from routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("phase7")

app = FastAPI(
    title="ACL Risk Screening — Phase 7 API",
    version="1.0.0",
    description="Pose extraction → BiLSTM inference → overlay/timeline/report.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
def _startup():
    try:
        _, meta = load_model()
        log.info(
            f"Model loaded OK — "
            f"input_size={meta['input_size']}, "
            f"window_size={meta['window_size']}, "
            f"labels={meta['labels']}, "
            f"device={meta['device']}"
        )
    except FileNotFoundError as e:
        log.warning(f"Model not loaded at startup (will retry on first request): {e}")
    except Exception as e:
        log.exception(f"Unexpected error loading model at startup: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)