"""
FastAPI Application — Purplle Retail Analytics API

Main application entry point for the REST API.
Serves metrics and event data from the detection pipeline.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import structlog

from app.database import EventDatabase
from app.routes import router, init_routes

from fastapi.staticfiles import StaticFiles

# ─── Logging Setup ───────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger(__name__)


# ─── Application Lifecycle ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown."""
    logger.info("Starting Purplle Retail Analytics API...")

    # Initialize database
    db = EventDatabase()
    init_routes(db)

    logger.info("API ready. Database initialized", path=db.db_path)
    yield
    logger.info("Shutting down API...")


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Purplle Retail Analytics API",
    description=(
        "REST API for retail store analytics. "
        "Ingests detection events from the CV pipeline and serves "
        "metrics including visitor counts, conversion rates, "
        "dwell times, queue depth, and conversion funnels."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)

# Dashboard
app.mount("/dashboard", StaticFiles(directory="app/static", html=True), name="dashboard")


# ─── Root ────────────────────────────────────────────────────────────────────

from fastapi.responses import RedirectResponse

@app.get("/")
async def root():
    """Redirect to the interactive dashboard."""
    return RedirectResponse(url="/dashboard/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
