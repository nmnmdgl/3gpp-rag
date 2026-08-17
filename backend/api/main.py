from contextlib import asynccontextmanager
import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import initialize_graph_background, router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 70, flush=True)
    print("Starting 3GPP RAG backend", flush=True)
    print("=" * 70)

    # Start RAG initialization in a completely separate thread.
    # FastAPI startup does NOT wait for the RAG models/indexes.
    thread = threading.Thread(
        target=initialize_graph_background,
        daemon=True,
        name="rag-initializer",
    )

    thread.start()

    # Return immediately so Railway can reach /api/health.
    yield

    print(
        "Shutting down 3GPP RAG backend...",
        flush=True,
    )


app = FastAPI(
    title="3GPP RAG API",
    description=(
        "FastAPI backend for the 3GPP Retrieval-Augmented "
        "Generation system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


frontend_url = os.getenv(
    "FRONTEND_URL",
    "http://localhost:3000",
)

allowed_origins = [
    origin.strip()
    for origin in frontend_url.split(",")
    if origin.strip()
]

if "http://localhost:3000" not in allowed_origins:
    allowed_origins.append("http://localhost:3000")


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "3GPP RAG API",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
    }