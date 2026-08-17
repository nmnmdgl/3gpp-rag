# backend/api/main.py

from contextlib import asynccontextmanager
import asyncio
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import initialize_graph, router


async def initialize_rag():
    print("=" * 70, flush=True)
    print("Background RAG initialization started", flush=True)
    print("=" * 70, flush=True)

    try:
        await initialize_graph()

        print("=" * 70, flush=True)
        print("RAG graph initialized successfully.", flush=True)
        print("=" * 70, flush=True)

    except Exception as exc:
        print("=" * 70, flush=True)
        print("WARNING: Background RAG initialization failed:", flush=True)
        print(repr(exc), flush=True)
        print("=" * 70, flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 70, flush=True)
    print("Starting 3GPP RAG backend", flush=True)
    print("=" * 70, flush=True)

    asyncio.create_task(initialize_rag())

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