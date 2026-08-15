from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import get_graph, router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 70)
    print("Starting 3GPP RAG backend")
    print("=" * 70)

    try:
        print("Initializing RAG graph...")
        get_graph()
        print("RAG graph initialized successfully.")
    except Exception as exc:
        print("WARNING: RAG graph initialization failed:")
        print(repr(exc))
        print(
            "The API will start, but /api/chat may fail until "
            "the underlying RAG configuration is fixed."
        )

    yield

    print("Shutting down 3GPP RAG backend...")


app = FastAPI(
    title="3GPP RAG API",
    description=(
        "FastAPI backend for the 3GPP Retrieval-Augmented "
        "Generation system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Set FRONTEND_URL on Render to the deployed Vercel URL.
# Multiple comma-separated origins are supported.
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

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
