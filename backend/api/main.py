from contextlib import asynccontextmanager

import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import (
    router,
    start_graph_initialization,
)


# =========================================================
# LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 70)
    print("Starting 3GPP RAG backend")
    print("=" * 70)

    # -----------------------------------------------------
    # DO NOT BLOCK STARTUP
    # -----------------------------------------------------

    print("=" * 70)
    print("Starting background RAG initialization")
    print("=" * 70)

    start_graph_initialization()

    yield

    print("=" * 70)
    print("Shutting down 3GPP RAG backend...")
    print("=" * 70)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="3GPP RAG API",
    description=(
        "FastAPI backend for the 3GPP "
        "Retrieval-Augmented Generation system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

import os


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

    allowed_origins.append(
        "http://localhost:3000"
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ROUTES
# =========================================================

app.include_router(router)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "service": "3GPP RAG API",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
        "rag_status": "/api/rag-status",
    }