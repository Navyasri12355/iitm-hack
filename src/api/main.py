"""
Clinical Evidence Copilot — FastAPI Application

Real-time clinical evidence retrieval and recommendation system.
"""

import logging
import json
import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import get_settings
from ..models.core import (
    ClinicalQuery, ClinicalRecommendation, UrgencyLevel,
    ParsedDocument, DocumentType
)
from ..ingestion.pipeline import get_vector_store, get_pipeline, init_pipeline
from ..reasoning.engine import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self._connections: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, client_id: str):
        await ws.accept()
        async with self._lock:
            self._connections[client_id] = ws
        logger.info(f"WS connected: {client_id} ({len(self._connections)} total)")

    async def disconnect(self, client_id: str):
        async with self._lock:
            self._connections.pop(client_id, None)
        logger.info(f"WS disconnected: {client_id}")

    async def broadcast(self, message: Dict[str, Any]):
        payload = json.dumps(message, default=str)
        async with self._lock:
            dead = []
            for cid, ws in self._connections.items():
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(cid)
            for cid in dead:
                self._connections.pop(cid, None)

    async def send_to(self, client_id: str, message: Dict[str, Any]):
        async with self._lock:
            ws = self._connections.get(client_id)
        if ws:
            try:
                await ws.send_text(json.dumps(message, default=str))
            except Exception:
                await self.disconnect(client_id)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


ws_manager = ConnectionManager()

# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query_text: str = Field(..., min_length=3, max_length=2000)
    clinician_id: str = Field(default="anonymous")
    urgency_level: UrgencyLevel = UrgencyLevel.ROUTINE
    patient_context: Optional[Dict[str, Any]] = None


class DocumentUploadRequest(BaseModel):
    title: str
    content: str
    authors: List[str] = Field(default_factory=list)
    source: str = ""
    document_type: str = "research_paper"


class QueryResponse(BaseModel):
    query_id: str
    recommendation: Dict[str, Any]
    processing_time_seconds: float
    document_count: int


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("🚀 Starting Clinical Evidence Copilot...")

    if not settings.openai_api_key:
        logger.warning("⚠️  OPENAI_API_KEY not set — LLM features will use fallback mode")

    # Initialize Pathway ingestion pipeline
    try:
        pipeline = init_pipeline(
            documents_path=settings.documents_path,
            openai_api_key=settings.openai_api_key or "",
            embedding_model=settings.embedding_model,
        )
        logger.info(f"✅ Pathway pipeline watching: {settings.documents_path}")
    except Exception as e:
        logger.error(f"Pipeline init failed: {e}")

    yield

    # Shutdown
    pipeline = get_pipeline()
    if pipeline:
        pipeline.stop()
    logger.info("👋 Clinical Evidence Copilot shut down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Clinical Evidence Copilot",
    description="Real-time evidence-backed clinical decision support",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent.parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text())
    return HTMLResponse("<h1>Clinical Evidence Copilot</h1><p>Frontend not found.</p>")


@app.get("/health")
async def health():
    store = get_vector_store()
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "documents_indexed": store.document_count,
        "pipeline_running": pipeline is not None and pipeline._running,
        "websocket_connections": ws_manager.connection_count,
    }


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a clinical query and return evidence-backed recommendations."""
    start = datetime.now()
    settings = get_settings()

    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY in environment."
        )

    query_id = f"q_{int(start.timestamp())}_{request.clinician_id[:8]}"

    query = ClinicalQuery(
        id=query_id,
        query_text=request.query_text,
        clinician_id=request.clinician_id,
        urgency_level=request.urgency_level,
        patient_context=request.patient_context,
        timestamp=start,
    )

    try:
        engine = get_engine()
        recommendation = engine.generate(query)
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reasoning engine error: {str(e)}")

    processing_time = (datetime.now() - start).total_seconds()

    # Broadcast to all connected clients
    await ws_manager.broadcast({
        "type": "new_recommendation",
        "query_id": query_id,
        "query_text": request.query_text[:100],
        "confidence": recommendation.confidence_score,
        "evidence_count": len(recommendation.supporting_evidence),
        "timestamp": datetime.now().isoformat(),
    })

    rec_dict = recommendation.model_dump()
    # Make evidence serializable
    rec_dict["supporting_evidence"] = [e.model_dump() for e in recommendation.supporting_evidence]

    return QueryResponse(
        query_id=query_id,
        recommendation=rec_dict,
        processing_time_seconds=round(processing_time, 2),
        document_count=get_vector_store().document_count,
    )


@app.get("/api/documents")
async def list_documents(limit: int = 50, offset: int = 0):
    """List all indexed documents."""
    store = get_vector_store()
    docs = store.list_documents()
    return {
        "documents": docs[offset:offset + limit],
        "total": len(docs),
        "limit": limit,
        "offset": offset,
    }


@app.post("/api/documents")
async def upload_document(request: DocumentUploadRequest):
    """Upload and index a new document."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OpenAI API key required for indexing")

    pipeline = get_pipeline()
    if not pipeline:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not running")

    timestamp = int(datetime.now().timestamp())
    filename = f"upload_{timestamp}_{request.title[:30].replace(' ', '_').lower()}.txt"

    # Build document content
    content_parts = []
    if request.title:
        content_parts.append(f"# {request.title}\n")
    if request.authors:
        content_parts.append(f"Authors: {', '.join(request.authors)}\n")
    if request.source:
        content_parts.append(f"Source: {request.source}\n")
    content_parts.append(f"Document Type: {request.document_type}\n\n")
    content_parts.append(request.content)

    full_content = "\n".join(content_parts)

    try:
        doc_id = pipeline.ingest_document(full_content, filename)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

    # Broadcast document addition
    await ws_manager.broadcast({
        "type": "document_indexed",
        "doc_id": doc_id,
        "title": request.title,
        "timestamp": datetime.now().isoformat(),
        "document_count": get_vector_store().document_count,
    })

    return {
        "id": doc_id,
        "title": request.title,
        "message": "Document indexed successfully",
        "document_count": get_vector_store().document_count,
    }


@app.post("/api/documents/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for indexing."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OpenAI API key required")

    pipeline = get_pipeline()
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not running")

    content = await file.read()
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    if len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="File content too short")

    filename = file.filename or f"upload_{int(datetime.now().timestamp())}.txt"
    try:
        doc_id = pipeline.ingest_document(text, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    await ws_manager.broadcast({
        "type": "document_indexed",
        "doc_id": doc_id,
        "title": filename,
        "timestamp": datetime.now().isoformat(),
        "document_count": get_vector_store().document_count,
    })

    return {"id": doc_id, "filename": filename, "document_count": get_vector_store().document_count}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from the index."""
    store = get_vector_store()
    store.remove(doc_id)
    await ws_manager.broadcast({
        "type": "document_removed",
        "doc_id": doc_id,
        "document_count": store.document_count,
        "timestamp": datetime.now().isoformat(),
    })
    return {"message": f"Document {doc_id} removed", "document_count": store.document_count}


@app.get("/api/stats")
async def get_stats():
    """System statistics."""
    store = get_vector_store()
    pipeline = get_pipeline()
    return {
        "documents_indexed": store.document_count,
        "chunks_indexed": len(store._chunks),
        "pipeline_running": pipeline is not None and pipeline._running,
        "websocket_connections": ws_manager.connection_count,
        "documents_path": str(pipeline.documents_path) if pipeline else None,
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await ws_manager.connect(websocket, client_id)
    try:
        # Send current state on connect
        store = get_vector_store()
        await ws_manager.send_to(client_id, {
            "type": "connected",
            "message": f"Connected to Clinical Evidence Copilot",
            "document_count": store.document_count,
            "timestamp": datetime.now().isoformat(),
        })

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws_manager.send_to(client_id, {
                        "type": "pong",
                        "timestamp": datetime.now().isoformat(),
                        "document_count": store.document_count,
                    })
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await ws_manager.send_to(client_id, {
                        "type": "heartbeat",
                        "document_count": store.document_count,
                        "timestamp": datetime.now().isoformat(),
                    })
                except Exception:
                    break
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"WS error for {client_id}: {e}")
                break
    finally:
        await ws_manager.disconnect(client_id)


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )