"""
FastAPI main application for Clinical Evidence Copilot.

This module implements the REST API endpoints for:
- /query endpoint for clinical questions
- /documents endpoint for document management  
- /recommendations endpoint for tracking changes

Validates Requirements 5.1, 5.3:
- Provide API endpoints for seamless integration
- Format outputs compatible with electronic health record systems
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json

from ..config import get_settings
from ..models.core import (
    ClinicalQuery, ClinicalRecommendation, ParsedDocument, 
    UrgencyLevel, PatientContext, Evidence
)
from .models import (
    QueryRequest, QueryResponse, DocumentUploadRequest, DocumentResponse,
    RecommendationHistoryResponse, HealthCheckResponse, ErrorResponse,
    NotificationRequest, NotificationResponse
)
from .services import ClinicalService
from .websocket import websocket_manager, WebSocketManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
clinical_service: Optional[ClinicalService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global clinical_service
    
    # Startup
    logger.info("Starting Clinical Evidence Copilot API...")
    try:
        clinical_service = ClinicalService()
        await clinical_service.initialize()
        logger.info("Clinical service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize clinical service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Clinical Evidence Copilot API...")
    if clinical_service:
        await clinical_service.cleanup()


# Create FastAPI application
app = FastAPI(
    title="Clinical Evidence Copilot API",
    description="Real-time evidence-backed medical information system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_clinical_service() -> ClinicalService:
    """Dependency to get the clinical service instance."""
    if clinical_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clinical service not initialized"
        )
    return clinical_service


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )


@app.post("/query", response_model=QueryResponse)
async def process_clinical_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    service: ClinicalService = Depends(get_clinical_service)
):
    """
    Process a clinical query and return evidence-backed recommendations.
    
    This endpoint implements the core functionality for clinicians to submit
    medical questions and receive real-time, evidence-based responses.
    
    Validates Requirements 1.1, 1.2, 1.3, 1.4:
    - Provide evidence-backed answers within 30 seconds
    - Cite specific medical literature sources with publication dates
    - Rank recommendations based on current evidence strength
    - Flag contradictions and explain differences
    """
    try:
        logger.info(f"Processing clinical query from clinician {request.clinician_id}")
        
        # Create ClinicalQuery object
        clinical_query = ClinicalQuery(
            id=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.clinician_id}",
            query_text=request.query_text,
            clinician_id=request.clinician_id,
            patient_context=request.patient_context,
            urgency_level=request.urgency_level,
            timestamp=datetime.now()
        )
        
        # Process the query
        recommendation = await service.process_query(clinical_query)
        
        # Schedule background tasks for notifications if needed
        if recommendation.change_reason:
            background_tasks.add_task(
                service.notify_recommendation_change,
                recommendation
            )
            # Also notify via WebSocket
            background_tasks.add_task(
                websocket_manager.notify_recommendation_change,
                recommendation,
                clinical_query
            )
        
        # Format response for EHR compatibility
        response = QueryResponse(
            query_id=clinical_query.id,
            recommendation=recommendation,
            processing_time_seconds=(datetime.now() - clinical_query.timestamp).total_seconds(),
            citations=service.get_citations_for_recommendation(recommendation),
            reasoning_steps=service.get_reasoning_steps(recommendation)
        )
        
        logger.info(f"Successfully processed query {clinical_query.id} in {response.processing_time_seconds:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error processing clinical query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing clinical query: {str(e)}"
        )


@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    limit: int = 100,
    offset: int = 0,
    document_type: Optional[str] = None,
    service: ClinicalService = Depends(get_clinical_service)
):
    """
    List medical documents in the knowledge base.
    
    Provides document management capabilities for administrators
    to view and manage the medical literature collection.
    """
    try:
        documents = await service.list_documents(
            limit=limit,
            offset=offset,
            document_type=document_type
        )
        
        response = [
            DocumentResponse(
                id=doc.id,
                title=doc.title,
                authors=doc.authors,
                publication_date=doc.publication_date,
                document_type=doc.document_type,
                source=doc.source,
                credibility_score=doc.credibility_score,
                indexed_at=doc.metadata.get('indexed_at', datetime.now())
            )
            for doc in documents
        ]
        
        return response
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@app.post("/documents", response_model=DocumentResponse)
async def upload_document(
    request: DocumentUploadRequest,
    background_tasks: BackgroundTasks,
    service: ClinicalService = Depends(get_clinical_service)
):
    """
    Upload a new medical document to the knowledge base.
    
    Allows administrators to add new medical literature that will be
    processed and indexed for use in clinical recommendations.
    """
    try:
        logger.info(f"Uploading document: {request.title}")
        
        # Process the document upload
        document = await service.upload_document(request)
        
        # Schedule background indexing
        background_tasks.add_task(
            service.index_document,
            document
        )
        
        response = DocumentResponse(
            id=document.id,
            title=document.title,
            authors=document.authors,
            publication_date=document.publication_date,
            document_type=document.document_type,
            source=document.source,
            credibility_score=document.credibility_score,
            indexed_at=datetime.now()
        )
        
        logger.info(f"Successfully uploaded document {document.id}")
        return response
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    service: ClinicalService = Depends(get_clinical_service)
):
    """Get details of a specific document."""
    try:
        document = await service.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        return DocumentResponse(
            id=document.id,
            title=document.title,
            authors=document.authors,
            publication_date=document.publication_date,
            document_type=document.document_type,
            source=document.source,
            credibility_score=document.credibility_score,
            indexed_at=document.metadata.get('indexed_at', datetime.now())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting document: {str(e)}"
        )


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    service: ClinicalService = Depends(get_clinical_service)
):
    """Delete a document from the knowledge base."""
    try:
        success = await service.delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Schedule background cleanup
        background_tasks.add_task(
            service.cleanup_document_references,
            document_id
        )
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@app.get("/recommendations/{query_id}/history", response_model=RecommendationHistoryResponse)
async def get_recommendation_history(
    query_id: str,
    service: ClinicalService = Depends(get_clinical_service)
):
    """
    Get the history of recommendations for a specific query.
    
    Provides tracking of how recommendations change over time as new
    evidence becomes available, supporting Requirements 4.2, 4.3:
    - Explain why recommendations changed
    - Maintain history of previous recommendations with timestamps
    """
    try:
        history = await service.get_recommendation_history(query_id)
        
        if not history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No recommendation history found for query {query_id}"
            )
        
        return RecommendationHistoryResponse(
            query_id=query_id,
            recommendations=history,
            total_changes=len(history) - 1 if len(history) > 1 else 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendation history for {query_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting recommendation history: {str(e)}"
        )


@app.get("/recommendations/recent", response_model=List[ClinicalRecommendation])
async def get_recent_recommendations(
    limit: int = 50,
    clinician_id: Optional[str] = None,
    service: ClinicalService = Depends(get_clinical_service)
):
    """Get recent recommendations, optionally filtered by clinician."""
    try:
        recommendations = await service.get_recent_recommendations(
            limit=limit,
            clinician_id=clinician_id
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting recent recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting recent recommendations: {str(e)}"
        )


@app.websocket("/ws/{clinician_id}")
async def websocket_endpoint(websocket: WebSocket, clinician_id: str):
    """
    WebSocket endpoint for real-time updates.
    
    Provides live recommendation updates and evidence change notifications.
    
    Validates Requirements 1.5, 4.1, 4.4:
    - Notify relevant users when new evidence becomes available
    - Update affected recommendations immediately when new contradictory evidence is ingested
    - Proactively notify clinicians who previously queried related topics
    """
    # Connect to WebSocket manager
    connected = await websocket_manager.connect(websocket, clinician_id)
    
    if not connected:
        await websocket.close(code=1011, reason="Failed to establish connection")
        return
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await websocket_manager.handle_message(clinician_id, message)
            except json.JSONDecodeError:
                await websocket_manager._send_error(clinician_id, "Invalid JSON message")
            except Exception as e:
                logger.error(f"Error handling WebSocket message from {clinician_id}: {e}")
                await websocket_manager._send_error(clinician_id, "Error processing message")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for clinician {clinician_id}")
    except Exception as e:
        logger.error(f"WebSocket error for clinician {clinician_id}: {e}")
    finally:
        await websocket_manager.disconnect(clinician_id)


@app.post("/notifications/subscribe", response_model=dict)
async def subscribe_to_notifications(
    request: NotificationRequest,
    service: ClinicalService = Depends(get_clinical_service)
):
    """
    Subscribe to notifications for specific queries or keywords.
    
    This endpoint allows clinicians to set up notification preferences
    for receiving updates about recommendation changes and new evidence.
    """
    try:
        # In a full implementation, this would store subscription preferences in a database
        # For now, we'll just acknowledge the subscription
        
        logger.info(f"Clinician {request.clinician_id} subscribed to notifications for keywords: {request.query_keywords}")
        
        return {
            "message": "Subscription created successfully",
            "clinician_id": request.clinician_id,
            "keywords": request.query_keywords,
            "notification_types": request.notification_types,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating notification subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating subscription: {str(e)}"
        )


@app.get("/notifications/{clinician_id}", response_model=List[NotificationResponse])
async def get_notifications(
    clinician_id: str,
    limit: int = 50,
    unread_only: bool = False,
    service: ClinicalService = Depends(get_clinical_service)
):
    """Get notifications for a specific clinician."""
    try:
        # In a full implementation, this would retrieve notifications from a database
        # For now, return empty list as notifications are handled via WebSocket
        
        return []
        
    except Exception as e:
        logger.error(f"Error getting notifications for {clinician_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting notifications: {str(e)}"
        )


@app.get("/ws/stats", response_model=dict)
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    try:
        stats = websocket_manager.get_connection_stats()
        return {
            "websocket_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting WebSocket stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting WebSocket stats: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred",
            timestamp=datetime.now()
        ).dict()
    )


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )