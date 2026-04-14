#!/usr/bin/env python3
"""
Mock API server for testing the Clinical Evidence Copilot frontend.
Provides mock responses for all API endpoints.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import json
import random
import uvicorn
from typing import List, Dict, Any
import asyncio

app = FastAPI(title="Mock Clinical Evidence Copilot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data
mock_documents = [
    {
        "id": "doc_001",
        "title": "Efficacy of ACE Inhibitors in Elderly Hypertensive Patients",
        "authors": ["Dr. Jane Smith", "Dr. Robert Johnson"],
        "publication_date": "2024-01-15T00:00:00",
        "document_type": "systematic_review",
        "source": "Journal of Hypertension",
        "credibility_score": 0.92,
        "indexed_at": "2024-12-28T14:30:00"
    },
    {
        "id": "doc_002", 
        "title": "Diabetes Management Guidelines 2024",
        "authors": ["Dr. Maria Garcia", "Dr. Ahmed Hassan"],
        "publication_date": "2024-03-20T00:00:00",
        "document_type": "guideline",
        "source": "American Diabetes Association",
        "credibility_score": 0.95,
        "indexed_at": "2024-12-28T15:00:00"
    },
    {
        "id": "doc_003",
        "title": "Cardiovascular Risk Assessment in Primary Care",
        "authors": ["Dr. Lisa Chen", "Dr. Michael Brown"],
        "publication_date": "2024-02-10T00:00:00",
        "document_type": "research_paper",
        "source": "New England Journal of Medicine",
        "credibility_score": 0.88,
        "indexed_at": "2024-12-28T16:00:00"
    }
]

mock_recommendations = [
    {
        "id": "rec_001",
        "query_id": "query_001",
        "recommendation_text": "Based on current evidence, ACE inhibitors are recommended as first-line treatment for hypertension in elderly patients. Start with low doses and monitor kidney function closely.",
        "confidence_score": 0.85,
        "last_updated": "2024-12-28T14:30:25",
        "change_reason": None,
        "supporting_evidence": [],
        "contradictions": []
    },
    {
        "id": "rec_002",
        "query_id": "query_002", 
        "recommendation_text": "For type 2 diabetes management, metformin remains the first-line therapy. Consider SGLT-2 inhibitors for patients with cardiovascular disease.",
        "confidence_score": 0.92,
        "last_updated": "2024-12-28T15:45:10",
        "change_reason": "Updated based on new cardiovascular outcome studies",
        "supporting_evidence": [],
        "contradictions": []
    }
]

# WebSocket connections
websocket_connections: Dict[str, WebSocket] = {}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0-mock"
    }

@app.post("/query")
async def process_clinical_query(request: dict):
    # Simulate processing time
    await asyncio.sleep(random.uniform(1, 3))
    
    query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.get('clinician_id', 'unknown')}"
    
    # Generate mock recommendation based on query
    query_text = request.get('query_text', '').lower()
    
    if 'hypertension' in query_text or 'blood pressure' in query_text:
        recommendation_text = "Based on current evidence, ACE inhibitors are recommended as first-line treatment for hypertension in elderly patients. Start with low doses (e.g., lisinopril 5mg daily) and monitor kidney function closely. Consider adding a thiazide diuretic if blood pressure remains elevated after 4-6 weeks."
        confidence = 0.85
    elif 'diabetes' in query_text:
        recommendation_text = "For type 2 diabetes management, metformin remains the first-line therapy unless contraindicated. Target HbA1c should be <7% for most adults. Consider SGLT-2 inhibitors for patients with established cardiovascular disease or heart failure."
        confidence = 0.92
    elif 'cardio' in query_text or 'heart' in query_text:
        recommendation_text = "Cardiovascular risk assessment should include traditional risk factors (age, gender, smoking, diabetes, hypertension, cholesterol) and consider using validated risk calculators like ASCVD Risk Calculator. Statin therapy is recommended for high-risk patients."
        confidence = 0.78
    else:
        recommendation_text = "Based on available evidence, a comprehensive clinical assessment is recommended. Please consult current clinical guidelines and consider patient-specific factors including comorbidities, contraindications, and patient preferences."
        confidence = 0.65
    
    recommendation = {
        "id": f"rec_{query_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "query_id": query_id,
        "recommendation_text": recommendation_text,
        "confidence_score": confidence,
        "last_updated": datetime.now().isoformat(),
        "change_reason": None,
        "supporting_evidence": [],
        "contradictions": []
    }
    
    # Mock citations
    citations = [
        {
            "citation_number": 1,
            "title": "Clinical Guidelines for " + query_text.title(),
            "authors": ["Dr. Expert", "Dr. Researcher"],
            "journal": "Medical Journal",
            "publication_date": "2024",
            "evidence_level": "Systematic Review",
            "relevance_score": 0.92,
            "doi": "10.1234/example.2024.001"
        },
        {
            "citation_number": 2,
            "title": "Recent Advances in Treatment",
            "authors": ["Dr. Specialist"],
            "journal": "Clinical Medicine",
            "publication_date": "2024",
            "evidence_level": "Randomized Controlled Trial",
            "relevance_score": 0.87
        }
    ]
    
    # Mock reasoning steps
    reasoning_steps = [
        {
            "step_number": 1,
            "step_type": "Query Analysis",
            "description": "Analyzed clinical query for key medical concepts",
            "reasoning": f"Identified primary focus on {query_text} with clinical context",
            "confidence": 0.95
        },
        {
            "step_number": 2,
            "step_type": "Evidence Search",
            "description": "Searched medical literature database",
            "reasoning": "Found relevant systematic reviews and clinical trials",
            "confidence": 0.88
        },
        {
            "step_number": 3,
            "step_type": "Evidence Ranking",
            "description": "Ranked evidence by quality and relevance",
            "reasoning": "Prioritized systematic reviews and high-quality RCTs",
            "confidence": 0.90
        },
        {
            "step_number": 4,
            "step_type": "Recommendation Synthesis",
            "description": "Synthesized evidence into clinical recommendation",
            "reasoning": "Combined evidence with clinical best practices",
            "confidence": confidence
        }
    ]
    
    return {
        "query_id": query_id,
        "recommendation": recommendation,
        "processing_time_seconds": random.uniform(1.5, 3.2),
        "citations": citations,
        "reasoning_steps": reasoning_steps
    }

@app.get("/documents")
async def list_documents(limit: int = 100, offset: int = 0, document_type: str = None):
    documents = mock_documents.copy()
    
    if document_type:
        documents = [doc for doc in documents if doc["document_type"] == document_type]
    
    return documents[offset:offset+limit]

@app.post("/documents")
async def upload_document(request: dict):
    # Simulate document processing
    await asyncio.sleep(1)
    
    doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(100, 999)}"
    
    document = {
        "id": doc_id,
        "title": request.get("title", "Untitled Document"),
        "authors": request.get("authors", []),
        "publication_date": request.get("publication_date", datetime.now().isoformat()),
        "document_type": request.get("document_type", "research_paper"),
        "source": request.get("source", "Unknown Source"),
        "credibility_score": random.uniform(0.7, 0.95),
        "indexed_at": datetime.now().isoformat()
    }
    
    # Add to mock documents
    mock_documents.append(document)
    
    return document

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    document = next((doc for doc in mock_documents if doc["id"] == document_id), None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    global mock_documents
    original_length = len(mock_documents)
    mock_documents = [doc for doc in mock_documents if doc["id"] != document_id]
    
    if len(mock_documents) == original_length:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": f"Document {document_id} deleted successfully"}

@app.get("/recommendations/recent")
async def get_recent_recommendations(limit: int = 50, clinician_id: str = None):
    recommendations = mock_recommendations.copy()
    
    # Add some random recent recommendations
    for i in range(3):
        rec = {
            "id": f"rec_recent_{i}",
            "query_id": f"query_recent_{i}",
            "recommendation_text": f"Recent clinical recommendation #{i+1} based on latest evidence...",
            "confidence_score": random.uniform(0.7, 0.95),
            "last_updated": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
            "change_reason": "Updated with new evidence" if random.random() > 0.5 else None,
            "supporting_evidence": [],
            "contradictions": []
        }
        recommendations.append(rec)
    
    return recommendations[:limit]

@app.get("/recommendations/{query_id}/history")
async def get_recommendation_history(query_id: str):
    # Mock recommendation history
    history = [
        {
            "id": f"rec_{query_id}_v1",
            "query_id": query_id,
            "recommendation_text": "Initial recommendation based on available evidence at the time...",
            "confidence_score": 0.75,
            "last_updated": (datetime.now() - timedelta(days=7)).isoformat(),
            "change_reason": None,
            "supporting_evidence": [],
            "contradictions": []
        },
        {
            "id": f"rec_{query_id}_v2",
            "query_id": query_id,
            "recommendation_text": "Updated recommendation incorporating new systematic review findings...",
            "confidence_score": 0.85,
            "last_updated": (datetime.now() - timedelta(days=2)).isoformat(),
            "change_reason": "New systematic review added to knowledge base",
            "supporting_evidence": [],
            "contradictions": []
        }
    ]
    
    return {
        "query_id": query_id,
        "recommendations": history,
        "total_changes": len(history) - 1
    }

@app.websocket("/ws/{clinician_id}")
async def websocket_endpoint(websocket: WebSocket, clinician_id: str):
    await websocket.accept()
    websocket_connections[clinician_id] = websocket
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": f"Connected to real-time updates for {clinician_id}",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Simulate periodic updates
        while True:
            await asyncio.sleep(30)  # Send update every 30 seconds
            
            # Send mock update
            update_types = ["new_evidence", "recommendation_update", "notification"]
            update_type = random.choice(update_types)
            
            if update_type == "new_evidence":
                message = "New research paper added to knowledge base"
            elif update_type == "recommendation_update":
                message = "Recommendation updated based on new evidence"
            else:
                message = "System notification: All services running normally"
            
            await websocket.send_text(json.dumps({
                "type": update_type,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "level": "info"
            }))
            
    except WebSocketDisconnect:
        if clinician_id in websocket_connections:
            del websocket_connections[clinician_id]
    except Exception as e:
        print(f"WebSocket error for {clinician_id}: {e}")
        if clinician_id in websocket_connections:
            del websocket_connections[clinician_id]

@app.post("/notifications/subscribe")
async def subscribe_to_notifications(request: dict):
    return {
        "message": "Subscription created successfully",
        "clinician_id": request.get("clinician_id"),
        "keywords": request.get("query_keywords", []),
        "notification_types": request.get("notification_types", []),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/notifications/{clinician_id}")
async def get_notifications(clinician_id: str, limit: int = 50, unread_only: bool = False):
    # Return empty list as notifications are handled via WebSocket in this mock
    return []

@app.get("/ws/stats")
async def get_websocket_stats():
    return {
        "websocket_stats": {
            "active_connections": len(websocket_connections),
            "connected_clinicians": list(websocket_connections.keys())
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)