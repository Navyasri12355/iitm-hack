"""
WebSocket support for real-time updates in Clinical Evidence Copilot.

This module implements:
- WebSocket connections for live recommendation updates
- Notification system for evidence changes
- User session management

Validates Requirements 1.5, 4.1, 4.4:
- Update previous recommendations and notify relevant users when new evidence becomes available
- Update affected recommendations immediately when new contradictory evidence is ingested
- Proactively notify clinicians who previously queried related topics when significant evidence updates occur
"""

import logging
import json
import asyncio
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from ..models.core import ClinicalRecommendation, ClinicalQuery
from .models import NotificationResponse

logger = logging.getLogger(__name__)


@dataclass
class WebSocketSession:
    """Represents a WebSocket session for a clinician."""
    websocket: WebSocket
    clinician_id: str
    connected_at: datetime
    subscriptions: Set[str]  # Query IDs or keywords the clinician is subscribed to
    last_activity: datetime


@dataclass
class NotificationMessage:
    """Represents a notification message to be sent via WebSocket."""
    notification_id: str
    clinician_id: str
    notification_type: str
    title: str
    message: str
    related_query_id: Optional[str] = None
    timestamp: datetime = None
    data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class WebSocketManager:
    """
    Manages WebSocket connections and real-time notifications.
    
    Handles connection lifecycle, message routing, and notification delivery
    for live recommendation updates and evidence changes.
    """
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        # Active connections: clinician_id -> WebSocketSession
        self.active_connections: Dict[str, WebSocketSession] = {}
        
        # Subscription mappings: query_id -> set of clinician_ids
        self.query_subscriptions: Dict[str, Set[str]] = {}
        
        # Keyword subscriptions: keyword -> set of clinician_ids
        self.keyword_subscriptions: Dict[str, Set[str]] = {}
        
        # Notification queue for offline users
        self.notification_queue: Dict[str, List[NotificationMessage]] = {}
        
        logger.info("WebSocketManager initialized")
    
    async def connect(self, websocket: WebSocket, clinician_id: str) -> bool:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            clinician_id: Identifier of the connecting clinician
            
        Returns:
            True if connection was successful
        """
        try:
            await websocket.accept()
            
            # Create session
            session = WebSocketSession(
                websocket=websocket,
                clinician_id=clinician_id,
                connected_at=datetime.now(),
                subscriptions=set(),
                last_activity=datetime.now()
            )
            
            # Store connection
            self.active_connections[clinician_id] = session
            
            # Send queued notifications
            await self._send_queued_notifications(clinician_id)
            
            # Send connection confirmation
            await self._send_message(clinician_id, {
                "type": "connection_established",
                "message": "Connected to Clinical Evidence Copilot",
                "timestamp": datetime.now().isoformat(),
                "clinician_id": clinician_id
            })
            
            logger.info(f"WebSocket connection established for clinician {clinician_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error establishing WebSocket connection for {clinician_id}: {e}")
            return False
    
    async def disconnect(self, clinician_id: str):
        """
        Handle WebSocket disconnection.
        
        Args:
            clinician_id: Identifier of the disconnecting clinician
        """
        try:
            if clinician_id in self.active_connections:
                session = self.active_connections[clinician_id]
                
                # Clean up subscriptions
                self._cleanup_subscriptions(clinician_id)
                
                # Remove connection
                del self.active_connections[clinician_id]
                
                logger.info(f"WebSocket connection closed for clinician {clinician_id}")
            
        except Exception as e:
            logger.error(f"Error handling disconnect for {clinician_id}: {e}")
    
    async def handle_message(self, clinician_id: str, message: dict):
        """
        Handle incoming WebSocket message from a clinician.
        
        Args:
            clinician_id: Identifier of the sending clinician
            message: Message data
        """
        try:
            message_type = message.get("type")
            
            if message_type == "subscribe_query":
                await self._handle_query_subscription(clinician_id, message)
            elif message_type == "subscribe_keywords":
                await self._handle_keyword_subscription(clinician_id, message)
            elif message_type == "unsubscribe":
                await self._handle_unsubscription(clinician_id, message)
            elif message_type == "ping":
                await self._handle_ping(clinician_id)
            else:
                logger.warning(f"Unknown message type from {clinician_id}: {message_type}")
                await self._send_error(clinician_id, f"Unknown message type: {message_type}")
            
            # Update last activity
            if clinician_id in self.active_connections:
                self.active_connections[clinician_id].last_activity = datetime.now()
            
        except Exception as e:
            logger.error(f"Error handling message from {clinician_id}: {e}")
            await self._send_error(clinician_id, "Error processing message")
    
    async def notify_recommendation_change(
        self, 
        recommendation: ClinicalRecommendation,
        original_query: Optional[ClinicalQuery] = None
    ):
        """
        Notify relevant clinicians about recommendation changes.
        
        Args:
            recommendation: Updated recommendation
            original_query: Original query that generated the recommendation
        """
        try:
            # Find clinicians subscribed to this query
            subscribers = self.query_subscriptions.get(recommendation.query_id, set())
            
            # Find clinicians subscribed to related keywords
            if original_query:
                keyword_subscribers = self._find_keyword_subscribers(original_query.query_text)
                subscribers.update(keyword_subscribers)
            
            if not subscribers:
                logger.debug(f"No subscribers for recommendation change: {recommendation.id}")
                return
            
            # Create notification
            notification = NotificationMessage(
                notification_id=f"rec_change_{recommendation.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                clinician_id="",  # Will be set per recipient
                notification_type="recommendation_change",
                title="Recommendation Updated",
                message=f"A recommendation has been updated due to new evidence: {recommendation.change_reason}",
                related_query_id=recommendation.query_id,
                data={
                    "recommendation_id": recommendation.id,
                    "confidence_score": recommendation.confidence_score,
                    "change_reason": recommendation.change_reason,
                    "evidence_count": len(recommendation.supporting_evidence),
                    "contradiction_count": len(recommendation.contradictions)
                }
            )
            
            # Send to all subscribers
            for clinician_id in subscribers:
                notification.clinician_id = clinician_id
                await self._send_notification(notification)
            
            logger.info(f"Sent recommendation change notification to {len(subscribers)} clinicians")
            
        except Exception as e:
            logger.error(f"Error sending recommendation change notification: {e}")
    
    async def notify_new_evidence(
        self, 
        document_title: str, 
        affected_queries: List[str],
        evidence_level: str
    ):
        """
        Notify clinicians about new evidence that affects their queries.
        
        Args:
            document_title: Title of the new document
            affected_queries: List of query IDs affected by the new evidence
            evidence_level: Level of the new evidence
        """
        try:
            all_subscribers = set()
            
            # Collect subscribers from affected queries
            for query_id in affected_queries:
                subscribers = self.query_subscriptions.get(query_id, set())
                all_subscribers.update(subscribers)
            
            if not all_subscribers:
                logger.debug(f"No subscribers for new evidence: {document_title}")
                return
            
            # Create notification
            notification = NotificationMessage(
                notification_id=f"new_evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                clinician_id="",  # Will be set per recipient
                notification_type="new_evidence",
                title="New Evidence Available",
                message=f"New {evidence_level.replace('_', ' ')} evidence added: {document_title}",
                data={
                    "document_title": document_title,
                    "evidence_level": evidence_level,
                    "affected_queries": affected_queries,
                    "query_count": len(affected_queries)
                }
            )
            
            # Send to all subscribers
            for clinician_id in all_subscribers:
                notification.clinician_id = clinician_id
                await self._send_notification(notification)
            
            logger.info(f"Sent new evidence notification to {len(all_subscribers)} clinicians")
            
        except Exception as e:
            logger.error(f"Error sending new evidence notification: {e}")
    
    async def notify_system_status(self, status: str, message: str):
        """
        Notify all connected clinicians about system status changes.
        
        Args:
            status: System status (e.g., "maintenance", "degraded", "operational")
            message: Status message
        """
        try:
            notification = NotificationMessage(
                notification_id=f"system_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                clinician_id="",  # Will be set per recipient
                notification_type="system_status",
                title=f"System Status: {status.title()}",
                message=message,
                data={"status": status}
            )
            
            # Send to all connected clinicians
            for clinician_id in self.active_connections.keys():
                notification.clinician_id = clinician_id
                await self._send_notification(notification)
            
            logger.info(f"Sent system status notification to {len(self.active_connections)} clinicians")
            
        except Exception as e:
            logger.error(f"Error sending system status notification: {e}")
    
    async def _handle_query_subscription(self, clinician_id: str, message: dict):
        """Handle query subscription request."""
        query_id = message.get("query_id")
        if not query_id:
            await self._send_error(clinician_id, "Missing query_id in subscription request")
            return
        
        # Add to subscriptions
        if query_id not in self.query_subscriptions:
            self.query_subscriptions[query_id] = set()
        self.query_subscriptions[query_id].add(clinician_id)
        
        # Add to session subscriptions
        if clinician_id in self.active_connections:
            self.active_connections[clinician_id].subscriptions.add(query_id)
        
        await self._send_message(clinician_id, {
            "type": "subscription_confirmed",
            "subscription_type": "query",
            "query_id": query_id,
            "message": f"Subscribed to updates for query {query_id}"
        })
        
        logger.info(f"Clinician {clinician_id} subscribed to query {query_id}")
    
    async def _handle_keyword_subscription(self, clinician_id: str, message: dict):
        """Handle keyword subscription request."""
        keywords = message.get("keywords", [])
        if not keywords:
            await self._send_error(clinician_id, "Missing keywords in subscription request")
            return
        
        # Add to keyword subscriptions
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self.keyword_subscriptions:
                self.keyword_subscriptions[keyword_lower] = set()
            self.keyword_subscriptions[keyword_lower].add(clinician_id)
        
        await self._send_message(clinician_id, {
            "type": "subscription_confirmed",
            "subscription_type": "keywords",
            "keywords": keywords,
            "message": f"Subscribed to updates for keywords: {', '.join(keywords)}"
        })
        
        logger.info(f"Clinician {clinician_id} subscribed to keywords: {keywords}")
    
    async def _handle_unsubscription(self, clinician_id: str, message: dict):
        """Handle unsubscription request."""
        subscription_type = message.get("subscription_type")
        
        if subscription_type == "query":
            query_id = message.get("query_id")
            if query_id and query_id in self.query_subscriptions:
                self.query_subscriptions[query_id].discard(clinician_id)
                if not self.query_subscriptions[query_id]:
                    del self.query_subscriptions[query_id]
        
        elif subscription_type == "keywords":
            keywords = message.get("keywords", [])
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in self.keyword_subscriptions:
                    self.keyword_subscriptions[keyword_lower].discard(clinician_id)
                    if not self.keyword_subscriptions[keyword_lower]:
                        del self.keyword_subscriptions[keyword_lower]
        
        await self._send_message(clinician_id, {
            "type": "unsubscription_confirmed",
            "subscription_type": subscription_type,
            "message": "Unsubscribed successfully"
        })
    
    async def _handle_ping(self, clinician_id: str):
        """Handle ping message."""
        await self._send_message(clinician_id, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    
    async def _send_notification(self, notification: NotificationMessage):
        """Send a notification to a specific clinician."""
        clinician_id = notification.clinician_id
        
        # Try to send to active connection
        if clinician_id in self.active_connections:
            message = {
                "type": "notification",
                "notification": asdict(notification)
            }
            await self._send_message(clinician_id, message)
        else:
            # Queue for offline user
            if clinician_id not in self.notification_queue:
                self.notification_queue[clinician_id] = []
            self.notification_queue[clinician_id].append(notification)
            
            # Limit queue size
            if len(self.notification_queue[clinician_id]) > 100:
                self.notification_queue[clinician_id] = self.notification_queue[clinician_id][-100:]
    
    async def _send_message(self, clinician_id: str, message: dict):
        """Send a message to a specific clinician."""
        if clinician_id not in self.active_connections:
            return
        
        session = self.active_connections[clinician_id]
        
        try:
            if session.websocket.client_state == WebSocketState.CONNECTED:
                await session.websocket.send_text(json.dumps(message, default=str))
            else:
                # Connection is not active, remove it
                await self.disconnect(clinician_id)
        except Exception as e:
            logger.error(f"Error sending message to {clinician_id}: {e}")
            await self.disconnect(clinician_id)
    
    async def _send_error(self, clinician_id: str, error_message: str):
        """Send an error message to a clinician."""
        await self._send_message(clinician_id, {
            "type": "error",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _send_queued_notifications(self, clinician_id: str):
        """Send queued notifications to a newly connected clinician."""
        if clinician_id not in self.notification_queue:
            return
        
        notifications = self.notification_queue[clinician_id]
        
        for notification in notifications:
            await self._send_notification(notification)
        
        # Clear the queue
        del self.notification_queue[clinician_id]
        
        logger.info(f"Sent {len(notifications)} queued notifications to {clinician_id}")
    
    def _cleanup_subscriptions(self, clinician_id: str):
        """Clean up subscriptions for a disconnected clinician."""
        # Remove from query subscriptions
        for query_id, subscribers in list(self.query_subscriptions.items()):
            subscribers.discard(clinician_id)
            if not subscribers:
                del self.query_subscriptions[query_id]
        
        # Remove from keyword subscriptions
        for keyword, subscribers in list(self.keyword_subscriptions.items()):
            subscribers.discard(clinician_id)
            if not subscribers:
                del self.keyword_subscriptions[keyword]
    
    def _find_keyword_subscribers(self, query_text: str) -> Set[str]:
        """Find clinicians subscribed to keywords that match the query text."""
        subscribers = set()
        query_lower = query_text.lower()
        
        for keyword, clinician_ids in self.keyword_subscriptions.items():
            if keyword in query_lower:
                subscribers.update(clinician_ids)
        
        return subscribers
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about WebSocket connections."""
        return {
            "active_connections": len(self.active_connections),
            "query_subscriptions": len(self.query_subscriptions),
            "keyword_subscriptions": len(self.keyword_subscriptions),
            "queued_notifications": sum(len(queue) for queue in self.notification_queue.values()),
            "total_subscribers": len(set().union(*self.query_subscriptions.values(), *self.keyword_subscriptions.values())) if self.query_subscriptions or self.keyword_subscriptions else 0
        }


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


# Export the manager and related classes
__all__ = [
    'WebSocketManager',
    'WebSocketSession', 
    'NotificationMessage',
    'websocket_manager'
]