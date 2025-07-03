"""
Asynchronous event-driven architecture for ZKPAS protocol.

This module implements a publish-subscribe event system that decouples components
and enables realistic simulation of distributed authentication protocols.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID, uuid4

from loguru import logger


class EventType(Enum):
    """Types of events in the ZKPAS protocol."""
    
    # Authentication flow events
    AUTH_REQUEST = auto()
    COMMITMENT_GENERATED = auto()
    CHALLENGE_CREATED = auto()
    ZKP_COMPUTED = auto()
    VERIFICATION_COMPLETE = auto()
    SESSION_ESTABLISHED = auto()
    SESSION_EXPIRED = auto()
    
    # Network events
    NETWORK_FAILURE = auto()
    NETWORK_RESTORED = auto()
    MESSAGE_TRANSMITTED = auto()
    MESSAGE_RECEIVED = auto()
    
    # Mobility events
    LOCATION_CHANGED = auto()
    MOBILITY_PREDICTED = auto()
    HANDOFF_INITIATED = auto()
    HANDOFF_COMPLETE = auto()
    
    # Error events
    CRYPTO_FAILURE = auto()
    TIMEOUT_EXPIRED = auto()
    INVALID_PROOF = auto()
    PROTOCOL_VIOLATION = auto()
    
    # System events
    COMPONENT_STARTED = auto()
    COMPONENT_STOPPED = auto()
    DEGRADED_MODE_ENTERED = auto()
    DEGRADED_MODE_EXITED = auto()
    
    # Device events
    DEVICE_REGISTERED = auto()
    DEVICE_DEREGISTERED = auto()
    DEVICE_AUTHENTICATED = auto()
    DEVICE_AUTHENTICATION_FAILED = auto()
    
    # Gateway events
    GATEWAY_REGISTERED = auto()
    GATEWAY_DEREGISTERED = auto()
    GATEWAY_AVAILABLE = auto()
    GATEWAY_UNAVAILABLE = auto()
    
    # State machine events
    STATE_TRANSITION = auto()
    STATE_MACHINE_ERROR = auto()
    INVALID_TRANSITION = auto()
    TIMEOUT_STATE_CHANGE = auto()
    
    # Phase 5: Sliding Window Authentication events
    WINDOW_CREATED = auto()
    WINDOW_EXPIRED = auto()
    TOKEN_GENERATED = auto()
    TOKEN_VALIDATED = auto()
    TOKEN_VALIDATION_REQUEST = auto()
    TOKEN_VALIDATION_RESPONSE = auto()
    TOKEN_EXPIRED = auto()
    FALLBACK_MODE_ENABLED = auto()
    FALLBACK_MODE_DISABLED = auto()
    
    # Phase 5: Byzantine Fault Tolerance events
    CROSS_DOMAIN_AUTH_REQUEST = auto()
    CROSS_DOMAIN_AUTH_SUCCESS = auto()
    CROSS_DOMAIN_AUTH_FAILURE = auto()
    SIGNATURE_SHARE_GENERATED = auto()
    SIGNATURE_SHARE_VALIDATED = auto()
    THRESHOLD_SIGNATURE_CREATED = auto()
    BYZANTINE_FAULT_DETECTED = auto()
    TRUST_ANCHOR_COMPROMISED = auto()
    NETWORK_PARTITION_DETECTED = auto()
    CONSENSUS_REACHED = auto()
    CONSENSUS_FAILED = auto()


@dataclass
class Event:
    """Represents an event in the ZKPAS system."""
    
    event_type: EventType
    correlation_id: UUID
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    target: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.name,
            "correlation_id": str(self.correlation_id),
            "timestamp": self.timestamp,
            "source": self.source,
            "target": self.target,
            "data": self.data,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            event_type=EventType[data["event_type"]],
            correlation_id=UUID(data["correlation_id"]),
            timestamp=data["timestamp"],
            source=data["source"],
            target=data.get("target"),
            data=data.get("data", {}),
            metadata=data.get("metadata", {})
        )


EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], asyncio.Task]


class EventBus:
    """Asynchronous event bus for ZKPAS components."""
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self._subscribers: Dict[EventType, Set[AsyncEventHandler]] = {}
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._metrics = {
            "events_published": 0,
            "events_processed": 0,
            "events_dropped": 0,
            "handler_errors": 0
        }
    
    async def start(self) -> None:
        """Start the event bus processor."""
        if self._running:
            logger.warning("Event bus already running")
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop the event bus processor."""
        if not self._running:
            return
        
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining events
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                await self._handle_event(event)
            except asyncio.QueueEmpty:
                break
        
        logger.info("Event bus stopped")
    
    def subscribe_sync(self, event_type: EventType, handler: AsyncEventHandler) -> None:
        """Subscribe to events of a specific type (synchronous version)."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        
        self._subscribers[event_type].add(handler)
        logger.debug(f"Subscribed handler to {event_type.name}")
    
    def unsubscribe(self, event_type: EventType, handler: AsyncEventHandler) -> None:
        """Unsubscribe from events of a specific type."""
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(handler)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]
        
        logger.debug(f"Unsubscribed handler from {event_type.name}")
    
    # Add an async version of subscribe for validation compatibility
    async def subscribe(self, event_type: EventType, handler: AsyncEventHandler) -> None:
        """Async version of subscribe (alias for sync version)."""
        # Call the sync subscribe method
        self.subscribe_sync(event_type, handler)
    
    async def publish(self, event: Event) -> None:
        """Publish an event to the bus."""
        try:
            await self._event_queue.put(event)
            self._metrics["events_published"] += 1
            
            logger.debug(
                f"Published event {event.event_type.name}",
                extra={
                    "correlation_id": str(event.correlation_id),
                    "source": event.source,
                    "target": event.target
                }
            )
        except asyncio.QueueFull:
            self._metrics["events_dropped"] += 1
            logger.warning(
                f"Event queue full, dropped {event.event_type.name}",
                extra={"correlation_id": str(event.correlation_id)}
            )
    
    async def publish_event(
        self,
        event_type: EventType,
        correlation_id: UUID,
        source: str,
        target: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Convenience method to publish an event."""
        event = Event(
            event_type=event_type,
            correlation_id=correlation_id,
            source=source,
            target=target,
            data=data or {},
            metadata=metadata or {}
        )
        await self.publish(event)
    
    async def _process_events(self) -> None:
        """Main event processing loop."""
        while self._running:
            try:
                # Wait for an event with timeout to check running status
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                await self._handle_event(event)
                self._metrics["events_processed"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, event: Event) -> None:
        """Handle a single event by calling all subscribers."""
        if event.event_type not in self._subscribers:
            return
        
        # Create tasks for all handlers
        tasks = []
        for handler in self._subscribers[event.event_type]:
            try:
                task = asyncio.create_task(handler(event))
                tasks.append(task)
            except Exception as e:
                self._metrics["handler_errors"] += 1
                logger.error(
                    f"Error creating task for handler: {e}",
                    extra={"correlation_id": str(event.correlation_id)}
                )
        
        # Wait for all handlers to complete
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                self._metrics["handler_errors"] += 1
                logger.error(
                    f"Error in event handler: {e}",
                    extra={"correlation_id": str(event.correlation_id)}
                )
    
    def get_metrics(self) -> Dict[str, int]:
        """Get event bus metrics."""
        return self._metrics.copy()
    
    def clear_metrics(self) -> None:
        """Clear event bus metrics."""
        self._metrics = {
            "events_published": 0,
            "events_processed": 0,
            "events_dropped": 0,
            "handler_errors": 0
        }


class EventLogger:
    """Logs events for audit and debugging purposes."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self._event_history: List[Event] = []
        self._max_history = 10000
    
    async def log_event(self, event: Event) -> None:
        """Log an event."""
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Log with structured data
        logger.info(
            f"Event: {event.event_type.name}",
            extra={
                "event_type": event.event_type.name,
                "correlation_id": str(event.correlation_id),
                "timestamp": event.timestamp,
                "source": event.source,
                "target": event.target,
                "data": event.data,
                "metadata": event.metadata
            }
        )
        
        # Write to file if specified
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")
            except Exception as e:
                logger.error(f"Failed to write event to file: {e}")
    
    def get_events_by_correlation_id(self, correlation_id: UUID) -> List[Event]:
        """Get all events for a specific correlation ID."""
        return [e for e in self._event_history if e.correlation_id == correlation_id]
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get all events of a specific type."""
        return [e for e in self._event_history if e.event_type == event_type]
    
    def get_events_in_timeframe(self, start_time: float, end_time: float) -> List[Event]:
        """Get all events within a time frame."""
        return [
            e for e in self._event_history 
            if start_time <= e.timestamp <= end_time
        ]


class CorrelationManager:
    """Manages correlation IDs for tracking related events."""
    
    def __init__(self):
        self._active_correlations: Dict[UUID, Dict[str, Any]] = {}
    
    def create_correlation(
        self,
        context: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """Create a new correlation ID."""
        correlation_id = uuid4()
        self._active_correlations[correlation_id] = {
            "context": context,
            "created_at": time.time(),
            "metadata": metadata or {}
        }
        
        logger.debug(
            f"Created correlation {correlation_id} for {context}",
            extra={"correlation_id": str(correlation_id)}
        )
        
        return correlation_id
    
    def get_correlation_info(self, correlation_id: UUID) -> Optional[Dict[str, Any]]:
        """Get information about a correlation ID."""
        return self._active_correlations.get(correlation_id)
    
    def close_correlation(self, correlation_id: UUID) -> None:
        """Close a correlation ID."""
        if correlation_id in self._active_correlations:
            context = self._active_correlations[correlation_id]["context"]
            del self._active_correlations[correlation_id]
            
            logger.debug(
                f"Closed correlation {correlation_id} for {context}",
                extra={"correlation_id": str(correlation_id)}
            )
    
    def cleanup_old_correlations(self, max_age_seconds: int = 3600) -> None:
        """Clean up old correlation IDs."""
        current_time = time.time()
        expired_ids = [
            cid for cid, info in self._active_correlations.items()
            if current_time - info["created_at"] > max_age_seconds
        ]
        
        for cid in expired_ids:
            self.close_correlation(cid)
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired correlations")


# Global instances
event_bus = EventBus()
event_logger = EventLogger()
correlation_manager = CorrelationManager()
