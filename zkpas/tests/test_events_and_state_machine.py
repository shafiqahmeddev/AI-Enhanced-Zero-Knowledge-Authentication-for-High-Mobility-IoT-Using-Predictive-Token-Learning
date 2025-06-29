"""
Tests for event-driven architecture and state machine implementation.

This module tests the async event system, formal state machines, and
their integration with the ZKPAS protocol components.
"""

import asyncio
import pytest
import time
from uuid import uuid4

from app.events import Event, EventBus, EventType, EventLogger, CorrelationManager
from app.state_machine import GatewayStateMachine, DeviceStateMachine, StateType
from app.mobility_predictor import MobilityPredictor, LocationPoint


class TestEventBus:
    """Test the async event bus implementation."""
    
    @pytest.fixture
    async def event_bus(self):
        """Create and start an event bus for testing."""
        bus = EventBus(max_queue_size=100)
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_event_publishing_and_processing(self, event_bus):
        """Test basic event publishing and processing."""
        events_received = []
        
        async def event_handler(event: Event) -> None:
            events_received.append(event)
        
        # Subscribe to events
        event_bus.subscribe(EventType.AUTH_REQUEST, event_handler)
        
        # Publish event
        correlation_id = uuid4()
        await event_bus.publish_event(
            event_type=EventType.AUTH_REQUEST,
            correlation_id=correlation_id,
            source="test_component",
            data={"device_id": "test_device"}
        )
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify event was received
        assert len(events_received) == 1
        event = events_received[0]
        assert event.event_type == EventType.AUTH_REQUEST
        assert event.correlation_id == correlation_id
        assert event.source == "test_component"
        assert event.data["device_id"] == "test_device"
    
    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_bus):
        """Test that multiple subscribers receive the same event."""
        events_received_1 = []
        events_received_2 = []
        
        async def handler_1(event: Event) -> None:
            events_received_1.append(event)
        
        async def handler_2(event: Event) -> None:
            events_received_2.append(event)
        
        # Subscribe both handlers
        event_bus.subscribe(EventType.COMMITMENT_GENERATED, handler_1)
        event_bus.subscribe(EventType.COMMITMENT_GENERATED, handler_2)
        
        # Publish event
        correlation_id = uuid4()
        await event_bus.publish_event(
            event_type=EventType.COMMITMENT_GENERATED,
            correlation_id=correlation_id,
            source="test_device",
            data={"commitment": "test_commitment"}
        )
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Both handlers should receive the event
        assert len(events_received_1) == 1
        assert len(events_received_2) == 1
        assert events_received_1[0].correlation_id == correlation_id
        assert events_received_2[0].correlation_id == correlation_id
    
    @pytest.mark.asyncio
    async def test_event_bus_metrics(self, event_bus):
        """Test event bus metrics collection."""
        async def dummy_handler(event: Event) -> None:
            pass
        
        event_bus.subscribe(EventType.AUTH_REQUEST, dummy_handler)
        
        # Clear metrics
        event_bus.clear_metrics()
        
        # Publish some events
        for i in range(5):
            await event_bus.publish_event(
                event_type=EventType.AUTH_REQUEST,
                correlation_id=uuid4(),
                source="test",
                data={"index": i}
            )
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        metrics = event_bus.get_metrics()
        assert metrics["events_published"] == 5
        assert metrics["events_processed"] == 5
        assert metrics["events_dropped"] == 0
    
    @pytest.mark.asyncio
    async def test_queue_overflow(self):
        """Test behavior when event queue overflows."""
        small_bus = EventBus(max_queue_size=2)
        await small_bus.start()
        
        try:
            # Fill the queue beyond capacity
            for i in range(5):
                await small_bus.publish_event(
                    event_type=EventType.AUTH_REQUEST,
                    correlation_id=uuid4(),
                    source="test",
                    data={"index": i}
                )
            
            await asyncio.sleep(0.1)
            
            metrics = small_bus.get_metrics()
            assert metrics["events_dropped"] > 0
        
        finally:
            await small_bus.stop()


class TestStateMachine:
    """Test the formal state machine implementation."""
    
    @pytest.fixture
    async def event_bus(self):
        """Create event bus for state machine testing."""
        bus = EventBus()
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_gateway_state_machine_initialization(self, event_bus):
        """Test gateway state machine initialization."""
        gateway_sm = GatewayStateMachine("test_gateway", event_bus)
        
        assert gateway_sm.component_id == "test_gateway"
        assert gateway_sm.current_state == StateType.IDLE
        assert gateway_sm.previous_state is None
        assert len(gateway_sm.transitions) > 0
    
    @pytest.mark.asyncio
    async def test_gateway_auth_flow_transitions(self, event_bus):
        """Test gateway authentication flow state transitions."""
        gateway_sm = GatewayStateMachine("test_gateway", event_bus)
        
        # Start authentication
        auth_event = Event(
            event_type=EventType.AUTH_REQUEST,
            correlation_id=uuid4(),
            source="test_device",
            data={"device_id": "test_device"}
        )
        
        await gateway_sm.handle_event(auth_event)
        assert gateway_sm.current_state == StateType.AWAITING_COMMITMENT
        
        # Receive commitment
        commitment_event = Event(
            event_type=EventType.COMMITMENT_GENERATED,
            correlation_id=auth_event.correlation_id,
            source="test_device",
            data={"commitment": "test_commitment"}
        )
        
        await gateway_sm.handle_event(commitment_event)
        assert gateway_sm.current_state == StateType.AWAITING_RESPONSE
        
        # Valid ZKP response
        zkp_event = Event(
            event_type=EventType.VERIFICATION_COMPLETE,
            correlation_id=auth_event.correlation_id,
            source="test_device",
            data={"zkp": {"valid": True}}
        )
        
        await gateway_sm.handle_event(zkp_event)
        assert gateway_sm.current_state == StateType.AUTHENTICATED
    
    @pytest.mark.asyncio
    async def test_device_state_machine_initialization(self, event_bus):
        """Test device state machine initialization."""
        device_sm = DeviceStateMachine("test_device", event_bus)
        
        assert device_sm.component_id == "test_device"
        assert device_sm.current_state == StateType.IDLE
        assert device_sm.previous_state is None
    
    @pytest.mark.asyncio
    async def test_state_machine_timeout_handling(self, event_bus):
        """Test state machine timeout handling."""
        gateway_sm = GatewayStateMachine("test_gateway", event_bus)
        
        # Set very short timeout for testing
        gateway_sm.state_timeouts[StateType.AWAITING_COMMITMENT] = 0.1
        
        # Start authentication
        auth_event = Event(
            event_type=EventType.AUTH_REQUEST,
            correlation_id=uuid4(),
            source="test_device",
            data={"device_id": "test_device"}
        )
        
        await gateway_sm.handle_event(auth_event)
        assert gateway_sm.current_state == StateType.AWAITING_COMMITMENT
        
        # Wait for timeout
        await asyncio.sleep(0.2)
        
        # Should transition back to IDLE due to timeout
        assert gateway_sm.current_state == StateType.IDLE
    
    @pytest.mark.asyncio
    async def test_state_machine_error_handling(self, event_bus):
        """Test state machine error handling."""
        gateway_sm = GatewayStateMachine("test_gateway", event_bus)
        
        # Force error state
        await gateway_sm._transition_to_error("Test error")
        assert gateway_sm.current_state == StateType.ERROR
        assert gateway_sm.state_data["error_message"] == "Test error"
    
    @pytest.mark.asyncio
    async def test_state_machine_info_retrieval(self, event_bus):
        """Test state machine information retrieval."""
        gateway_sm = GatewayStateMachine("test_gateway", event_bus)
        
        # Get initial state info
        info = gateway_sm.get_state_info()
        assert info["component_id"] == "test_gateway"
        assert info["current_state"] == "IDLE"
        assert info["previous_state"] is None
        assert "time_in_state" in info
        
        # Check state queries
        assert gateway_sm.is_in_state(StateType.IDLE)
        assert not gateway_sm.is_in_state(StateType.AUTHENTICATED)
        assert gateway_sm.can_handle_event(EventType.AUTH_REQUEST)
        assert not gateway_sm.can_handle_event(EventType.VERIFICATION_COMPLETE)


class TestEventLogger:
    """Test the event logging functionality."""
    
    def test_event_logging(self):
        """Test basic event logging."""
        logger = EventLogger()
        
        event = Event(
            event_type=EventType.AUTH_REQUEST,
            correlation_id=uuid4(),
            source="test_component",
            data={"test": "data"}
        )
        
        # Log event (synchronous for testing)
        asyncio.run(logger.log_event(event))
        
        # Check event was stored
        events = logger.get_events_by_type(EventType.AUTH_REQUEST)
        assert len(events) == 1
        assert events[0].correlation_id == event.correlation_id
    
    def test_event_filtering(self):
        """Test event filtering by correlation ID and type."""
        logger = EventLogger()
        correlation_id = uuid4()
        
        # Create multiple events
        events = [
            Event(EventType.AUTH_REQUEST, correlation_id, source="test1"),
            Event(EventType.COMMITMENT_GENERATED, correlation_id, source="test1"),
            Event(EventType.AUTH_REQUEST, uuid4(), source="test2")
        ]
        
        # Log all events
        for event in events:
            asyncio.run(logger.log_event(event))
        
        # Filter by correlation ID
        correlation_events = logger.get_events_by_correlation_id(correlation_id)
        assert len(correlation_events) == 2
        
        # Filter by type
        auth_events = logger.get_events_by_type(EventType.AUTH_REQUEST)
        assert len(auth_events) == 2
    
    def test_event_timeframe_filtering(self):
        """Test event filtering by timeframe."""
        logger = EventLogger()
        
        start_time = time.time()
        
        # Create events with different timestamps
        old_event = Event(
            EventType.AUTH_REQUEST,
            uuid4(),
            timestamp=start_time - 100,
            source="old"
        )
        
        new_event = Event(
            EventType.AUTH_REQUEST,
            uuid4(),
            timestamp=start_time + 10,
            source="new"
        )
        
        # Log events
        asyncio.run(logger.log_event(old_event))
        asyncio.run(logger.log_event(new_event))
        
        # Filter by timeframe
        recent_events = logger.get_events_in_timeframe(start_time, start_time + 20)
        assert len(recent_events) == 1
        assert recent_events[0].source == "new"


class TestCorrelationManager:
    """Test correlation ID management."""
    
    def test_correlation_creation_and_retrieval(self):
        """Test correlation ID creation and information retrieval."""
        manager = CorrelationManager()
        
        correlation_id = manager.create_correlation(
            context="test_auth",
            metadata={"device_id": "test_device"}
        )
        
        # Check correlation info
        info = manager.get_correlation_info(correlation_id)
        assert info is not None
        assert info["context"] == "test_auth"
        assert info["metadata"]["device_id"] == "test_device"
        assert "created_at" in info
    
    def test_correlation_cleanup(self):
        """Test correlation ID cleanup."""
        manager = CorrelationManager()
        
        # Create correlation
        correlation_id = manager.create_correlation("test_auth")
        assert manager.get_correlation_info(correlation_id) is not None
        
        # Close correlation
        manager.close_correlation(correlation_id)
        assert manager.get_correlation_info(correlation_id) is None
    
    def test_old_correlation_cleanup(self):
        """Test automatic cleanup of old correlations."""
        manager = CorrelationManager()
        
        # Create old correlation by manipulating created_at
        correlation_id = manager.create_correlation("old_auth")
        manager._active_correlations[correlation_id]["created_at"] = time.time() - 7200  # 2 hours ago
        
        # Create recent correlation
        recent_id = manager.create_correlation("recent_auth")
        
        # Cleanup old correlations (max age 1 hour)
        manager.cleanup_old_correlations(max_age_seconds=3600)
        
        # Old correlation should be gone, recent should remain
        assert manager.get_correlation_info(correlation_id) is None
        assert manager.get_correlation_info(recent_id) is not None


class TestMobilityPredictor:
    """Test mobility prediction functionality."""
    
    @pytest.fixture
    async def event_bus(self):
        """Create event bus for mobility testing."""
        bus = EventBus()
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.fixture
    def predictor(self, event_bus):
        """Create mobility predictor."""
        return MobilityPredictor(event_bus)
    
    @pytest.mark.asyncio
    async def test_location_update(self, predictor):
        """Test location update handling."""
        device_id = "test_device"
        location = LocationPoint(
            latitude=37.7749,
            longitude=-122.4194,
            timestamp=time.time()
        )
        
        await predictor.update_location(device_id, location)
        
        # Check location was stored
        assert device_id in predictor.mobility_history
        assert len(predictor.mobility_history[device_id]) == 1
        assert predictor.mobility_history[device_id][0] == location
    
    @pytest.mark.asyncio
    async def test_feature_extraction(self, predictor):
        """Test mobility feature extraction."""
        device_id = "test_device"
        
        # Add multiple locations to enable feature extraction
        locations = [
            LocationPoint(37.7749, -122.4194, time.time() - 60),
            LocationPoint(37.7750, -122.4195, time.time() - 30),
            LocationPoint(37.7751, -122.4196, time.time())
        ]
        
        for location in locations:
            await predictor.update_location(device_id, location)
        
        # Check features were extracted
        assert device_id in predictor.feature_history
        assert len(predictor.feature_history[device_id]) >= 1
        
        features = predictor.feature_history[device_id][-1]
        assert hasattr(features, "hour_of_day")
        assert hasattr(features, "speed")
        assert hasattr(features, "distance_from_last")
    
    @pytest.mark.asyncio
    async def test_prediction_without_training(self, predictor):
        """Test prediction behavior when model is not trained."""
        device_id = "test_device"
        location = LocationPoint(37.7749, -122.4194, time.time())
        
        await predictor.update_location(device_id, location)
        predictions = await predictor.predict_mobility(device_id)
        
        # Should return empty list when not trained
        assert predictions == []
    
    @pytest.mark.asyncio
    async def test_device_stats(self, predictor):
        """Test device statistics retrieval."""
        device_id = "test_device"
        
        # Add some location data
        for i in range(5):
            location = LocationPoint(
                latitude=37.7749 + i * 0.001,
                longitude=-122.4194 + i * 0.001,
                timestamp=time.time() + i * 60
            )
            await predictor.update_location(device_id, location)
        
        stats = predictor.get_device_stats(device_id)
        
        assert stats["device_id"] == device_id
        assert stats["total_locations"] == 5
        assert "avg_speed" in stats
        assert "total_distance" in stats
        assert "first_seen" in stats
        assert "last_seen" in stats
    
    def test_distance_calculation(self, predictor):
        """Test distance calculation between coordinates."""
        # Test distance between San Francisco and Los Angeles (approx 560km)
        sf_lat, sf_lon = 37.7749, -122.4194
        la_lat, la_lon = 34.0522, -118.2437
        
        distance = predictor._calculate_distance(sf_lat, sf_lon, la_lat, la_lon)
        
        # Should be approximately 560,000 meters (allow 10% tolerance)
        expected_distance = 560000
        assert abs(distance - expected_distance) < expected_distance * 0.1


class TestIntegration:
    """Integration tests for event-driven architecture."""
    
    @pytest.fixture
    async def system(self):
        """Create integrated system for testing."""
        event_bus = EventBus()
        await event_bus.start()
        
        gateway_sm = GatewayStateMachine("test_gateway", event_bus)
        device_sm = DeviceStateMachine("test_device", event_bus)
        predictor = MobilityPredictor(event_bus)
        
        yield {
            "event_bus": event_bus,
            "gateway_sm": gateway_sm,
            "device_sm": device_sm,
            "predictor": predictor
        }
        
        await event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_complete_auth_flow_with_events(self, system):
        """Test complete authentication flow using events."""
        event_bus = system["event_bus"]
        gateway_sm = system["gateway_sm"]
        device_sm = system["device_sm"]
        
        correlation_id = uuid4()
        
        # Start authentication flow
        await event_bus.publish_event(
            event_type=EventType.AUTH_REQUEST,
            correlation_id=correlation_id,
            source="test_device",
            target="test_gateway",
            data={"device_id": "test_device"}
        )
        
        # Wait for event processing
        await asyncio.sleep(0.1)
        
        # Check state machines responded
        assert gateway_sm.current_state == StateType.AWAITING_COMMITMENT
        
        # Continue flow with commitment
        await event_bus.publish_event(
            event_type=EventType.COMMITMENT_GENERATED,
            correlation_id=correlation_id,
            source="test_device",
            data={"commitment": "test_commitment", "device_id": "test_device"}
        )
        
        await asyncio.sleep(0.1)
        assert gateway_sm.current_state == StateType.AWAITING_RESPONSE
        
        # Complete with valid ZKP
        await event_bus.publish_event(
            event_type=EventType.VERIFICATION_COMPLETE,
            correlation_id=correlation_id,
            source="test_gateway",
            data={"zkp": {"valid": True}}
        )
        
        await asyncio.sleep(0.1)
        assert gateway_sm.current_state == StateType.AUTHENTICATED
    
    @pytest.mark.asyncio
    async def test_mobility_event_handling(self, system):
        """Test mobility prediction event handling."""
        event_bus = system["event_bus"]
        predictor = system["predictor"]
        
        correlation_id = uuid4()
        
        # Publish location change event
        await event_bus.publish_event(
            event_type=EventType.LOCATION_CHANGED,
            correlation_id=correlation_id,
            source="test_device",
            data={
                "device_id": "test_device",
                "location": {
                    "latitude": 37.7749,
                    "longitude": -122.4194,
                    "timestamp": time.time()
                }
            }
        )
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check location was stored
        assert "test_device" in predictor.mobility_history
        assert len(predictor.mobility_history["test_device"]) == 1
    
    @pytest.mark.asyncio
    async def test_error_propagation(self, system):
        """Test error event propagation through the system."""
        event_bus = system["event_bus"]
        gateway_sm = system["gateway_sm"]
        
        error_events = []
        
        async def error_handler(event: Event) -> None:
            error_events.append(event)
        
        event_bus.subscribe(EventType.PROTOCOL_VIOLATION, error_handler)
        
        correlation_id = uuid4()
        
        # Publish protocol violation
        await event_bus.publish_event(
            event_type=EventType.PROTOCOL_VIOLATION,
            correlation_id=correlation_id,
            source="test_component",
            data={"error": "Invalid message format"}
        )
        
        await asyncio.sleep(0.1)
        
        # Check error was captured
        assert len(error_events) == 1
        assert error_events[0].data["error"] == "Invalid message format"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
