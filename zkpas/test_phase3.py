#!/usr/bin/env python3
"""
Simple test runner for Phase 3 implementation validation.

This script validates the event-driven architecture and state machine
implementation without requiring external test frameworks.
"""

import asyncio
import sys
import time
from pathlib import Path
from uuid import uuid4

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.events import Event, EventBus, EventType, event_logger, correlation_manager
from app.state_machine import GatewayStateMachine, DeviceStateMachine, StateType
from app.mobility_predictor import MobilityPredictor, LocationPoint


async def test_event_bus():
    """Test basic event bus functionality."""
    print("Testing Event Bus...")
    
    event_bus = EventBus(max_queue_size=100)
    await event_bus.start()
    
    events_received = []
    
    async def event_handler(event: Event) -> None:
        events_received.append(event)
        print(f"  Received event: {event.event_type.name} from {event.source}")
    
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
    
    # Verify
    assert len(events_received) == 1, f"Expected 1 event, got {len(events_received)}"
    assert events_received[0].event_type == EventType.AUTH_REQUEST
    
    await event_bus.stop()
    print("  ✓ Event Bus test passed")


async def test_state_machine():
    """Test state machine functionality."""
    print("Testing State Machine...")
    
    event_bus = EventBus()
    await event_bus.start()
    
    gateway_sm = GatewayStateMachine("test_gateway", event_bus)
    
    # Test initial state
    assert gateway_sm.current_state == StateType.IDLE, f"Expected IDLE, got {gateway_sm.current_state}"
    print("  ✓ Initial state is IDLE")
    
    # Test state transition
    auth_event = Event(
        event_type=EventType.AUTH_REQUEST,
        correlation_id=uuid4(),
        source="test_device",
        data={"device_id": "test_device"}
    )
    
    await gateway_sm.handle_event(auth_event)
    assert gateway_sm.current_state == StateType.AWAITING_COMMITMENT
    print("  ✓ State transition IDLE -> AWAITING_COMMITMENT works")
    
    # Test state info
    info = gateway_sm.get_state_info()
    assert info["component_id"] == "test_gateway"
    assert info["current_state"] == "AWAITING_COMMITMENT"
    print("  ✓ State info retrieval works")
    
    await event_bus.stop()
    print("  ✓ State Machine test passed")


async def test_mobility_predictor():
    """Test mobility predictor functionality."""
    print("Testing Mobility Predictor...")
    
    event_bus = EventBus()
    await event_bus.start()
    
    predictor = MobilityPredictor(event_bus)
    
    # Test location update
    device_id = "test_device"
    location = LocationPoint(
        latitude=37.7749,
        longitude=-122.4194,
        timestamp=time.time()
    )
    
    await predictor.update_location(device_id, location)
    
    # Verify location stored
    assert device_id in predictor.mobility_history
    assert len(predictor.mobility_history[device_id]) == 1
    print("  ✓ Location update works")
    
    # Test stats
    stats = predictor.get_device_stats(device_id)
    assert stats["device_id"] == device_id
    assert stats["total_locations"] == 1
    print("  ✓ Device statistics work")
    
    # Test distance calculation
    distance = predictor._calculate_distance(37.7749, -122.4194, 37.7750, -122.4195)
    assert distance > 0, "Distance should be positive"
    print("  ✓ Distance calculation works")
    
    await event_bus.stop()
    print("  ✓ Mobility Predictor test passed")


async def test_correlation_manager():
    """Test correlation manager functionality."""
    print("Testing Correlation Manager...")
    
    manager = correlation_manager
    
    # Test correlation creation
    correlation_id = manager.create_correlation(
        context="test_auth",
        metadata={"device_id": "test_device"}
    )
    
    # Test retrieval
    info = manager.get_correlation_info(correlation_id)
    assert info is not None
    assert info["context"] == "test_auth"
    assert info["metadata"]["device_id"] == "test_device"
    print("  ✓ Correlation creation and retrieval works")
    
    # Test cleanup
    manager.close_correlation(correlation_id)
    assert manager.get_correlation_info(correlation_id) is None
    print("  ✓ Correlation cleanup works")
    
    print("  ✓ Correlation Manager test passed")


async def test_integration():
    """Test integration between components."""
    print("Testing Integration...")
    
    event_bus = EventBus()
    await event_bus.start()
    
    gateway_sm = GatewayStateMachine("test_gateway", event_bus)
    device_sm = DeviceStateMachine("test_device", event_bus)
    predictor = MobilityPredictor(event_bus)
    
    correlation_id = uuid4()
    
    # Test authentication flow
    await event_bus.publish_event(
        event_type=EventType.AUTH_REQUEST,
        correlation_id=correlation_id,
        source="test_device",
        target="test_gateway",
        data={"device_id": "test_device"}
    )
    
    await asyncio.sleep(0.1)
    assert gateway_sm.current_state == StateType.AWAITING_COMMITMENT
    print("  ✓ Gateway responded to auth request")
    
    # Test location event
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
    
    await asyncio.sleep(0.1)
    assert "test_device" in predictor.mobility_history
    print("  ✓ Mobility predictor handled location event")
    
    await event_bus.stop()
    print("  ✓ Integration test passed")


async def test_event_logging():
    """Test event logging functionality."""
    print("Testing Event Logger...")
    
    logger = event_logger
    
    event = Event(
        event_type=EventType.AUTH_REQUEST,
        correlation_id=uuid4(),
        source="test_component",
        data={"test": "data"}
    )
    
    await logger.log_event(event)
    
    # Test filtering
    events = logger.get_events_by_type(EventType.AUTH_REQUEST)
    assert len(events) >= 1
    print("  ✓ Event logging and filtering works")
    
    print("  ✓ Event Logger test passed")


async def main():
    """Run all tests."""
    print("🚀 Running Phase 3 Implementation Tests")
    print("=" * 50)
    
    try:
        await test_event_bus()
        await test_state_machine()
        await test_mobility_predictor()
        await test_correlation_manager()
        await test_integration()
        await test_event_logging()
        
        print("=" * 50)
        print("✅ All Phase 3 tests passed!")
        print("\n📊 Phase 3 Implementation Summary:")
        print("  • Async Event Bus: ✓ Working")
        print("  • Formal State Machines: ✓ Working")
        print("  • Mobility Prediction: ✓ Working")
        print("  • Correlation Management: ✓ Working")
        print("  • Component Integration: ✓ Working")
        print("  • Event Logging: ✓ Working")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
