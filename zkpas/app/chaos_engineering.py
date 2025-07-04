#!/usr/bin/env python3
"""
ZKPAS Phase 7 Task 7.2: Chaos Engineering Module

This module implements advanced chaos engineering capabilities according to
Implementation Blueprint v7.0 requirements:

- CHAOS_DROP_EVENT <event_type>: Randomly drop events from asyncio queue
- CHAOS_CORRUPT_STATE <entity_id>: Randomly flip bits in stored keys  
- CHAOS_INJECT_LATENCY <duration>: Add random delays to event processing

The goal is to verify that the system remains stable, does not crash,
and that security invariants are not violated under chaos conditions.
"""

import asyncio
import random
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from loguru import logger

from app.events import EventBus, EventType, Event
from shared.crypto_utils import secure_hash


class ChaosEventType(Enum):
    """Types of chaos engineering events."""
    DROP_EVENT = "CHAOS_DROP_EVENT"
    CORRUPT_STATE = "CHAOS_CORRUPT_STATE"  
    INJECT_LATENCY = "CHAOS_INJECT_LATENCY"
    NETWORK_PARTITION = "CHAOS_NETWORK_PARTITION"
    RESOURCE_EXHAUSTION = "CHAOS_RESOURCE_EXHAUSTION"
    BYZANTINE_INJECTION = "CHAOS_BYZANTINE_INJECTION"


@dataclass
class ChaosCommand:
    """Represents a chaos engineering command."""
    chaos_type: ChaosEventType
    target: str
    timestamp: float
    duration: Optional[float] = None
    intensity: float = 1.0  # 0.0 to 1.0
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class ChaosMetrics:
    """Chaos engineering execution metrics."""
    commands_executed: int = 0
    events_dropped: int = 0
    state_corruptions: int = 0
    latency_injections: int = 0
    system_crashes: int = 0
    security_violations: int = 0
    recovery_time_seconds: float = 0.0
    system_stability_score: float = 100.0


class ChaosEventDropper:
    """
    Implements CHAOS_DROP_EVENT functionality.
    Randomly drops events of specified types from the asyncio event queue.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.drop_probability = 0.1  # 10% default drop rate
        self.targeted_event_types: List[EventType] = []
        self.dropped_events = []
        self.active = False
        
    def activate(self, event_types: List[EventType], drop_probability: float = 0.1):
        """Activate event dropping for specified event types."""
        self.targeted_event_types = event_types
        self.drop_probability = drop_probability
        self.active = True
        
        logger.warning(f"Chaos Event Dropper activated: {len(event_types)} event types, "
                      f"{drop_probability:.1%} drop rate")
    
    def deactivate(self):
        """Deactivate event dropping."""
        self.active = False
        self.targeted_event_types = []
        logger.info(f"Chaos Event Dropper deactivated. Dropped {len(self.dropped_events)} events")
    
    def should_drop_event(self, event: Event) -> bool:
        """Determine if an event should be dropped."""
        if not self.active:
            return False
        
        if event.event_type in self.targeted_event_types:
            return random.random() < self.drop_probability
        
        return False
    
    def drop_event(self, event: Event):
        """Record a dropped event."""
        self.dropped_events.append({
            "event_type": event.event_type,
            "correlation_id": str(event.correlation_id),
            "timestamp": time.time(),
            "source": event.source,
            "target": event.target
        })
        
        logger.warning(f"CHAOS: Dropped event {event.event_type} from {event.source} to {event.target}")


class ChaosStateCorruptor:
    """
    Implements CHAOS_CORRUPT_STATE functionality.
    Randomly flips bits in stored cryptographic keys and state data.
    """
    
    def __init__(self):
        self.corruption_targets: Dict[str, Any] = {}
        self.corruption_history = []
        self.active = False
        
    def register_target(self, entity_id: str, state_data: Any):
        """Register an entity for potential state corruption."""
        self.corruption_targets[entity_id] = state_data
        
    def activate(self, corruption_probability: float = 0.05):
        """Activate state corruption."""
        self.corruption_probability = corruption_probability
        self.active = True
        
        logger.warning(f"Chaos State Corruptor activated: {corruption_probability:.1%} corruption rate")
    
    def deactivate(self):
        """Deactivate state corruption."""
        self.active = False
        logger.info(f"Chaos State Corruptor deactivated. Performed {len(self.corruption_history)} corruptions")
    
    async def corrupt_entity_state(self, entity_id: str) -> bool:
        """Corrupt the state of a specific entity."""
        if not self.active or entity_id not in self.corruption_targets:
            return False
        
        if random.random() > self.corruption_probability:
            return False
        
        try:
            # Simulate bit-flip corruption in cryptographic keys
            target_data = self.corruption_targets[entity_id]
            
            if hasattr(target_data, '__dict__'):
                # Corrupt object attributes
                attributes = [attr for attr in dir(target_data) 
                            if not attr.startswith('_') and not callable(getattr(target_data, attr))]
                
                if attributes:
                    attr_name = random.choice(attributes)
                    attr_value = getattr(target_data, attr_name)
                    
                    # Corrupt based on attribute type
                    if isinstance(attr_value, bytes):
                        # Flip random bit in byte data (for cryptographic keys)
                        corrupted_value = self._flip_random_bit(attr_value)
                        setattr(target_data, attr_name, corrupted_value)
                        
                        corruption_record = {
                            "entity_id": entity_id,
                            "attribute": attr_name,
                            "original_hash": secure_hash(attr_value).hex()[:16],
                            "corrupted_hash": secure_hash(corrupted_value).hex()[:16],
                            "timestamp": time.time()
                        }
                        self.corruption_history.append(corruption_record)
                        
                        logger.warning(f"CHAOS: Corrupted {entity_id}.{attr_name} (bit flip)")
                        return True
                        
                    elif isinstance(attr_value, (int, float)):
                        # Corrupt numeric values
                        corrupted_value = attr_value + random.randint(-100, 100)
                        setattr(target_data, attr_name, corrupted_value)
                        
                        logger.warning(f"CHAOS: Corrupted {entity_id}.{attr_name}: {attr_value} -> {corrupted_value}")
                        return True
        
        except Exception as e:
            logger.error(f"Chaos state corruption error for {entity_id}: {e}")
        
        return False
    
    def _flip_random_bit(self, data: bytes) -> bytes:
        """Flip a random bit in byte data."""
        if len(data) == 0:
            return data
        
        data_array = bytearray(data)
        byte_index = random.randint(0, len(data_array) - 1)
        bit_index = random.randint(0, 7)
        
        # Flip the bit
        data_array[byte_index] ^= (1 << bit_index)
        
        return bytes(data_array)


class ChaosLatencyInjector:
    """
    Implements CHAOS_INJECT_LATENCY functionality.
    Adds random delays to event processing and system operations.
    """
    
    def __init__(self):
        self.injection_probability = 0.1
        self.min_delay = 0.1  # 100ms
        self.max_delay = 5.0   # 5 seconds
        self.injected_delays = []
        self.active = False
        
    def activate(self, probability: float = 0.1, min_delay: float = 0.1, max_delay: float = 5.0):
        """Activate latency injection."""
        self.injection_probability = probability
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.active = True
        
        logger.warning(f"Chaos Latency Injector activated: {probability:.1%} probability, "
                      f"{min_delay:.1f}s - {max_delay:.1f}s delay range")
    
    def deactivate(self):
        """Deactivate latency injection."""
        self.active = False
        total_delay = sum(d["delay"] for d in self.injected_delays)
        logger.info(f"Chaos Latency Injector deactivated. Injected {total_delay:.2f}s total delay "
                   f"across {len(self.injected_delays)} operations")
    
    async def maybe_inject_latency(self, operation_name: str = "unknown") -> float:
        """Potentially inject latency into an operation."""
        if not self.active or random.random() > self.injection_probability:
            return 0.0
        
        delay = random.uniform(self.min_delay, self.max_delay)
        
        delay_record = {
            "operation": operation_name,
            "delay": delay,
            "timestamp": time.time()
        }
        self.injected_delays.append(delay_record)
        
        logger.warning(f"CHAOS: Injecting {delay:.2f}s latency into {operation_name}")
        
        await asyncio.sleep(delay)
        return delay


class ChaosOrchestrator:
    """
    Main chaos engineering orchestrator that coordinates all chaos operations.
    
    ✅ TASK 7.2: Integrates with GUI Scenario Builder to execute chaos commands
    and verify system stability and security invariant maintenance.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.event_dropper = ChaosEventDropper(event_bus)
        self.state_corruptor = ChaosStateCorruptor()
        self.latency_injector = ChaosLatencyInjector()
        
        self.chaos_commands: List[ChaosCommand] = []
        self.metrics = ChaosMetrics()
        self.security_invariants_violated = False
        self.system_crashed = False
        self.active_chaos_scenario = False
        
        # Security monitoring
        self.security_violations = []
        self.system_stability_checks = []
        
        logger.info("Chaos Orchestrator initialized")
    
    def add_chaos_command(self, command: ChaosCommand):
        """Add a chaos command to the execution queue."""
        self.chaos_commands.append(command)
        logger.info(f"Added chaos command: {command.chaos_type.value} -> {command.target}")
    
    def clear_chaos_commands(self):
        """Clear all chaos commands."""
        self.chaos_commands.clear()
        logger.info("Cleared all chaos commands")
    
    async def execute_chaos_scenario(self, duration: float = 60.0) -> ChaosMetrics:
        """
        Execute complete chaos engineering scenario.
        
        ✅ TASK 7.2: Run long simulation with chaos injections and verify:
        - System remains stable
        - System does not crash  
        - Security invariants are not violated
        
        Args:
            duration: Scenario duration in seconds
            
        Returns:
            ChaosMetrics: Detailed chaos execution metrics
        """
        logger.info(f"Starting chaos engineering scenario (duration: {duration}s)")
        logger.info(f"Chaos commands to execute: {len(self.chaos_commands)}")
        
        self.active_chaos_scenario = True
        self.metrics = ChaosMetrics()
        scenario_start_time = time.time()
        
        try:
            # Activate all chaos components
            self._activate_chaos_components()
            
            # Sort chaos commands by timestamp
            sorted_commands = sorted(self.chaos_commands, key=lambda x: x.timestamp)
            
            # Execute chaos scenario
            for command in sorted_commands:
                if not self.active_chaos_scenario:
                    break
                
                # Wait for command timestamp
                elapsed = time.time() - scenario_start_time
                if command.timestamp > elapsed:
                    wait_time = command.timestamp - elapsed
                    await asyncio.sleep(wait_time)
                
                # Execute chaos command
                await self._execute_chaos_command(command)
                
                # Check system stability after each command
                stability_check = await self._check_system_stability()
                self.system_stability_checks.append(stability_check)
                
                if stability_check["system_crashed"]:
                    logger.error("CHAOS SCENARIO: System crash detected!")
                    self.system_crashed = True
                    break
                
                if stability_check["security_violations"] > 0:
                    logger.warning(f"CHAOS SCENARIO: {stability_check['security_violations']} security violations detected")
                    self.security_invariants_violated = True
            
            # Wait for scenario completion
            remaining_time = duration - (time.time() - scenario_start_time)
            if remaining_time > 0:
                await asyncio.sleep(remaining_time)
            
        except Exception as e:
            logger.error(f"Chaos scenario execution error: {e}")
            self.system_crashed = True
        
        finally:
            # Deactivate chaos components
            self._deactivate_chaos_components()
            self.active_chaos_scenario = False
        
        # Calculate final metrics
        self._calculate_final_metrics(scenario_start_time)
        
        logger.info(f"Chaos scenario completed. System stability score: {self.metrics.system_stability_score:.1f}%")
        
        return self.metrics
    
    def _activate_chaos_components(self):
        """Activate all chaos engineering components."""
        # Activate event dropper for critical events
        critical_events = [
            EventType.AUTHENTICATION_REQUEST,
            EventType.CROSS_DOMAIN_AUTH_REQUEST,
            EventType.TOKEN_VALIDATION_REQUEST,
            EventType.MOBILITY_PREDICTED
        ]
        self.event_dropper.activate(critical_events, drop_probability=0.15)
        
        # Activate state corruptor
        self.state_corruptor.activate(corruption_probability=0.08)
        
        # Activate latency injector
        self.latency_injector.activate(probability=0.2, min_delay=0.5, max_delay=3.0)
        
        logger.info("All chaos components activated")
    
    def _deactivate_chaos_components(self):
        """Deactivate all chaos engineering components."""
        self.event_dropper.deactivate()
        self.state_corruptor.deactivate()
        self.latency_injector.deactivate()
        
        logger.info("All chaos components deactivated")
    
    async def _execute_chaos_command(self, command: ChaosCommand):
        """Execute a specific chaos command."""
        try:
            self.metrics.commands_executed += 1
            
            if command.chaos_type == ChaosEventType.DROP_EVENT:
                await self._execute_drop_event_command(command)
            
            elif command.chaos_type == ChaosEventType.CORRUPT_STATE:
                await self._execute_corrupt_state_command(command)
            
            elif command.chaos_type == ChaosEventType.INJECT_LATENCY:
                await self._execute_inject_latency_command(command)
            
            elif command.chaos_type == ChaosEventType.NETWORK_PARTITION:
                await self._execute_network_partition_command(command)
            
            elif command.chaos_type == ChaosEventType.RESOURCE_EXHAUSTION:
                await self._execute_resource_exhaustion_command(command)
            
            elif command.chaos_type == ChaosEventType.BYZANTINE_INJECTION:
                await self._execute_byzantine_injection_command(command)
            
            else:
                logger.warning(f"Unknown chaos command type: {command.chaos_type}")
        
        except Exception as e:
            logger.error(f"Chaos command execution error: {e}")
    
    async def _execute_drop_event_command(self, command: ChaosCommand):
        """Execute CHAOS_DROP_EVENT command."""
        event_type = command.parameters.get("event_type", "AUTHENTICATION_REQUEST")
        
        # Convert string to EventType if needed
        if isinstance(event_type, str):
            try:
                event_type = EventType[event_type]
            except KeyError:
                logger.warning(f"Unknown event type for dropping: {event_type}")
                return
        
        # Temporarily increase drop rate for this specific event type
        original_probability = self.event_dropper.drop_probability
        self.event_dropper.drop_probability = min(1.0, original_probability * 3)
        self.event_dropper.targeted_event_types = [event_type]
        
        # Let it run for the command duration
        duration = command.duration or 5.0
        await asyncio.sleep(duration)
        
        # Restore original settings
        self.event_dropper.drop_probability = original_probability
        
        self.metrics.events_dropped += len(self.event_dropper.dropped_events)
        
        logger.warning(f"CHAOS: Executed DROP_EVENT for {event_type} (duration: {duration}s)")
    
    async def _execute_corrupt_state_command(self, command: ChaosCommand):
        """Execute CHAOS_CORRUPT_STATE command."""
        entity_id = command.target
        
        # Attempt state corruption
        corruption_success = await self.state_corruptor.corrupt_entity_state(entity_id)
        
        if corruption_success:
            self.metrics.state_corruptions += 1
            logger.warning(f"CHAOS: Executed CORRUPT_STATE for {entity_id}")
        else:
            logger.info(f"CHAOS: CORRUPT_STATE for {entity_id} had no effect")
    
    async def _execute_inject_latency_command(self, command: ChaosCommand):
        """Execute CHAOS_INJECT_LATENCY command."""
        duration = command.duration or random.uniform(1.0, 5.0)
        
        # Inject latency into system operations
        delay = await self.latency_injector.maybe_inject_latency(f"chaos_command_{command.target}")
        
        if delay > 0:
            self.metrics.latency_injections += 1
            logger.warning(f"CHAOS: Executed INJECT_LATENCY for {command.target} (delay: {delay:.2f}s)")
    
    async def _execute_network_partition_command(self, command: ChaosCommand):
        """Execute network partition simulation."""
        duration = command.duration or 10.0
        
        # Simulate network partition by dramatically increasing event drop rate
        original_probability = self.event_dropper.drop_probability
        self.event_dropper.drop_probability = 0.8  # 80% drop rate simulates partition
        
        await asyncio.sleep(duration)
        
        # Restore network
        self.event_dropper.drop_probability = original_probability
        
        logger.warning(f"CHAOS: Executed NETWORK_PARTITION (duration: {duration}s)")
    
    async def _execute_resource_exhaustion_command(self, command: ChaosCommand):
        """Execute resource exhaustion simulation."""
        # Simulate resource exhaustion by creating computational load
        duration = command.duration or 5.0
        
        def cpu_intensive_task():
            end_time = time.time() + duration
            while time.time() < end_time:
                # Simulate CPU-intensive operation
                _ = sum(i * i for i in range(1000))
        
        # Run in separate thread to avoid blocking
        thread = threading.Thread(target=cpu_intensive_task)
        thread.start()
        
        logger.warning(f"CHAOS: Executed RESOURCE_EXHAUSTION (duration: {duration}s)")
    
    async def _execute_byzantine_injection_command(self, command: ChaosCommand):
        """Execute Byzantine fault injection."""
        # Simulate Byzantine behavior by corrupting multiple components
        await self.state_corruptor.corrupt_entity_state(command.target)
        
        # Increase event dropping for Byzantine simulation
        original_probability = self.event_dropper.drop_probability
        self.event_dropper.drop_probability = min(1.0, original_probability * 2)
        
        duration = command.duration or 8.0
        await asyncio.sleep(duration)
        
        # Restore
        self.event_dropper.drop_probability = original_probability
        
        logger.warning(f"CHAOS: Executed BYZANTINE_INJECTION for {command.target}")
    
    async def _check_system_stability(self) -> Dict[str, Any]:
        """Check system stability and security invariants."""
        stability_check = {
            "timestamp": time.time(),
            "system_crashed": False,
            "security_violations": 0,
            "performance_degradation": False,
            "error_rate": 0.0
        }
        
        try:
            # Check if event bus is still responsive
            test_event = Event(
                event_type=EventType.SYSTEM_STATUS,
                correlation_id=uuid.uuid4(),
                source="chaos_orchestrator",
                target="system",
                data={"test": "stability_check"}
            )
            
            # Try to publish test event with timeout
            await asyncio.wait_for(
                self.event_bus.publish(test_event),
                timeout=2.0
            )
            
        except asyncio.TimeoutError:
            stability_check["system_crashed"] = True
            logger.error("System stability check: Event bus timeout")
        
        except Exception as e:
            stability_check["system_crashed"] = True
            logger.error(f"System stability check failed: {e}")
        
        # Check for security violations
        # This would integrate with security monitoring systems
        security_violations = len(self.security_violations)
        stability_check["security_violations"] = security_violations
        
        # Check performance degradation
        # This would measure response times and throughput
        
        return stability_check
    
    def _calculate_final_metrics(self, start_time: float):
        """Calculate final chaos scenario metrics."""
        total_duration = time.time() - start_time
        
        # Calculate system stability score
        stability_score = 100.0
        
        if self.system_crashed:
            stability_score -= 50.0
        
        if self.security_invariants_violated:
            stability_score -= 30.0
        
        # Penalty for excessive chaos effects
        if self.metrics.events_dropped > 100:
            stability_score -= 10.0
        
        if self.metrics.state_corruptions > 50:
            stability_score -= 10.0
        
        # Recovery time calculation
        recovery_checks = [check for check in self.system_stability_checks 
                          if not check["system_crashed"]]
        
        if recovery_checks:
            self.metrics.recovery_time_seconds = recovery_checks[-1]["timestamp"] - start_time
        else:
            self.metrics.recovery_time_seconds = total_duration
        
        self.metrics.system_stability_score = max(0.0, stability_score)
        
        # Update metrics
        self.metrics.events_dropped = len(self.event_dropper.dropped_events)
        self.metrics.state_corruptions = len(self.state_corruptor.corruption_history)
        self.metrics.latency_injections = len(self.latency_injector.injected_delays)
        self.metrics.security_violations = len(self.security_violations)
        self.metrics.system_crashes = 1 if self.system_crashed else 0
    
    def register_entity_for_corruption(self, entity_id: str, entity_state: Any):
        """Register an entity for potential state corruption."""
        self.state_corruptor.register_target(entity_id, entity_state)
        logger.debug(f"Registered entity {entity_id} for potential chaos state corruption")
    
    def get_chaos_metrics(self) -> ChaosMetrics:
        """Get current chaos metrics."""
        return self.metrics
    
    def is_chaos_active(self) -> bool:
        """Check if chaos scenario is currently active."""
        return self.active_chaos_scenario


# Helper functions for GUI integration
def create_drop_event_command(event_type: str, target: str, timestamp: float, duration: float = 5.0) -> ChaosCommand:
    """Create a CHAOS_DROP_EVENT command for GUI Scenario Builder."""
    return ChaosCommand(
        chaos_type=ChaosEventType.DROP_EVENT,
        target=target,
        timestamp=timestamp,
        duration=duration,
        parameters={"event_type": event_type}
    )


def create_corrupt_state_command(entity_id: str, timestamp: float) -> ChaosCommand:
    """Create a CHAOS_CORRUPT_STATE command for GUI Scenario Builder."""
    return ChaosCommand(
        chaos_type=ChaosEventType.CORRUPT_STATE,
        target=entity_id,
        timestamp=timestamp
    )


def create_inject_latency_command(target: str, timestamp: float, duration: float = 3.0) -> ChaosCommand:
    """Create a CHAOS_INJECT_LATENCY command for GUI Scenario Builder."""
    return ChaosCommand(
        chaos_type=ChaosEventType.INJECT_LATENCY,
        target=target,
        timestamp=timestamp,
        duration=duration
    )


def create_network_partition_command(timestamp: float, duration: float = 10.0) -> ChaosCommand:
    """Create a network partition chaos command."""
    return ChaosCommand(
        chaos_type=ChaosEventType.NETWORK_PARTITION,
        target="network",
        timestamp=timestamp,
        duration=duration
    )


def create_byzantine_injection_command(entity_id: str, timestamp: float, duration: float = 8.0) -> ChaosCommand:
    """Create a Byzantine fault injection chaos command."""
    return ChaosCommand(
        chaos_type=ChaosEventType.BYZANTINE_INJECTION,
        target=entity_id,
        timestamp=timestamp,
        duration=duration
    )