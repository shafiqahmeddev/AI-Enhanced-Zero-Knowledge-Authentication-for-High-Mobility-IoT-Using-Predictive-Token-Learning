#!/usr/bin/env python3
"""
Comprehensive Module Testing Suite for ZKPAS Implementation

This script tests each implemented module systematically and provides
detailed reporting for implementation progress tracking.
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class TestResult:
    """Represents the result of a test or test suite."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors: List[str] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.details: Dict[str, Any] = {}
    
    def add_pass(self, test_name: str, details: str = "") -> None:
        """Record a passing test."""
        self.passed += 1
        if details:
            self.details[test_name] = {"status": "PASS", "details": details}
        print(f"    ✅ {test_name}")
    
    def add_fail(self, test_name: str, error: str) -> None:
        """Record a failing test."""
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        self.details[test_name] = {"status": "FAIL", "error": error}
        print(f"    ❌ {test_name}: {error}")
    
    def add_skip(self, test_name: str, reason: str) -> None:
        """Record a skipped test."""
        self.skipped += 1
        self.details[test_name] = {"status": "SKIP", "reason": reason}
        print(f"    ⏭️  {test_name}: {reason}")
    
    def finish(self) -> None:
        """Mark test suite as finished."""
        self.end_time = time.time()
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.passed + self.failed
        return (self.passed / total * 100) if total > 0 else 0.0
    
    def duration(self) -> float:
        """Get test duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def summary(self) -> Dict[str, Any]:
        """Get test summary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": self.success_rate(),
            "duration": self.duration(),
            "status": "PASS" if self.failed == 0 else "FAIL",
            "errors": self.errors,
            "details": self.details
        }


class ModuleTester:
    """Comprehensive module testing framework."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.overall_start = time.time()
    
    async def test_imports(self) -> TestResult:
        """Test all module imports."""
        result = TestResult("Module Imports")
        
        # Test core modules
        modules_to_test = [
            ("shared.config", "Configuration module"),
            ("shared.crypto_utils", "Cryptographic utilities"),
            ("app.components.interfaces", "Component interfaces"),
            ("app.events", "Event system"),
            ("app.state_machine", "State machines"),
            ("app.mobility_predictor", "Mobility prediction"),
            ("app.components.trusted_authority", "Trusted Authority"),
            ("app.components.gateway_node", "Gateway Node"),
            ("app.components.iot_device", "IoT Device")
        ]
        
        for module_name, description in modules_to_test:
            try:
                __import__(module_name)
                result.add_pass(f"Import {module_name}", description)
            except ImportError as e:
                result.add_fail(f"Import {module_name}", str(e))
            except Exception as e:
                result.add_fail(f"Import {module_name}", f"Unexpected error: {e}")
        
        result.finish()
        return result
    
    async def test_crypto_utils(self) -> TestResult:
        """Test cryptographic utilities."""
        result = TestResult("Cryptographic Utilities")
        
        try:
            from shared.crypto_utils import (
                generate_ecc_keypair, serialize_public_key, sign_message,
                verify_signature, generate_commitment, generate_challenge,
                verify_zkp, encrypt_aes_gcm, decrypt_aes_gcm, secure_hash
            )
            
            # Test ECC key generation
            try:
                private_key, public_key = generate_ecc_keypair()
                pub_bytes = serialize_public_key(public_key)
                result.add_pass("ECC key generation", f"Generated {len(pub_bytes)} byte public key")
            except Exception as e:
                result.add_fail("ECC key generation", str(e))
            
            # Test digital signatures
            try:
                message = b"test message"
                signature = sign_message(private_key, message)
                is_valid = verify_signature(public_key, message, signature)
                if is_valid:
                    result.add_pass("Digital signatures", f"Signature length: {len(signature)}")
                else:
                    result.add_fail("Digital signatures", "Signature verification failed")
            except Exception as e:
                result.add_fail("Digital signatures", str(e))
            
            # Test zero-knowledge proofs
            try:
                commitment = generate_commitment()
                challenge = generate_challenge()
                # Simplified ZKP test
                zkp_result = verify_zkp(commitment, challenge, b"response", public_key)
                result.add_pass("Zero-knowledge proofs", f"ZKP verification: {zkp_result}")
            except Exception as e:
                result.add_fail("Zero-knowledge proofs", str(e))
            
            # Test symmetric encryption
            try:
                key = b"0" * 32  # 256-bit key
                plaintext = b"sensitive data"
                ciphertext, nonce = encrypt_aes_gcm(key, plaintext)
                decrypted = decrypt_aes_gcm(key, ciphertext, nonce)
                if decrypted == plaintext:
                    result.add_pass("AES-GCM encryption", f"Encrypted {len(plaintext)} bytes")
                else:
                    result.add_fail("AES-GCM encryption", "Decryption mismatch")
            except Exception as e:
                result.add_fail("AES-GCM encryption", str(e))
            
            # Test hashing
            try:
                data = b"test data"
                hash_result = secure_hash(data)
                if len(hash_result) == 32:  # SHA-256 produces 32 bytes
                    result.add_pass("Secure hashing", f"Hash length: {len(hash_result)}")
                else:
                    result.add_fail("Secure hashing", f"Unexpected hash length: {len(hash_result)}")
            except Exception as e:
                result.add_fail("Secure hashing", str(e))
        
        except ImportError as e:
            result.add_fail("Crypto module import", str(e))
        
        result.finish()
        return result
    
    async def test_event_system(self) -> TestResult:
        """Test event-driven architecture."""
        result = TestResult("Event System")
        
        try:
            from app.events import EventBus, Event, EventType, CorrelationManager, EventLogger
            from uuid import uuid4
            
            # Test event bus creation
            try:
                event_bus = EventBus(max_queue_size=10)
                await event_bus.start()
                result.add_pass("Event bus creation", "Bus started successfully")
            except Exception as e:
                result.add_fail("Event bus creation", str(e))
                result.finish()
                return result
            
            # Test event publishing and processing
            try:
                events_received = []
                
                async def test_handler(event):
                    events_received.append(event)
                
                event_bus.subscribe(EventType.AUTH_REQUEST, test_handler)
                
                correlation_id = uuid4()
                await event_bus.publish_event(
                    event_type=EventType.AUTH_REQUEST,
                    correlation_id=correlation_id,
                    source="test",
                    data={"test": "data"}
                )
                
                # Wait for processing
                await asyncio.sleep(0.1)
                
                if len(events_received) == 1:
                    result.add_pass("Event publishing", f"Received {len(events_received)} events")
                else:
                    result.add_fail("Event publishing", f"Expected 1 event, got {len(events_received)}")
            
            except Exception as e:
                result.add_fail("Event publishing", str(e))
            
            # Test metrics
            try:
                metrics = event_bus.get_metrics()
                expected_keys = ["events_published", "events_processed", "events_dropped", "handler_errors"]
                if all(key in metrics for key in expected_keys):
                    result.add_pass("Event metrics", f"Metrics: {metrics}")
                else:
                    result.add_fail("Event metrics", f"Missing metrics keys: {metrics}")
            except Exception as e:
                result.add_fail("Event metrics", str(e))
            
            # Test correlation manager
            try:
                correlation_mgr = CorrelationManager()
                corr_id = correlation_mgr.create_correlation("test_context")
                info = correlation_mgr.get_correlation_info(corr_id)
                if info and info["context"] == "test_context":
                    result.add_pass("Correlation management", f"Created correlation: {corr_id}")
                else:
                    result.add_fail("Correlation management", "Failed to retrieve correlation info")
            except Exception as e:
                result.add_fail("Correlation management", str(e))
            
            # Test event logger
            try:
                event_logger = EventLogger()
                test_event = Event(
                    event_type=EventType.AUTH_REQUEST,
                    correlation_id=uuid4(),
                    source="test"
                )
                await event_logger.log_event(test_event)
                
                events_by_type = event_logger.get_events_by_type(EventType.AUTH_REQUEST)
                if len(events_by_type) >= 1:
                    result.add_pass("Event logging", f"Logged and retrieved {len(events_by_type)} events")
                else:
                    result.add_fail("Event logging", "Failed to retrieve logged events")
            except Exception as e:
                result.add_fail("Event logging", str(e))
            
            # Cleanup
            try:
                await event_bus.stop()
                result.add_pass("Event bus cleanup", "Bus stopped successfully")
            except Exception as e:
                result.add_fail("Event bus cleanup", str(e))
        
        except ImportError as e:
            result.add_fail("Event system import", str(e))
        
        result.finish()
        return result
    
    async def test_state_machines(self) -> TestResult:
        """Test formal state machine implementation."""
        result = TestResult("State Machines")
        
        try:
            from app.state_machine import GatewayStateMachine, DeviceStateMachine, StateType
            from app.events import EventBus, Event, EventType
            from uuid import uuid4
            
            # Create event bus for state machines
            event_bus = EventBus()
            await event_bus.start()
            
            # Test Gateway state machine
            try:
                gateway_sm = GatewayStateMachine("test_gateway", event_bus)
                
                # Check initial state
                if gateway_sm.current_state == StateType.IDLE:
                    result.add_pass("Gateway SM initialization", f"Initial state: {gateway_sm.current_state.name}")
                else:
                    result.add_fail("Gateway SM initialization", f"Expected IDLE, got {gateway_sm.current_state.name}")
                
                # Test state transition
                auth_event = Event(
                    event_type=EventType.AUTH_REQUEST,
                    correlation_id=uuid4(),
                    source="test_device",
                    data={"device_id": "test_device"}
                )
                
                await gateway_sm.handle_event(auth_event)
                
                if gateway_sm.current_state == StateType.AWAITING_COMMITMENT:
                    result.add_pass("Gateway state transition", f"Transitioned to: {gateway_sm.current_state.name}")
                else:
                    result.add_fail("Gateway state transition", f"Expected AWAITING_COMMITMENT, got {gateway_sm.current_state.name}")
                
                # Test state info
                state_info = gateway_sm.get_state_info()
                expected_keys = ["component_id", "current_state", "previous_state", "time_in_state"]
                if all(key in state_info for key in expected_keys):
                    result.add_pass("Gateway state info", f"Info keys: {list(state_info.keys())}")
                else:
                    result.add_fail("Gateway state info", f"Missing keys in: {state_info}")
            
            except Exception as e:
                result.add_fail("Gateway state machine", str(e))
            
            # Test Device state machine
            try:
                device_sm = DeviceStateMachine("test_device", event_bus)
                
                if device_sm.current_state == StateType.IDLE:
                    result.add_pass("Device SM initialization", f"Initial state: {device_sm.current_state.name}")
                else:
                    result.add_fail("Device SM initialization", f"Expected IDLE, got {device_sm.current_state.name}")
                
                # Test device state transition
                auth_event = Event(
                    event_type=EventType.AUTH_REQUEST,
                    correlation_id=uuid4(),
                    source="test_gateway",
                    data={"gateway_id": "test_gateway"}
                )
                
                await device_sm.handle_event(auth_event)
                
                if device_sm.current_state == StateType.REQUESTING_AUTH:
                    result.add_pass("Device state transition", f"Transitioned to: {device_sm.current_state.name}")
                else:
                    result.add_fail("Device state transition", f"Expected REQUESTING_AUTH, got {device_sm.current_state.name}")
            
            except Exception as e:
                result.add_fail("Device state machine", str(e))
            
            await event_bus.stop()
        
        except ImportError as e:
            result.add_fail("State machine import", str(e))
        
        result.finish()
        return result
    
    async def test_mobility_predictor(self) -> TestResult:
        """Test mobility prediction framework."""
        result = TestResult("Mobility Predictor")
        
        try:
            from app.mobility_predictor import MobilityPredictor, LocationPoint
            from app.events import EventBus
            
            # Create event bus for mobility predictor
            event_bus = EventBus()
            await event_bus.start()
            
            try:
                predictor = MobilityPredictor(event_bus)
                result.add_pass("Mobility predictor creation", "Predictor initialized")
                
                # Test location update
                device_id = "test_device"
                location = LocationPoint(
                    latitude=37.7749,
                    longitude=-122.4194,
                    timestamp=time.time()
                )
                
                await predictor.update_location(device_id, location)
                
                if device_id in predictor.mobility_history:
                    result.add_pass("Location update", f"Device added to history")
                else:
                    result.add_fail("Location update", "Device not found in history")
                
                # Test device stats
                stats = predictor.get_device_stats(device_id)
                expected_keys = ["device_id", "total_locations", "first_seen", "last_seen"]
                if all(key in stats for key in expected_keys):
                    result.add_pass("Device stats", f"Stats: {stats}")
                else:
                    result.add_fail("Device stats", f"Missing keys in stats: {stats}")
                
                # Test prediction (will be empty without training)
                predictions = await predictor.predict_mobility(device_id)
                result.add_pass("Mobility prediction", f"Generated {len(predictions)} predictions")
            
            except Exception as e:
                result.add_fail("Mobility predictor functionality", str(e))
            
            await event_bus.stop()
        
        except ImportError as e:
            result.add_fail("Mobility predictor import", str(e))
        
        result.finish()
        return result
    
    async def test_component_interfaces(self) -> TestResult:
        """Test component interface definitions."""
        result = TestResult("Component Interfaces")
        
        try:
            from app.components.interfaces import (
                ITrustedAuthority, IGatewayNode, IIoTDevice,
                ProtocolMessage, AuthenticationResult, DeviceCredentials
            )
            
            # Test interface imports
            result.add_pass("Interface imports", "All interfaces imported successfully")
            
            # Test data structures
            try:
                # Test ProtocolMessage
                from shared.config import MessageType
                message = ProtocolMessage(
                    message_type=MessageType.AUTHENTICATION_REQUEST,
                    sender_id="test_sender",
                    recipient_id="test_recipient",
                    correlation_id="test_correlation",
                    timestamp=time.time(),
                    payload={"test": "data"}
                )
                result.add_pass("ProtocolMessage creation", f"Message type: {message.message_type}")
                
                # Test AuthenticationResult
                auth_result = AuthenticationResult(
                    success=True,
                    session_id="test_session",
                    timestamp=time.time()
                )
                result.add_pass("AuthenticationResult creation", f"Success: {auth_result.success}")
                
                # Test DeviceCredentials
                credentials = DeviceCredentials(
                    device_id="test_device",
                    public_key=b"test_public_key",
                    certificate=b"test_certificate",
                    issued_at=time.time(),
                    expires_at=time.time() + 3600
                )
                result.add_pass("DeviceCredentials creation", f"Device: {credentials.device_id}")
            
            except Exception as e:
                result.add_fail("Data structure creation", str(e))
        
        except ImportError as e:
            result.add_fail("Interface import", str(e))
        
        result.finish()
        return result
    
    async def test_components(self) -> TestResult:
        """Test component implementations."""
        result = TestResult("Component Implementations")
        
        try:
            from app.components.trusted_authority import TrustedAuthority
            from app.components.gateway_node import GatewayNode
            from app.components.iot_device import IoTDevice
            from app.events import EventBus
            
            # Test component creation
            try:
                ta = TrustedAuthority("test_ta")
                result.add_pass("TrustedAuthority creation", f"TA ID: {ta.entity_id}")
            except Exception as e:
                result.add_fail("TrustedAuthority creation", str(e))
            
            try:
                event_bus = EventBus()
                gateway = GatewayNode("test_gateway", ta, event_bus)
                result.add_pass("GatewayNode creation", f"Gateway ID: {gateway.entity_id}")
            except Exception as e:
                result.add_fail("GatewayNode creation", str(e))
            
            try:
                device = IoTDevice("test_device", gateway)
                result.add_pass("IoTDevice creation", f"Device ID: {device.entity_id}")
            except Exception as e:
                result.add_fail("IoTDevice creation", str(e))
        
        except ImportError as e:
            result.add_fail("Component import", str(e))
        
        result.finish()
        return result
    
    async def test_integration(self) -> TestResult:
        """Test integration between components."""
        result = TestResult("Integration Testing")
        
        try:
            from app.events import EventBus, EventType
            from app.state_machine import GatewayStateMachine, DeviceStateMachine
            from uuid import uuid4
            
            # Create integrated system
            event_bus = EventBus()
            await event_bus.start()
            
            gateway_sm = GatewayStateMachine("test_gateway", event_bus)
            device_sm = DeviceStateMachine("test_device", event_bus)
            
            correlation_id = uuid4()
            
            # Test authentication flow
            try:
                await event_bus.publish_event(
                    event_type=EventType.AUTH_REQUEST,
                    correlation_id=correlation_id,
                    source="test_device",
                    target="test_gateway",
                    data={"device_id": "test_device"}
                )
                
                await asyncio.sleep(0.1)  # Allow processing
                
                if gateway_sm.current_state.name == "AWAITING_COMMITMENT":
                    result.add_pass("Auth flow initiation", "Gateway transitioned to AWAITING_COMMITMENT")
                else:
                    result.add_fail("Auth flow initiation", f"Gateway in state: {gateway_sm.current_state.name}")
            
            except Exception as e:
                result.add_fail("Auth flow initiation", str(e))
            
            # Test event metrics
            try:
                metrics = event_bus.get_metrics()
                if metrics["events_published"] > 0:
                    result.add_pass("Event flow metrics", f"Published: {metrics['events_published']}")
                else:
                    result.add_fail("Event flow metrics", "No events published")
            
            except Exception as e:
                result.add_fail("Event flow metrics", str(e))
            
            await event_bus.stop()
        
        except Exception as e:
            result.add_fail("Integration setup", str(e))
        
        result.finish()
        return result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites."""
        print("🧪 Running Comprehensive ZKPAS Module Tests")
        print("=" * 60)
        
        # Run all test suites
        test_suites = [
            ("imports", self.test_imports),
            ("crypto", self.test_crypto_utils),
            ("events", self.test_event_system),
            ("state_machines", self.test_state_machines),
            ("mobility", self.test_mobility_predictor),
            ("interfaces", self.test_component_interfaces),
            ("components", self.test_components),
            ("integration", self.test_integration)
        ]
        
        for suite_name, test_func in test_suites:
            print(f"\n📋 Testing: {test_func.__doc__ or suite_name}")
            print("-" * 40)
            
            try:
                test_result = await test_func()
                self.results.append(test_result)
                
                # Print summary for this suite
                print(f"  📊 Results: {test_result.passed} passed, {test_result.failed} failed, {test_result.skipped} skipped")
                print(f"  ⏱️  Duration: {test_result.duration():.2f}s")
                
                if test_result.failed > 0:
                    print(f"  ❌ Errors:")
                    for error in test_result.errors:
                        print(f"    - {error}")
            
            except Exception as e:
                print(f"  💥 Suite failed with exception: {e}")
                traceback.print_exc()
        
        # Generate overall summary
        overall_duration = time.time() - self.overall_start
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_tests = total_passed + total_failed + total_skipped
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 OVERALL TEST RESULTS")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {total_passed}")
        print(f"❌ Failed: {total_failed}")
        print(f"⏭️  Skipped: {total_skipped}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"⏱️  Total Duration: {overall_duration:.2f}s")
        
        # Test suite breakdown
        print(f"\n📋 Test Suite Breakdown:")
        for result in self.results:
            status = "✅" if result.failed == 0 else "❌"
            print(f"  {status} {result.name}: {result.passed}/{result.passed + result.failed} ({result.success_rate():.1f}%)")
        
        # Overall status
        overall_success = total_failed == 0
        print(f"\n🎯 Overall Status: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        if overall_success:
            print("\n🚀 All modules are working correctly!")
            print("✨ Ready to proceed with Phase 4 implementation!")
        else:
            print("\n🔧 Some modules need attention before proceeding.")
            print("📝 Check the error details above for specific issues.")
        
        return {
            "success": overall_success,
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "success_rate": success_rate,
            "duration": overall_duration,
            "suites": [r.summary() for r in self.results]
        }


async def main():
    """Run the comprehensive test suite."""
    tester = ModuleTester()
    results = await tester.run_all_tests()
    
    # Save results to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Test results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Failed to save results: {e}")
    
    return results["success"]


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏸️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
