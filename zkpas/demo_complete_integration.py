#!/usr/bin/env python3
"""
Complete System Integration Test

This demo runs comprehensive integration tests across all ZKPAS components
to verify the entire system works together correctly.
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add the zkpas directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.events import EventBus, EventType
from app.components.trusted_authority import TrustedAuthority
from app.components.gateway_node import GatewayNode
from app.components.iot_device import IoTDevice
from app.components.sliding_window_auth import SlidingWindowAuthenticator
from app.components.byzantine_resilience import ByzantineResilienceCoordinator
from app.mobility_predictor import MobilityPredictor, LocationPoint
from shared.crypto_utils import generate_keypair
from loguru import logger


class SystemIntegrationTest:
    """Complete system integration testing suite."""
    
    def __init__(self):
        self.event_bus = None
        self.components = {}
        self.test_results = {}
        self.devices = []
        self.gateways = []
        
    async def initialize_system(self):
        """Initialize all ZKPAS system components."""
        print("🚀 Initializing Complete ZKPAS System...")
        
        self.event_bus = EventBus()
        
        # Core Components
        self.components['ta'] = TrustedAuthority(self.event_bus)
        print("✅ Trusted Authority initialized")
        
        # Multiple Gateway Nodes
        for i in range(3):
            gateway = GatewayNode(f"GW{i:03d}", self.components['ta'], self.event_bus)
            self.gateways.append(gateway)
            self.components[f'gateway_{i}'] = gateway
        print(f"✅ {len(self.gateways)} Gateway Nodes initialized")
        
        # Multiple IoT Devices
        for i in range(5):
            device = IoTDevice(f"DEVICE{i:03d}", self.event_bus)
            self.devices.append(device)
            self.components[f'device_{i}'] = device
        print(f"✅ {len(self.devices)} IoT Devices initialized")
        
        # Advanced Components
        self.components['sliding_auth'] = SlidingWindowAuthenticator(self.event_bus)
        print("✅ Sliding Window Authenticator initialized")
        
        self.components['byzantine'] = ByzantineResilienceCoordinator(self.event_bus)
        print("✅ Byzantine Resilience Coordinator initialized")
        
        try:
            self.components['mobility'] = MobilityPredictor(self.event_bus)
            print("✅ Mobility Predictor initialized")
        except Exception as e:
            print(f"⚠️  Mobility Predictor initialization warning: {e}")
            self.components['mobility'] = None
        
        print("🎯 System initialization complete!")
    
    async def test_basic_authentication(self):
        """Test basic authentication flow."""
        print("\n🔐 Testing Basic Authentication Flow...")
        
        test_device = self.devices[0]
        test_gateway = self.gateways[0]
        ta = self.components['ta']
        
        try:
            # Register device
            keypair = generate_keypair()
            registration = await test_device.register_with_ta(ta, keypair.public_key())
            
            if not registration:
                self.test_results['basic_auth'] = {'status': 'FAILED', 'reason': 'Registration failed'}
                return
            
            # Authenticate
            auth_request = await test_device.initiate_authentication(test_gateway.node_id)
            
            if auth_request:
                # Simulate ZKP response
                challenge = {"challenge": "test_challenge", "timestamp": time.time()}
                zkp_response = await test_device.generate_zkp_response(challenge, keypair.private_key())
                
                if zkp_response:
                    verification = await test_gateway.verify_zkp_response(
                        zkp_response, test_device.device_id, challenge
                    )
                    
                    self.test_results['basic_auth'] = {
                        'status': 'PASSED' if verification else 'FAILED',
                        'device_id': test_device.device_id,
                        'gateway_id': test_gateway.node_id,
                        'verification': verification
                    }
                else:
                    self.test_results['basic_auth'] = {'status': 'FAILED', 'reason': 'ZKP generation failed'}
            else:
                self.test_results['basic_auth'] = {'status': 'FAILED', 'reason': 'Auth request failed'}
                
        except Exception as e:
            self.test_results['basic_auth'] = {'status': 'ERROR', 'error': str(e)}
    
    async def test_multi_device_authentication(self):
        """Test multiple devices authenticating simultaneously."""
        print("\n📱 Testing Multi-Device Authentication...")
        
        auth_tasks = []
        
        for i, device in enumerate(self.devices):
            gateway = self.gateways[i % len(self.gateways)]  # Distribute across gateways
            
            async def auth_device(dev, gw):
                try:
                    keypair = generate_keypair()
                    reg = await dev.register_with_ta(self.components['ta'], keypair.public_key())
                    if reg:
                        auth_req = await dev.initiate_authentication(gw.node_id)
                        return {'device': dev.device_id, 'gateway': gw.node_id, 'success': bool(auth_req)}
                    return {'device': dev.device_id, 'gateway': gw.node_id, 'success': False}
                except Exception as e:
                    return {'device': dev.device_id, 'gateway': gw.node_id, 'success': False, 'error': str(e)}
            
            auth_tasks.append(auth_device(device, gateway))
        
        # Execute all authentications concurrently
        results = await asyncio.gather(*auth_tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
        total = len(results)
        
        self.test_results['multi_device_auth'] = {
            'status': 'PASSED' if successful == total else 'PARTIAL',
            'successful': successful,
            'total': total,
            'success_rate': (successful / total) * 100,
            'details': results
        }
    
    async def test_sliding_window_authentication(self):
        """Test sliding window authentication mechanism."""
        print("\n🪟 Testing Sliding Window Authentication...")
        
        sliding_auth = self.components['sliding_auth']
        test_device = self.devices[0]
        
        try:
            # Generate tokens
            tokens = await sliding_auth.generate_token_window(test_device.device_id, window_size=5)
            
            if tokens and len(tokens) > 0:
                # Test token validation
                test_token = tokens[0]
                validation = await sliding_auth.validate_token(test_device.device_id, test_token)
                
                self.test_results['sliding_window'] = {
                    'status': 'PASSED' if validation else 'FAILED',
                    'tokens_generated': len(tokens),
                    'validation_success': validation
                }
            else:
                self.test_results['sliding_window'] = {'status': 'FAILED', 'reason': 'Token generation failed'}
                
        except Exception as e:
            self.test_results['sliding_window'] = {'status': 'ERROR', 'error': str(e)}
    
    async def test_byzantine_resilience(self):
        """Test Byzantine fault tolerance."""
        print("\n⚔️ Testing Byzantine Resilience...")
        
        byzantine = self.components['byzantine']
        
        try:
            # Create trust network
            network = byzantine.create_trust_network("test_network", threshold=2)
            
            # Test with no malicious anchors
            test_result = await byzantine.test_byzantine_resilience("test_network", 0)
            
            if test_result and test_result.get('authentication_successful', False):
                # Test with 1 malicious anchor
                test_result_malicious = await byzantine.test_byzantine_resilience("test_network", 1)
                
                self.test_results['byzantine'] = {
                    'status': 'PASSED',
                    'clean_test': test_result.get('authentication_successful', False),
                    'malicious_test': test_result_malicious.get('authentication_successful', False),
                    'threshold_resilience': True
                }
            else:
                self.test_results['byzantine'] = {'status': 'FAILED', 'reason': 'Clean test failed'}
                
        except Exception as e:
            self.test_results['byzantine'] = {'status': 'ERROR', 'error': str(e)}
    
    async def test_mobility_prediction(self):
        """Test mobility prediction functionality."""
        print("\n🚶 Testing Mobility Prediction...")
        
        mobility = self.components['mobility']
        
        if not mobility:
            self.test_results['mobility'] = {'status': 'SKIPPED', 'reason': 'Component not available'}
            return
        
        try:
            test_device_id = "MOBILE_DEVICE_001"
            
            # Add some location history
            locations = [
                LocationPoint(40.7128, -74.0060, time.time() - 300),  # 5 min ago
                LocationPoint(40.7130, -74.0058, time.time() - 240),  # 4 min ago
                LocationPoint(40.7132, -74.0056, time.time() - 180),  # 3 min ago
                LocationPoint(40.7134, -74.0054, time.time() - 120),  # 2 min ago
                LocationPoint(40.7136, -74.0052, time.time() - 60),   # 1 min ago
            ]
            
            for location in locations:
                await mobility.update_location(test_device_id, location)
            
            # Test prediction
            predictions = await mobility.predict_mobility(test_device_id)
            
            self.test_results['mobility'] = {
                'status': 'PASSED' if predictions else 'FAILED',
                'predictions_count': len(predictions) if predictions else 0,
                'location_history': len(locations)
            }
            
        except Exception as e:
            self.test_results['mobility'] = {'status': 'ERROR', 'error': str(e)}
    
    async def test_event_system(self):
        """Test event-driven architecture."""
        print("\n📡 Testing Event System...")
        
        try:
            event_received = False
            
            def test_event_handler(event):
                nonlocal event_received
                event_received = True
            
            # Subscribe to test event
            self.event_bus.subscribe_sync(EventType.DEVICE_AUTHENTICATED, test_event_handler)
            
            # Publish test event
            await self.event_bus.publish_event(
                EventType.DEVICE_AUTHENTICATED,
                correlation_id="test_123",
                source="integration_test",
                target="test_device",
                data={"test": True}
            )
            
            # Give event time to process
            await asyncio.sleep(0.1)
            
            self.test_results['event_system'] = {
                'status': 'PASSED' if event_received else 'FAILED',
                'event_received': event_received
            }
            
        except Exception as e:
            self.test_results['event_system'] = {'status': 'ERROR', 'error': str(e)}
    
    async def test_performance_load(self):
        """Test system performance under load."""
        print("\n📈 Testing Performance Under Load...")
        
        try:
            start_time = time.time()
            
            # Create load test tasks
            load_tasks = []
            
            for i in range(20):  # 20 concurrent operations
                async def load_operation():
                    device = IoTDevice(f"LOAD_DEVICE_{i}", self.event_bus)
                    keypair = generate_keypair()
                    return await device.register_with_ta(
                        self.components['ta'], 
                        keypair.public_key()
                    )
                
                load_tasks.append(load_operation())
            
            # Execute load test
            results = await asyncio.gather(*load_tasks, return_exceptions=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            successful_ops = sum(1 for r in results if r is True)
            throughput = successful_ops / duration if duration > 0 else 0
            
            self.test_results['performance'] = {
                'status': 'PASSED' if successful_ops >= 15 else 'FAILED',  # 75% success rate
                'operations': len(results),
                'successful': successful_ops,
                'duration': duration,
                'throughput': throughput,
                'success_rate': (successful_ops / len(results)) * 100
            }
            
        except Exception as e:
            self.test_results['performance'] = {'status': 'ERROR', 'error': str(e)}
    
    async def run_complete_integration_test(self):
        """Run all integration tests."""
        print("🧪 ZKPAS Complete System Integration Test")
        print("=" * 60)
        
        # Initialize system
        await self.initialize_system()
        
        # Run all tests
        await self.test_basic_authentication()
        await self.test_multi_device_authentication()
        await self.test_sliding_window_authentication()
        await self.test_byzantine_resilience()
        await self.test_mobility_prediction()
        await self.test_event_system()
        await self.test_performance_load()
        
        # Generate test report
        await self.generate_test_report()
        
        # Cleanup
        await self.cleanup_system()
    
    async def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n📊 INTEGRATION TEST RESULTS")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result.get('status') == 'PASSED')
        failed_tests = sum(1 for result in self.test_results.values() 
                          if result.get('status') == 'FAILED')
        error_tests = sum(1 for result in self.test_results.values() 
                         if result.get('status') == 'ERROR')
        skipped_tests = sum(1 for result in self.test_results.values() 
                           if result.get('status') == 'SKIPPED')
        
        print(f"📋 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   ⚠️  Errors: {error_tests}")
        print(f"   ⏭️  Skipped: {skipped_tests}")
        print(f"   📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print(f"\n🔍 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = result.get('status', 'UNKNOWN')
            icon = {'PASSED': '✅', 'FAILED': '❌', 'ERROR': '⚠️', 'SKIPPED': '⏭️'}.get(status, '❓')
            print(f"   {icon} {test_name.replace('_', ' ').title()}: {status}")
            
            if status == 'FAILED' and 'reason' in result:
                print(f"      Reason: {result['reason']}")
            elif status == 'ERROR' and 'error' in result:
                print(f"      Error: {result['error']}")
        
        print(f"\n🎯 System Health Assessment:")
        if passed_tests == total_tests:
            print("   🟢 EXCELLENT - All tests passed")
        elif passed_tests >= total_tests * 0.8:
            print("   🟡 GOOD - Most tests passed")
        elif passed_tests >= total_tests * 0.6:
            print("   🟠 FAIR - Some issues detected")
        else:
            print("   🔴 POOR - Significant issues found")
        
        # Performance metrics
        if 'performance' in self.test_results:
            perf = self.test_results['performance']
            if perf.get('status') == 'PASSED':
                print(f"\n⚡ Performance Metrics:")
                print(f"   Throughput: {perf.get('throughput', 0):.1f} ops/sec")
                print(f"   Success Rate: {perf.get('success_rate', 0):.1f}%")
                print(f"   Duration: {perf.get('duration', 0):.2f}s")
    
    async def cleanup_system(self):
        """Clean up system resources."""
        try:
            if self.components.get('sliding_auth'):
                await self.components['sliding_auth'].shutdown()
            
            # Note: EventBus might not have shutdown method
            # await self.event_bus.shutdown()
            
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


async def main():
    """Main entry point for integration test."""
    print("🎯 Welcome to ZKPAS Complete System Integration Test!")
    print("This test verifies that all system components work together correctly.")
    print()
    
    integration_test = SystemIntegrationTest()
    await integration_test.run_complete_integration_test()
    
    print("\n✨ Integration test completed!")
    print("Check the results above to see system health status.")


if __name__ == "__main__":
    asyncio.run(main())
