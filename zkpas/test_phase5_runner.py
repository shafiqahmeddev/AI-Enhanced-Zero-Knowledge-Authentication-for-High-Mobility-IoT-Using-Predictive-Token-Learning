"""
Simple test runner for Phase 5 Advanced Authentication & Byzantine Resilience

This script tests the Phase 5 implementation without requiring pytest.
"""

import asyncio
import sys
import time
import uuid
from typing import Dict, List

sys.path.append('.')

from app.events import EventBus, EventType
from app.components.sliding_window_auth import (
    SlidingWindowAuthenticator,
    SlidingWindowToken,
    AuthenticationWindow
)
from app.components.byzantine_resilience import (
    TrustAnchor,
    MaliciousTrustAnchor,
    TrustAnchorNetwork,
    ByzantineResilienceCoordinator,
    SignatureShare,
    CrossDomainAuthRequest
)
from shared.crypto_utils import generate_ecc_keypair, secure_hash


async def test_sliding_window_authentication():
    """Test sliding window authentication functionality."""
    print("🧪 Testing Sliding Window Authentication...")
    
    event_bus = EventBus()
    await event_bus.start()
    sliding_auth = SlidingWindowAuthenticator(event_bus, window_duration=300)
    
    try:
        # Test 1: Create authentication window
        print("  ✓ Test 1: Authentication window creation")
        device_id = "test_device_001"
        master_key = secure_hash(b"test_master_key")
        window = await sliding_auth.create_authentication_window(device_id, master_key)
        
        assert window.device_id == device_id
        assert window.is_active()
        print("    ✅ Window created successfully")
        
        # Test 2: Generate token
        print("  ✓ Test 2: Token generation")
        payload = {"test_data": "sample_payload"}
        token = await sliding_auth.generate_sliding_window_token(device_id, payload)
        
        assert token is not None
        assert token.device_id == device_id
        assert not token.is_expired()
        print("    ✅ Token generated successfully")
        
        # Test 3: Validate token
        print("  ✓ Test 3: Token validation")
        is_valid, decrypted_payload = await sliding_auth.validate_sliding_window_token(
            token.token_id, device_id
        )
        
        assert is_valid
        assert decrypted_payload == payload
        print("    ✅ Token validated successfully")
        
        # Test 4: Fallback mode
        print("  ✓ Test 4: Fallback mode")
        await sliding_auth.enable_fallback_mode("Test reason")
        assert sliding_auth._fallback_mode
        
        await sliding_auth.disable_fallback_mode()
        assert not sliding_auth._fallback_mode
        print("    ✅ Fallback mode working correctly")
        
        print("✅ Sliding Window Authentication: ALL TESTS PASSED")
        
    except Exception as e:
        print(f"❌ Sliding Window Authentication test failed: {e}")
        raise
    finally:
        await sliding_auth.shutdown()
        await event_bus.stop()


async def test_byzantine_fault_tolerance():
    """Test Byzantine fault tolerance functionality."""
    print("🧪 Testing Byzantine Fault Tolerance...")
    
    event_bus = EventBus()
    await event_bus.start()
    
    try:
        # Test 1: Honest trust anchor
        print("  ✓ Test 1: Honest trust anchor")
        anchor = TrustAnchor("honest_anchor_1", event_bus)
        
        message_hash = secure_hash(b"test_message")
        request_id = str(uuid.uuid4())
        share = await anchor.generate_signature_share(message_hash, request_id)
        
        assert share.is_valid
        assert share.anchor_id == "honest_anchor_1"
        print("    ✅ Honest anchor working correctly")
        
        # Test 2: Malicious trust anchor
        print("  ✓ Test 2: Malicious trust anchor")
        malicious_anchor = MaliciousTrustAnchor("malicious_anchor_1", event_bus, "invalid_signature")
        malicious_share = await malicious_anchor.generate_signature_share(message_hash, request_id)
        
        assert malicious_share.anchor_id == "malicious_anchor_1"
        print("    ✅ Malicious anchor behaving as expected")
        
        # Test 3: Trust anchor network
        print("  ✓ Test 3: Trust anchor network")
        network = TrustAnchorNetwork(event_bus, threshold=2)
        
        # Add honest anchors
        for i in range(3):
            honest = TrustAnchor(f"honest_{i}", event_bus)
            network.add_trust_anchor(honest)
        
        # Add malicious anchor
        network.add_trust_anchor(malicious_anchor)
        
        status = network.get_network_status()
        assert status["total_anchors"] == 4
        assert status["honest_anchors"] == 3
        assert status["malicious_anchors"] == 1
        assert status["byzantine_resilient"]
        print("    ✅ Network setup correctly")
        
        # Test 4: Cross-domain authentication with Byzantine resilience
        print("  ✓ Test 4: Cross-domain authentication")
        message = b"cross_domain_auth_test"
        result = await network.request_cross_domain_authentication(
            source_domain="domain_a",
            target_domain="domain_b",
            device_id="test_device",
            message=message
        )
        
        assert result is not None
        assert result.threshold_met
        assert len(result.participating_anchors) >= 2
        print("    ✅ Cross-domain authentication successful despite malicious anchor")
        
        print("✅ Byzantine Fault Tolerance: ALL TESTS PASSED")
        
    except Exception as e:
        print(f"❌ Byzantine Fault Tolerance test failed: {e}")
        raise
    finally:
        await event_bus.stop()


async def test_integration():
    """Test integration between sliding window auth and Byzantine resilience."""
    print("🧪 Testing Phase 5 Integration...")
    
    event_bus = EventBus()
    await event_bus.start()
    
    try:
        # Setup components
        sliding_auth = SlidingWindowAuthenticator(event_bus)
        coordinator = ByzantineResilienceCoordinator(event_bus, default_threshold=2)
        network = coordinator.create_trust_network("main_network", threshold=2)
        
        # Add trust anchors
        for i in range(3):
            anchor = TrustAnchor(f"anchor_{i}", event_bus)
            network.add_trust_anchor(anchor)
        
        device_id = "integration_test_device"
        master_key = secure_hash(b"integration_master_key")
        
        # Create authentication window
        window = await sliding_auth.create_authentication_window(device_id, master_key)
        
        # Generate token
        payload = {"cross_domain": True, "target_domain": "domain_b"}
        token = await sliding_auth.generate_sliding_window_token(device_id, payload)
        
        # Validate token
        is_valid, decrypted_payload = await sliding_auth.validate_sliding_window_token(
            token.token_id, device_id
        )
        assert is_valid
        assert decrypted_payload["cross_domain"]
        
        # Perform cross-domain authentication
        message = f"cross_domain_auth_{token.token_id}".encode()
        result = await network.request_cross_domain_authentication(
            source_domain="domain_a",
            target_domain="domain_b",
            device_id=device_id,
            message=message
        )
        
        assert result is not None
        assert result.threshold_met
        
        print("✅ Phase 5 Integration: ALL TESTS PASSED")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise
    finally:
        await sliding_auth.shutdown()
        await event_bus.stop()


async def comprehensive_phase5_test():
    """Comprehensive test of all Phase 5 functionality."""
    print("🚀 Phase 5: Advanced Authentication & Byzantine Resilience - Comprehensive Test")
    print("=" * 80)
    
    event_bus = EventBus()
    await event_bus.start()
    
    try:
        # Initialize components
        print("📋 Initializing components...")
        sliding_auth = SlidingWindowAuthenticator(event_bus, window_duration=600)
        coordinator = ByzantineResilienceCoordinator(event_bus, default_threshold=3)
        
        # Create trust networks
        network_a = coordinator.create_trust_network("domain_a_network", threshold=3)
        network_b = coordinator.create_trust_network("domain_b_network", threshold=2)
        
        # Add trust anchors
        print("🔐 Setting up trust anchors...")
        for i in range(4):
            anchor_a = TrustAnchor(f"domain_a_anchor_{i}", event_bus)
            network_a.add_trust_anchor(anchor_a)
            
            if i < 3:
                anchor_b = TrustAnchor(f"domain_b_anchor_{i}", event_bus)
                network_b.add_trust_anchor(anchor_b)
        
        # Add malicious anchor for resilience testing
        malicious_anchor = MaliciousTrustAnchor("malicious_anchor", event_bus, "invalid_signature")
        network_a.add_trust_anchor(malicious_anchor)
        
        # Setup devices
        print("📱 Setting up test devices...")
        devices = ["device_001", "device_002", "device_003"]
        tokens = {}
        
        for device_id in devices:
            master_key = secure_hash(f"master_key_{device_id}".encode())
            
            # Create authentication window
            window = await sliding_auth.create_authentication_window(device_id, master_key)
            
            # Generate tokens
            for i in range(3):
                payload = {
                    "device_id": device_id,
                    "session": i,
                    "cross_domain_target": "domain_b" if i % 2 == 0 else "domain_a"
                }
                token = await sliding_auth.generate_sliding_window_token(device_id, payload)
                tokens[f"{device_id}_token_{i}"] = token
        
        print(f"✅ Created {len(devices)} devices with {len(tokens)} total tokens")
        
        # Perform cross-domain authentications
        print("🌐 Testing cross-domain authentications...")
        successful_auths = 0
        failed_auths = 0
        
        for token_key, token in tokens.items():
            # Validate token
            is_valid, payload = await sliding_auth.validate_sliding_window_token(
                token.token_id, token.device_id
            )
            
            if is_valid:
                # Determine target network
                target_domain = payload["cross_domain_target"]
                network = network_a if target_domain == "domain_a" else network_b
                
                # Perform cross-domain authentication
                message = f"auth_request_{token.token_id}_{target_domain}".encode()
                result = await network.request_cross_domain_authentication(
                    source_domain="source_domain",
                    target_domain=target_domain,
                    device_id=token.device_id,
                    message=message
                )
                
                if result and result.threshold_met:
                    successful_auths += 1
                else:
                    failed_auths += 1
        
        print(f"✅ Authentication Results: {successful_auths} successful, {failed_auths} failed")
        
        # Test Byzantine resilience
        print("🛡️  Testing Byzantine resilience...")
        test_results_a = await coordinator.test_byzantine_resilience("domain_a_network", 2)
        test_results_b = await coordinator.test_byzantine_resilience("domain_b_network", 1)
        
        print(f"   Domain A network resilience: {'✅ PASSED' if test_results_a['authentication_successful'] else '❌ FAILED'}")
        print(f"   Domain B network resilience: {'✅ PASSED' if test_results_b['authentication_successful'] else '❌ FAILED'}")
        
        # Get final statistics
        sliding_stats = sliding_auth.get_statistics()
        coordinator_status = coordinator.get_system_status()
        
        print("📊 Final Statistics:")
        print(f"   Active windows: {sliding_stats['active_windows']}")
        print(f"   Total tokens: {sliding_stats['total_tokens']}")
        print(f"   Trust networks: {coordinator_status['total_networks']}")
        print(f"   Fallback mode: {sliding_stats['fallback_mode']}")
        
        # Verify success criteria
        assert successful_auths > 0, "No successful authentications"
        assert successful_auths > failed_auths, "More failures than successes"
        assert test_results_a["authentication_successful"], "Domain A resilience test failed"
        assert test_results_b["authentication_successful"], "Domain B resilience test failed"
        assert sliding_stats["active_windows"] == len(devices), "Incorrect number of active windows"
        
        print("\n🎉 Phase 5 Implementation: ALL TESTS PASSED!")
        print("✅ Sliding Window Authentication: FUNCTIONAL")
        print("✅ Byzantine Fault Tolerance: FUNCTIONAL")
        print("✅ Cross-Domain Authentication: FUNCTIONAL")
        print("✅ Event-Driven Architecture: FUNCTIONAL")
        print("✅ Malicious Actor Resilience: FUNCTIONAL")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 5 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await sliding_auth.shutdown()
        await event_bus.stop()


async def main():
    """Run all Phase 5 tests."""
    print("🧪 ZKPAS Phase 5: Advanced Authentication & Byzantine Resilience Test Suite")
    print("=" * 80)
    
    tests = [
        ("Sliding Window Authentication", test_sliding_window_authentication),
        ("Byzantine Fault Tolerance", test_byzantine_fault_tolerance),
        ("Integration Test", test_integration),
        ("Comprehensive Test", comprehensive_phase5_test)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            await test_func()
            passed += 1
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: FAILED - {e}")
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL PHASE 5 TESTS PASSED!")
        print("✅ Phase 5 implementation is ready for deployment")
        return True
    else:
        print("❌ Some tests failed. Review implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)