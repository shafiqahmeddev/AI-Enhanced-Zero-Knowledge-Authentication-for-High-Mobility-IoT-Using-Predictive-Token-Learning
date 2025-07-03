"""
Test Suite for Phase 5: Advanced Authentication & Byzantine Resilience

This module contains comprehensive tests for sliding window authentication
and Byzantine fault tolerance features implemented in Phase 5.
"""

import asyncio
import pytest
import time
import uuid
from typing import Dict, List

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


@pytest.fixture
async def event_bus():
    """Create event bus for testing."""
    bus = EventBus()
    yield bus
    await bus.shutdown()


@pytest.fixture
async def sliding_window_auth(event_bus):
    """Create sliding window authenticator for testing."""
    auth = SlidingWindowAuthenticator(event_bus, window_duration=300)
    yield auth
    await auth.shutdown()


@pytest.fixture
async def trust_anchor_network(event_bus):
    """Create trust anchor network for testing."""
    network = TrustAnchorNetwork(event_bus, threshold=2)
    yield network


@pytest.fixture
async def byzantine_coordinator(event_bus):
    """Create Byzantine resilience coordinator for testing."""
    coordinator = ByzantineResilienceCoordinator(event_bus, default_threshold=2)
    yield coordinator


class TestSlidingWindowAuthentication:
    """Test cases for sliding window authentication."""
    
    async def test_create_authentication_window(self, sliding_window_auth):
        """Test creation of authentication windows."""
        device_id = "test_device_001"
        master_key = secure_hash(b"test_master_key")
        
        window = await sliding_window_auth.create_authentication_window(device_id, master_key)
        
        assert window.device_id == device_id
        assert window.master_key == master_key
        assert window.is_active()
        assert window.sequence_counter == 0
        assert len(window.valid_tokens) == 0
    
    async def test_generate_sliding_window_token(self, sliding_window_auth):
        """Test token generation."""
        device_id = "test_device_002"
        master_key = secure_hash(b"test_master_key_2")
        
        # Create window first
        window = await sliding_window_auth.create_authentication_window(device_id, master_key)
        
        # Generate token
        payload = {"test_data": "sample_payload", "timestamp": time.time()}
        token = await sliding_window_auth.generate_sliding_window_token(device_id, payload)
        
        assert token is not None
        assert token.device_id == device_id
        assert not token.is_expired()
        assert token.sequence_number == 0
        assert len(token.encrypted_payload) > 0
        assert len(token.nonce) > 0
        
        # Check window state
        assert window.sequence_counter == 1
        assert token.token_id in window.valid_tokens
    
    async def test_validate_sliding_window_token(self, sliding_window_auth):
        """Test token validation."""
        device_id = "test_device_003"
        master_key = secure_hash(b"test_master_key_3")
        
        # Create window and generate token
        await sliding_window_auth.create_authentication_window(device_id, master_key)
        payload = {"auth_data": "secure_payload"}
        token = await sliding_window_auth.generate_sliding_window_token(device_id, payload)
        
        # Validate token
        is_valid, decrypted_payload = await sliding_window_auth.validate_sliding_window_token(
            token.token_id, device_id
        )
        
        assert is_valid
        assert decrypted_payload == payload
    
    async def test_token_validation_failure_cases(self, sliding_window_auth):
        """Test various token validation failure scenarios."""
        device_id = "test_device_004"
        master_key = secure_hash(b"test_master_key_4")
        
        # Create window and generate token
        await sliding_window_auth.create_authentication_window(device_id, master_key)
        payload = {"auth_data": "secure_payload"}
        token = await sliding_window_auth.generate_sliding_window_token(device_id, payload)
        
        # Test 1: Non-existent token
        is_valid, _ = await sliding_window_auth.validate_sliding_window_token(
            "non_existent_token", device_id
        )
        assert not is_valid
        
        # Test 2: Wrong device ID
        is_valid, _ = await sliding_window_auth.validate_sliding_window_token(
            token.token_id, "wrong_device_id"
        )
        assert not is_valid
        
        # Test 3: Expired token (simulate by setting expiry in past)
        token.expiry_timestamp = time.time() - 1
        is_valid, _ = await sliding_window_auth.validate_sliding_window_token(
            token.token_id, device_id
        )
        assert not is_valid
    
    async def test_fallback_mode(self, sliding_window_auth):
        """Test fallback mode functionality."""
        reason = "Network connectivity issues"
        
        # Enable fallback mode
        await sliding_window_auth.enable_fallback_mode(reason)
        assert sliding_window_auth._fallback_mode
        assert sliding_window_auth._fallback_reason == reason
        
        # Disable fallback mode
        await sliding_window_auth.disable_fallback_mode()
        assert not sliding_window_auth._fallback_mode
        assert sliding_window_auth._fallback_reason == ""
    
    async def test_token_cleanup(self, sliding_window_auth):
        """Test automatic cleanup of expired tokens."""
        device_id = "test_device_005"
        master_key = secure_hash(b"test_master_key_5")
        
        # Create window and generate token with short lifetime
        sliding_window_auth._token_lifetime = 1  # 1 second
        await sliding_window_auth.create_authentication_window(device_id, master_key)
        token = await sliding_window_auth.generate_sliding_window_token(device_id, {})
        
        # Verify token exists
        assert token.token_id in sliding_window_auth._token_cache
        
        # Wait for expiry and trigger cleanup
        await asyncio.sleep(2)
        await sliding_window_auth._remove_token(token.token_id)
        
        # Verify token is removed
        assert token.token_id not in sliding_window_auth._token_cache
    
    async def test_statistics(self, sliding_window_auth):
        """Test statistics reporting."""
        device_id = "test_device_006"
        master_key = secure_hash(b"test_master_key_6")
        
        # Initial statistics
        stats = sliding_window_auth.get_statistics()
        assert stats["active_windows"] == 0
        assert stats["total_tokens"] == 0
        assert not stats["fallback_mode"]
        
        # Create window and token
        await sliding_window_auth.create_authentication_window(device_id, master_key)
        await sliding_window_auth.generate_sliding_window_token(device_id, {})
        
        # Check updated statistics
        stats = sliding_window_auth.get_statistics()
        assert stats["active_windows"] == 1
        assert stats["total_tokens"] == 1


class TestByzantineFaultTolerance:
    """Test cases for Byzantine fault tolerance."""
    
    async def test_honest_trust_anchor(self, event_bus):
        """Test honest trust anchor behavior."""
        anchor = TrustAnchor("honest_anchor_1", event_bus)
        
        assert anchor.anchor_id == "honest_anchor_1"
        assert anchor.public_key is not None
        
        # Test signature share generation
        message_hash = secure_hash(b"test_message")
        request_id = str(uuid.uuid4())
        share = await anchor.generate_signature_share(message_hash, request_id)
        
        assert share.anchor_id == "honest_anchor_1"
        assert share.is_valid
        assert len(share.signature_share) > 0
        assert share.message_hash == message_hash
        
        # Test signature verification
        is_valid = await anchor.verify_signature_share(share)
        assert is_valid
    
    async def test_malicious_trust_anchor(self, event_bus):
        """Test malicious trust anchor behavior."""
        anchor = MaliciousTrustAnchor("malicious_anchor_1", event_bus, "invalid_signature")
        
        assert anchor.anchor_id == "malicious_anchor_1"
        assert anchor.public_key is not None
        
        # Test malicious signature share generation
        message_hash = secure_hash(b"test_message")
        request_id = str(uuid.uuid4())
        share = await anchor.generate_signature_share(message_hash, request_id)
        
        assert share.anchor_id == "malicious_anchor_1"
        # Note: Malicious anchor claims validity but signature is actually invalid
        assert share.is_valid  # Claims to be valid
        
        # Malicious anchor always claims shares are valid
        fake_share = SignatureShare(
            anchor_id="test",
            signature_share=b"fake",
            message_hash=message_hash,
            timestamp=time.time()
        )
        is_valid = await anchor.verify_signature_share(fake_share)
        assert is_valid  # Malicious behavior: always claims valid
    
    async def test_trust_anchor_network_setup(self, trust_anchor_network, event_bus):
        """Test trust anchor network setup."""
        network = trust_anchor_network
        
        # Add honest anchors
        for i in range(3):
            anchor = TrustAnchor(f"honest_anchor_{i}", event_bus)
            network.add_trust_anchor(anchor)
        
        # Add malicious anchor
        malicious_anchor = MaliciousTrustAnchor("malicious_anchor_1", event_bus)
        network.add_trust_anchor(malicious_anchor)
        
        status = network.get_network_status()
        assert status["total_anchors"] == 4
        assert status["honest_anchors"] == 3
        assert status["malicious_anchors"] == 1
        assert status["threshold"] == 2
        assert status["byzantine_resilient"]  # 3 honest >= 2 threshold
    
    async def test_cross_domain_authentication_success(self, trust_anchor_network, event_bus):
        """Test successful cross-domain authentication."""
        network = trust_anchor_network
        
        # Add sufficient honest anchors
        for i in range(3):
            anchor = TrustAnchor(f"honest_anchor_{i}", event_bus)
            network.add_trust_anchor(anchor)
        
        # Request cross-domain authentication
        message = b"cross_domain_auth_request"
        result = await network.request_cross_domain_authentication(
            source_domain="domain_a",
            target_domain="domain_b",
            device_id="test_device",
            message=message
        )
        
        assert result is not None
        assert result.threshold_met
        assert len(result.participating_anchors) >= 2
        assert len(result.aggregated_signature) > 0
    
    async def test_cross_domain_authentication_with_malicious_anchors(self, trust_anchor_network, event_bus):
        """Test cross-domain authentication resilience against malicious anchors."""
        network = trust_anchor_network
        
        # Add honest anchors (meets threshold)
        for i in range(2):
            anchor = TrustAnchor(f"honest_anchor_{i}", event_bus)
            network.add_trust_anchor(anchor)
        
        # Add malicious anchors (less than honest)
        malicious_anchor = MaliciousTrustAnchor("malicious_anchor_1", event_bus, "invalid_signature")
        network.add_trust_anchor(malicious_anchor)
        
        # Request should succeed despite malicious anchor
        message = b"cross_domain_auth_with_malicious"
        result = await network.request_cross_domain_authentication(
            source_domain="domain_a",
            target_domain="domain_b",
            device_id="test_device",
            message=message
        )
        
        assert result is not None
        assert result.threshold_met
        # Should only include honest anchors in final signature
        assert len(result.participating_anchors) == 2
        assert "malicious_anchor_1" not in result.participating_anchors
    
    async def test_cross_domain_authentication_failure(self, trust_anchor_network, event_bus):
        """Test cross-domain authentication failure when threshold not met."""
        network = trust_anchor_network
        
        # Add insufficient honest anchors
        honest_anchor = TrustAnchor("honest_anchor_1", event_bus)
        network.add_trust_anchor(honest_anchor)
        
        # Add malicious anchors
        for i in range(2):
            malicious_anchor = MaliciousTrustAnchor(f"malicious_anchor_{i}", event_bus)
            network.add_trust_anchor(malicious_anchor)
        
        # Request should fail due to insufficient honest anchors
        message = b"cross_domain_auth_insufficient"
        result = await network.request_cross_domain_authentication(
            source_domain="domain_a",
            target_domain="domain_b",
            device_id="test_device",
            message=message
        )
        
        assert result is None  # Should fail
    
    async def test_byzantine_resilience_coordinator(self, byzantine_coordinator, event_bus):
        """Test Byzantine resilience coordinator functionality."""
        coordinator = byzantine_coordinator
        
        # Create trust network
        network = coordinator.create_trust_network("test_network", threshold=2)
        assert network is not None
        
        # Add anchors to network
        for i in range(3):
            anchor = TrustAnchor(f"anchor_{i}", event_bus)
            network.add_trust_anchor(anchor)
        
        # Test Byzantine resilience
        test_results = await coordinator.test_byzantine_resilience("test_network", num_malicious=1)
        
        assert test_results["network_id"] == "test_network"
        assert test_results["num_malicious_anchors"] == 1
        assert test_results["authentication_successful"]  # Should succeed with 3 honest + 1 malicious
        assert test_results["threshold_met"]
        
        # Test system status
        status = coordinator.get_system_status()
        assert status["total_networks"] == 1
        assert status["default_threshold"] == 2
        assert "test_network" in status["networks"]


class TestPhase5Integration:
    """Integration tests for Phase 5 components."""
    
    async def test_sliding_window_with_byzantine_resilience(self, event_bus):
        """Test integration between sliding window auth and Byzantine resilience."""
        # Setup sliding window authenticator
        sliding_auth = SlidingWindowAuthenticator(event_bus)
        
        # Setup Byzantine coordinator
        coordinator = ByzantineResilienceCoordinator(event_bus)
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
        
        # Cleanup
        await sliding_auth.shutdown()
    
    async def test_event_bus_integration(self, event_bus):
        """Test event bus integration with Phase 5 components."""
        events_received = []
        
        async def event_handler(event):
            events_received.append(event)
        
        # Subscribe to Phase 5 events
        event_bus.subscribe(EventType.WINDOW_CREATED, event_handler)
        event_bus.subscribe(EventType.TOKEN_GENERATED, event_handler)
        event_bus.subscribe(EventType.CROSS_DOMAIN_AUTH_SUCCESS, event_handler)
        
        # Setup components
        sliding_auth = SlidingWindowAuthenticator(event_bus)
        coordinator = ByzantineResilienceCoordinator(event_bus)
        network = coordinator.create_trust_network("event_test_network")
        
        # Add anchors
        for i in range(2):
            anchor = TrustAnchor(f"event_anchor_{i}", event_bus)
            network.add_trust_anchor(anchor)
        
        # Trigger events
        device_id = "event_test_device"
        master_key = secure_hash(b"event_test_key")
        
        await sliding_auth.create_authentication_window(device_id, master_key)
        await sliding_auth.generate_sliding_window_token(device_id, {})
        
        await network.request_cross_domain_authentication(
            source_domain="domain_a",
            target_domain="domain_b",
            device_id=device_id,
            message=b"event_test_message"
        )
        
        # Give events time to propagate
        await asyncio.sleep(0.1)
        
        # Check events were received
        assert len(events_received) >= 3
        event_types = [event.event_type for event in events_received]
        assert EventType.WINDOW_CREATED in event_types
        assert EventType.TOKEN_GENERATED in event_types
        assert EventType.CROSS_DOMAIN_AUTH_SUCCESS in event_types
        
        # Cleanup
        await sliding_auth.shutdown()


@pytest.mark.asyncio
async def test_phase5_comprehensive_scenario():
    """Comprehensive scenario test for Phase 5 functionality."""
    event_bus = EventBus()
    
    try:
        # Initialize components
        sliding_auth = SlidingWindowAuthenticator(event_bus, window_duration=600)
        coordinator = ByzantineResilienceCoordinator(event_bus, default_threshold=3)
        
        # Create multiple trust networks for different domains
        network_a = coordinator.create_trust_network("domain_a_network", threshold=3)
        network_b = coordinator.create_trust_network("domain_b_network", threshold=2)
        
        # Add trust anchors to networks
        for i in range(4):
            anchor_a = TrustAnchor(f"domain_a_anchor_{i}", event_bus)
            network_a.add_trust_anchor(anchor_a)
            
            if i < 3:  # Only 3 anchors for domain B
                anchor_b = TrustAnchor(f"domain_b_anchor_{i}", event_bus)
                network_b.add_trust_anchor(anchor_b)
        
        # Add one malicious anchor to test resilience
        malicious_anchor = MaliciousTrustAnchor("malicious_anchor", event_bus, "invalid_signature")
        network_a.add_trust_anchor(malicious_anchor)
        
        # Simulate multiple devices requesting authentication
        devices = ["device_001", "device_002", "device_003"]
        tokens = {}
        
        for device_id in devices:
            master_key = secure_hash(f"master_key_{device_id}".encode())
            
            # Create authentication window
            window = await sliding_auth.create_authentication_window(device_id, master_key)
            
            # Generate multiple tokens for each device
            for i in range(3):
                payload = {
                    "device_id": device_id,
                    "session": i,
                    "cross_domain_target": "domain_b" if i % 2 == 0 else "domain_a"
                }
                token = await sliding_auth.generate_sliding_window_token(device_id, payload)
                tokens[f"{device_id}_token_{i}"] = token
        
        # Perform cross-domain authentications
        successful_auths = 0
        failed_auths = 0
        
        for token_key, token in tokens.items():
            # Validate token first
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
        
        # Verify results
        assert successful_auths > 0
        assert successful_auths > failed_auths  # Most should succeed
        
        # Test system statistics
        sliding_stats = sliding_auth.get_statistics()
        assert sliding_stats["active_windows"] == len(devices)
        assert sliding_stats["total_tokens"] == len(devices) * 3
        
        coordinator_status = coordinator.get_system_status()
        assert coordinator_status["total_networks"] == 2
        
        # Test Byzantine resilience
        test_results_a = await coordinator.test_byzantine_resilience("domain_a_network", 2)
        assert test_results_a["authentication_successful"]  # Should succeed with 4 honest + 2 malicious
        
        test_results_b = await coordinator.test_byzantine_resilience("domain_b_network", 2)
        assert test_results_b["authentication_successful"]  # Should succeed with 3 honest + 2 malicious
        
        print(f"✅ Phase 5 Comprehensive Test Results:")
        print(f"   Successful authentications: {successful_auths}")
        print(f"   Failed authentications: {failed_auths}")
        print(f"   Active windows: {sliding_stats['active_windows']}")
        print(f"   Total tokens: {sliding_stats['total_tokens']}")
        print(f"   Byzantine resilience: PASSED")
        
    finally:
        # Cleanup
        await sliding_auth.shutdown()
        await event_bus.shutdown()


if __name__ == "__main__":
    # Run the comprehensive test
    asyncio.run(test_phase5_comprehensive_scenario())