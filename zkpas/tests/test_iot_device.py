"""
Tests for IoT device component.
"""
import pytest
from unittest.mock import Mock, patch
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.components.iot_device import IoTDevice
from app.components.interfaces import AuthenticationResult, ProtocolMessage, DeviceLocation
from shared.config import ProtocolState, MessageType
from shared.crypto_utils import generate_ecc_keypair


class TestIoTDevice:
    """Test suite for IoT device component."""
    
    @pytest.fixture
    async def device(self):
        """Create a test IoT device."""
        import time
        initial_location = DeviceLocation(
            latitude=37.7749,
            longitude=-122.4194,
            timestamp=time.time()
        )
        device = IoTDevice(device_id="test_device_001", initial_location=initial_location)
        await device.initialize()
        return device
    
    @pytest.fixture
    def mock_gateway(self):
        """Create a mock gateway for testing."""
        gateway = Mock()
        gateway.authenticate_device = Mock()
        return gateway

    @pytest.mark.asyncio
    async def test_device_initialization(self):
        """Test device initialization process."""
        import time
        initial_location = DeviceLocation(
            latitude=37.7749,
            longitude=-122.4194,
            timestamp=time.time()
        )
        device = IoTDevice(device_id="test_device_001", initial_location=initial_location)
        
        # Should not be initialized yet
        assert not device.is_initialized
        
        # Initialize device
        await device.initialize()
        
        # Should now be initialized
        assert device.is_initialized
        assert device.device_id == "test_device_001"
        assert device.private_key is not None
        assert device.public_key is not None
        assert len(device.mobility_history) == 1  # Should have initial location
        assert len(device.token_cache) == 0

    @pytest.mark.asyncio
    async def test_location_update(self, device):
        """Test location tracking functionality."""
        import time
        
        # Should start with initial location
        assert len(device.mobility_history) == 1
        
        # Add another location
        location1 = DeviceLocation(
            latitude=37.7849,
            longitude=-122.4094,
            timestamp=time.time()
        )
        await device.update_location(location1)
        
        assert len(device.mobility_history) == 2
        location = device.mobility_history[-1]  # Latest location
        assert location.latitude == 37.7849
        assert location.longitude == -122.4094
        assert location.timestamp is not None
        
        # Add another location
        location2 = DeviceLocation(
            latitude=37.7949,
            longitude=-122.3994,
            timestamp=time.time()
        )
        await device.update_location(location2)
        
        assert len(device.mobility_history) == 3
        
        # Test location history limit
        for i in range(99):  # Add 99 more (1 initial + 2 above + 99 = 102, but limit is 100)
            location = DeviceLocation(
                latitude=37.7749 + i * 0.001,
                longitude=-122.4194,
                timestamp=time.time() + i
            )
            await device.update_location(location)
        
        # Should maintain only the last 100 locations
        assert len(device.mobility_history) == 100

    @pytest.mark.asyncio
    async def test_authentication_initiation(self, device, mock_gateway):
        """Test authentication request initiation."""
        # Set up mock gateway response
        auth_result = AuthenticationResult(
            success=True,
            correlation_id="test_session_123",
            timestamp=1234567890.0,
            session_key=b"test_session_key"
        )
        mock_gateway.authenticate_device.return_value = auth_result
        
        # Initiate authentication
        response = await device.initiate_authentication("test_gateway")
        
        # Verify response
        assert response.success is True
        assert response.correlation_id == "test_session_123"
        assert response.session_key == b"test_session_key"
        
        # Verify gateway was called correctly
        mock_gateway.authenticate_device.assert_called_once()
        call_args = mock_gateway.authenticate_device.call_args[0]
        device_id = call_args[0]
        
        assert device_id == "test_device_001"

    @pytest.mark.asyncio
    async def test_sliding_window_authentication(self, device, mock_gateway):
        """Test sliding window token-based authentication."""
        # Set up mock gateway to return cached token
        auth_result = AuthenticationResult(
            success=True,
            correlation_id="cached_session_456",
            timestamp=1234567890.0,
            session_key=b"cached_session_key"
        )
        mock_gateway.authenticate_device.return_value = auth_result
        
        # First authentication should succeed
        response = await device.initiate_authentication("test_gateway")
        
        assert response.success is True
        assert response.correlation_id == "cached_session_456"
        assert response.session_key == b"cached_session_key"

    @pytest.mark.asyncio
    async def test_zkp_commitment_generation(self, device):
        """Test zero-knowledge proof commitment generation."""
        challenge = b"test_challenge_data"
        
        commitment = await device._generate_zkp_commitment(challenge)
        
        # Commitment should be bytes
        assert isinstance(commitment, bytes)
        assert len(commitment) > 0
        
        # Different challenges should produce different commitments
        commitment2 = await device._generate_zkp_commitment(b"different_challenge")
        assert commitment != commitment2

    @pytest.mark.asyncio
    async def test_zkp_response_generation(self, device):
        """Test zero-knowledge proof response generation."""
        challenge = b"test_challenge_data"
        commitment = await device._generate_zkp_commitment(challenge)
        
        response = await device._generate_zkp_response(challenge, commitment)
        
        # Response should be bytes
        assert isinstance(response, bytes)
        assert len(response) > 0
        
        # Different challenges should produce different responses
        commitment2 = await device._generate_zkp_commitment(b"different_challenge")
        response2 = await device._generate_zkp_response(b"different_challenge", commitment2)
        assert response != response2

    @pytest.mark.asyncio
    async def test_mobility_simulation(self, device):
        """Test mobility pattern simulation."""
        # Start mobility simulation
        simulation_task = asyncio.create_task(device.simulate_mobility())
        
        # Let it run for a short time
        await asyncio.sleep(0.1)
        
        # Should have generated some mobility data
        assert len(device.mobility_history) > 0
        
        # Stop simulation
        simulation_task.cancel()
        try:
            await simulation_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_token_cache_expiry(self, device):
        """Test token cache expiry mechanism."""
        from app.components.iot_device import CachedToken
        import time
        
        # Add an expired token
        expired_token = CachedToken(
            token="expired_token",
            timestamp=time.time() - 3700,  # More than 1 hour ago
            gateway_id="test_gateway"
        )
        device.token_cache["expired_session"] = expired_token
        
        # Add a valid token
        valid_token = CachedToken(
            token="valid_token",
            timestamp=time.time() - 1800,  # 30 minutes ago
            gateway_id="test_gateway"
        )
        device.token_cache["valid_session"] = valid_token
        
        # Clean expired tokens
        device._cleanup_expired_tokens()
        
        # Expired token should be removed, valid token should remain
        assert "expired_session" not in device.token_cache
        assert "valid_session" in device.token_cache

    @pytest.mark.asyncio
    async def test_concurrent_authentication(self, device, mock_gateway):
        """Test handling of concurrent authentication requests."""
        # Set up mock gateway responses
        auth_results = [
            AuthenticationResult(
                success=True,
                correlation_id=f"session_{i}",
                timestamp=1234567890.0 + i,
                session_key=f"session_key_{i}".encode()
            )
            for i in range(3)
        ]
        mock_gateway.authenticate_device.side_effect = auth_results
        
        # Start multiple authentication requests concurrently
        tasks = [
            device.initiate_authentication("test_gateway")
            for _ in range(3)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should complete successfully
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert response.success is True
            assert response.correlation_id == f"session_{i}"
            assert response.session_key == f"session_key_{i}".encode()

    @pytest.mark.asyncio
    async def test_error_handling(self, device, mock_gateway):
        """Test error handling in authentication flow."""
        # Mock gateway to raise exception
        mock_gateway.authenticate_device.side_effect = Exception("Network error")
        
        # Authentication should handle the error gracefully
        with pytest.raises(Exception) as exc_info:
            await device.initiate_authentication("test_gateway")
        
        assert "Network error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_device_state_consistency(self, device):
        """Test device state remains consistent during operations."""
        import time
        
        initial_device_id = device.device_id
        initial_public_key = device.public_key
        initial_location_count = len(device.mobility_history)  # Should be 1 (initial location)
        
        # Perform various operations
        location1 = DeviceLocation(
            latitude=37.7749,
            longitude=-122.4194,
            timestamp=time.time()
        )
        await device.update_location(location1)
        
        location2 = DeviceLocation(
            latitude=37.7849,
            longitude=-122.4094,
            timestamp=time.time()
        )
        await device.update_location(location2)
        
        # Core identity should remain unchanged
        assert device.device_id == initial_device_id
        assert device.public_key == initial_public_key
        assert device.is_initialized
        
        # State should be updated appropriately
        assert len(device.mobility_history) == initial_location_count + 2

    @pytest.mark.asyncio
    async def test_resource_cleanup(self, device):
        """Test proper resource cleanup."""
        import time
        
        # Add data to various caches
        location = DeviceLocation(
            latitude=37.7749,
            longitude=-122.4194,
            timestamp=time.time()
        )
        await device.update_location(location)
        
        # Cleanup should work without errors
        device._cleanup_expired_tokens()
        
        # Should still be functional after cleanup
        assert device.is_initialized
        
        location2 = DeviceLocation(
            latitude=37.7849,
            longitude=-122.4094,
            timestamp=time.time()
        )
        await device.update_location(location2)
