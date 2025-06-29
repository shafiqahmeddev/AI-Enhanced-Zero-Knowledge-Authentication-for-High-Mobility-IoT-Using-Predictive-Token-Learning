"""
Tests for the Gateway Node implementation.
"""

import pytest
import asyncio
from app.components.gateway_node import GatewayNode
from app.components.trusted_authority import TrustedAuthority
from shared.config import ProtocolState
from shared.crypto_utils import generate_ecc_keypair, serialize_public_key


class TestGatewayNode:
    """Test suite for Gateway Node component."""
    
    @pytest.fixture
    def ta(self):
        """Create a test Trusted Authority instance."""
        return TrustedAuthority("test_ta_001")
    
    @pytest.fixture
    def gateway(self, ta):
        """Create a test Gateway Node instance."""
        return GatewayNode("test_gateway_001", ta)
    
    @pytest.fixture
    def device_keypair(self):
        """Generate test device keypair."""
        private_key, public_key = generate_ecc_keypair()
        return private_key, public_key
    
    def test_gateway_initialization(self, gateway, ta):
        """Test gateway initialization."""
        assert gateway.entity_id == "test_gateway_001"
        assert gateway.public_key is not None
        assert len(gateway.public_key) == 64
        assert gateway.state == ProtocolState.IDLE
        assert gateway.is_degraded_mode is False
    
    @pytest.mark.asyncio
    async def test_device_authentication_success(self, gateway, ta, device_keypair):
        """Test successful device authentication."""
        _, public_key = device_keypair
        pub_key_bytes = serialize_public_key(public_key)
        
        # Register device with TA first
        await ta.register_device("device_001", pub_key_bytes)
        
        # Register gateway with TA
        await ta.register_gateway(gateway.entity_id, gateway.public_key)
        
        # Authenticate device
        result = await gateway.authenticate_device("device_001")
        
        assert result.success is True
        assert result.session_key is not None
        assert result.error_message is None
        
        # Check that device is in authenticated set
        authenticated = gateway.get_authenticated_devices()
        assert "device_001" in authenticated
    
    @pytest.mark.asyncio
    async def test_device_authentication_unregistered_device(self, gateway, ta):
        """Test authentication of unregistered device."""
        # Register gateway with TA
        await ta.register_gateway(gateway.entity_id, gateway.public_key)
        
        # Try to authenticate unregistered device
        result = await gateway.authenticate_device("unregistered_device")
        
        assert result.success is False
        assert result.session_key is None
        assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_degraded_mode_operations(self, gateway, ta, device_keypair):
        """Test gateway operations in degraded mode."""
        _, public_key = device_keypair
        pub_key_bytes = serialize_public_key(public_key)
        
        # First, register and authenticate device when TA is available
        await ta.register_device("device_001", pub_key_bytes)
        await ta.register_gateway(gateway.entity_id, gateway.public_key)
        
        # Successful authentication (creates cache entry)
        result1 = await gateway.authenticate_device("device_001")
        assert result1.success is True
        
        # Now make TA unavailable
        ta.set_availability(False)
        
        # Gateway should enter degraded mode
        await gateway._check_degraded_mode()
        assert gateway.is_degraded_mode is True
        assert gateway.state == ProtocolState.DEGRADED_MODE
        
        # Authentication should still work using cache
        result2 = await gateway.authenticate_device("device_001")
        assert result2.success is True
        
        # But new device authentication should fail
        result3 = await gateway.authenticate_device("new_device")
        assert result3.success is False
        assert "not cached" in result3.error_message
    
    @pytest.mark.asyncio
    async def test_exit_degraded_mode(self, gateway, ta, device_keypair):
        """Test exiting degraded mode."""
        # Enter degraded mode
        await gateway.enter_degraded_mode("Test reason")
        assert gateway.is_degraded_mode is True
        
        # Should not be able to exit while TA is unavailable
        ta.set_availability(False)
        result1 = await gateway.exit_degraded_mode()
        assert result1 is False
        assert gateway.is_degraded_mode is True
        
        # Should be able to exit when TA becomes available
        ta.set_availability(True)
        result2 = await gateway.exit_degraded_mode()
        assert result2 is True
        assert gateway.is_degraded_mode is False
        assert gateway.state == ProtocolState.IDLE
    
    @pytest.mark.asyncio
    async def test_sliding_window_token_validation(self, gateway):
        """Test sliding window token validation."""
        device_id = "device_001"
        token = b"test_token_12345"
        
        # Token validation should fail if no token stored
        result1 = await gateway.validate_sliding_window_token(device_id, token)
        assert result1 is False
        
        # Store a token (simulating token generation)
        from app.components.gateway_node import SlidingWindowToken
        import time
        stored_token = SlidingWindowToken(
            device_id=device_id,
            token=token,
            expiry=time.time() + 300  # 5 minutes from now
        )
        gateway._sliding_window_tokens[device_id] = stored_token
        
        # Now validation should succeed
        result2 = await gateway.validate_sliding_window_token(device_id, token)
        assert result2 is True
        
        # Wrong token should fail
        wrong_token = b"wrong_token_123"
        result3 = await gateway.validate_sliding_window_token(device_id, wrong_token)
        assert result3 is False
    
    @pytest.mark.asyncio
    async def test_ta_unavailable_authentication(self, gateway, ta):
        """Test authentication when TA is unavailable from the start."""
        # Make TA unavailable
        ta.set_availability(False)
        
        # Try to authenticate device
        result = await gateway.authenticate_device("device_001")
        
        assert result.success is False
        assert gateway.is_degraded_mode is True
        assert "not cached" in result.error_message
    
    def test_cache_info(self, gateway):
        """Test cache information retrieval."""
        info = gateway.get_cache_info()
        
        assert "auth_cache_size" in info
        assert "sliding_window_tokens" in info
        assert "active_sessions" in info
        assert all(isinstance(v, int) for v in info.values())


if __name__ == "__main__":
    pytest.main([__file__])
