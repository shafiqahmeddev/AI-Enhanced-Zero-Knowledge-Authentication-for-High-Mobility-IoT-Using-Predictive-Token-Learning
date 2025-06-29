"""
Basic tests for the Trusted Authority implementation.
"""

import pytest
import asyncio
from app.components.trusted_authority import TrustedAuthority
from shared.crypto_utils import generate_ecc_keypair, serialize_public_key


class TestTrustedAuthority:
    """Test suite for Trusted Authority component."""
    
    @pytest.fixture
    def ta(self):
        """Create a test Trusted Authority instance."""
        return TrustedAuthority("test_ta_001")
    
    @pytest.fixture
    def device_keypair(self):
        """Generate test device keypair."""
        private_key, public_key = generate_ecc_keypair()
        return private_key, public_key
    
    @pytest.fixture
    def gateway_keypair(self):
        """Generate test gateway keypair."""
        private_key, public_key = generate_ecc_keypair()
        return private_key, public_key
    
    def test_ta_initialization(self, ta):
        """Test TA initialization."""
        assert ta.entity_id == "test_ta_001"
        assert ta.public_key is not None
        assert len(ta.public_key) == 64  # 32 bytes x + 32 bytes y
        assert ta.is_available() is True
    
    @pytest.mark.asyncio
    async def test_device_registration_success(self, ta, device_keypair):
        """Test successful device registration."""
        _, public_key = device_keypair
        pub_key_bytes = serialize_public_key(public_key)
        
        result = await ta.register_device("device_001", pub_key_bytes)
        assert result is True
        
        # Verify device is in registry
        devices = ta.get_registered_devices()
        assert "device_001" in devices
        assert devices["device_001"] == pub_key_bytes
    
    @pytest.mark.asyncio
    async def test_device_registration_duplicate(self, ta, device_keypair):
        """Test registration of duplicate device."""
        _, public_key = device_keypair
        pub_key_bytes = serialize_public_key(public_key)
        
        # First registration should succeed
        result1 = await ta.register_device("device_001", pub_key_bytes)
        assert result1 is True
        
        # Second registration should fail
        result2 = await ta.register_device("device_001", pub_key_bytes)
        assert result2 is False
    
    @pytest.mark.asyncio
    async def test_gateway_registration_success(self, ta, gateway_keypair):
        """Test successful gateway registration."""
        _, public_key = gateway_keypair
        pub_key_bytes = serialize_public_key(public_key)
        
        result = await ta.register_gateway("gateway_001", pub_key_bytes)
        assert result is True
        
        # Verify gateway is in registry
        gateways = ta.get_registered_gateways()
        assert "gateway_001" in gateways
        assert gateways["gateway_001"] == pub_key_bytes
    
    @pytest.mark.asyncio
    async def test_device_registration_invalid_key_size(self, ta):
        """Test device registration with invalid key size."""
        invalid_key = b"too_short"
        
        result = await ta.register_device("device_001", invalid_key)
        assert result is False
        
        # Verify device is not in registry
        devices = ta.get_registered_devices()
        assert "device_001" not in devices
    
    @pytest.mark.asyncio
    async def test_cross_domain_certificate_generation(self, ta, device_keypair):
        """Test cross-domain certificate generation."""
        _, public_key = device_keypair
        pub_key_bytes = serialize_public_key(public_key)
        
        # Register device first
        await ta.register_device("device_001", pub_key_bytes)
        
        # Generate certificate
        certificate = await ta.generate_cross_domain_certificate(
            "device_001", 
            "target_domain.com"
        )
        
        assert certificate is not None
        assert isinstance(certificate, bytes)
        assert len(certificate) > 0
        
        # Verify certificate count increased
        assert ta.get_issued_certificates_count() == 1
    
    @pytest.mark.asyncio
    async def test_cross_domain_certificate_unregistered_device(self, ta):
        """Test certificate generation for unregistered device."""
        certificate = await ta.generate_cross_domain_certificate(
            "unregistered_device", 
            "target_domain.com"
        )
        
        assert certificate is None
        assert ta.get_issued_certificates_count() == 0
    
    @pytest.mark.asyncio
    async def test_ta_unavailable_operations(self, ta, device_keypair):
        """Test operations when TA is unavailable."""
        _, public_key = device_keypair
        pub_key_bytes = serialize_public_key(public_key)
        
        # Make TA unavailable
        ta.set_availability(False)
        assert ta.is_available() is False
        
        # All operations should fail
        device_reg = await ta.register_device("device_001", pub_key_bytes)
        assert device_reg is False
        
        gateway_reg = await ta.register_gateway("gateway_001", pub_key_bytes)
        assert gateway_reg is False
        
        cert = await ta.generate_cross_domain_certificate("device_001", "domain.com")
        assert cert is None
        
        # Make TA available again
        ta.set_availability(True)
        assert ta.is_available() is True
        
        # Operations should work again
        device_reg = await ta.register_device("device_001", pub_key_bytes)
        assert device_reg is True


if __name__ == "__main__":
    pytest.main([__file__])
