"""
Trusted Authority Implementation

The Trusted Authority is responsible for device registration,
certificate issuance, and cross-domain authentication support.
"""

import asyncio
import uuid
from typing import Dict, Optional, Set
from loguru import logger
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from app.components.interfaces import (
    ITrustedAuthority, 
    ProtocolMessage,
    AuthenticationResult
)
from shared.config import MessageType, CryptoConfig
from shared.crypto_utils import (
    generate_ecc_keypair,
    serialize_public_key,
    secure_hash,
    CryptoError
)


class TrustedAuthority(ITrustedAuthority):
    """
    Implementation of the Trusted Authority component.
    
    Manages device registration, gateway registration, and cross-domain
    certificate issuance with high security standards.
    """
    
    def __init__(self, authority_id: str):
        """
        Initialize the Trusted Authority.
        
        Args:
            authority_id: Unique identifier for this authority
        """
        self._authority_id = authority_id
        self._private_key, self._public_key = generate_ecc_keypair()
        self._registered_devices: Dict[str, bytes] = {}
        self._registered_gateways: Dict[str, bytes] = {}
        self._issued_certificates: Set[str] = set()
        self._is_available = True
        
        logger.info(
            f"Trusted Authority {authority_id} initialized",
            extra={"correlation_id": "TA_INIT"}
        )
    
    @property
    def entity_id(self) -> str:
        """Unique identifier for this authority."""
        return self._authority_id
    
    @property
    def public_key(self) -> bytes:
        """Public key of this authority."""
        return serialize_public_key(self._public_key)
    
    async def register_device(self, device_id: str, public_key: bytes) -> bool:
        """
        Register a new IoT device.
        
        Args:
            device_id: Unique device identifier
            public_key: Device's public key
            
        Returns:
            True if registration successful, False otherwise
        """
        correlation_id = str(uuid.uuid4())
        
        try:
            if not self._is_available:
                logger.warning(
                    f"Device registration failed - TA unavailable: {device_id}",
                    extra={"correlation_id": correlation_id}
                )
                return False
            
            if device_id in self._registered_devices:
                logger.warning(
                    f"Device already registered: {device_id}",
                    extra={"correlation_id": correlation_id}
                )
                return False
            
            # Validate public key format
            if len(public_key) != CryptoConfig.PUBLIC_KEY_SIZE // 8:
                logger.error(
                    f"Invalid public key size for device {device_id}",
                    extra={"correlation_id": correlation_id}
                )
                return False
            
            self._registered_devices[device_id] = public_key
            
            logger.info(
                f"Device registered successfully: {device_id}",
                extra={"correlation_id": correlation_id}
            )
            return True
            
        except Exception as e:
            logger.error(
                f"Device registration failed for {device_id}: {e}",
                extra={"correlation_id": correlation_id}
            )
            return False
    
    async def register_gateway(self, gateway_id: str, public_key: bytes) -> bool:
        """
        Register a new gateway node.
        
        Args:
            gateway_id: Unique gateway identifier
            public_key: Gateway's public key
            
        Returns:
            True if registration successful, False otherwise
        """
        correlation_id = str(uuid.uuid4())
        
        try:
            if not self._is_available:
                logger.warning(
                    f"Gateway registration failed - TA unavailable: {gateway_id}",
                    extra={"correlation_id": correlation_id}
                )
                return False
            
            if gateway_id in self._registered_gateways:
                logger.warning(
                    f"Gateway already registered: {gateway_id}",
                    extra={"correlation_id": correlation_id}
                )
                return False
            
            # Validate public key format
            if len(public_key) != CryptoConfig.PUBLIC_KEY_SIZE // 8:
                logger.error(
                    f"Invalid public key size for gateway {gateway_id}",
                    extra={"correlation_id": correlation_id}
                )
                return False
            
            self._registered_gateways[gateway_id] = public_key
            
            logger.info(
                f"Gateway registered successfully: {gateway_id}",
                extra={"correlation_id": correlation_id}
            )
            return True
            
        except Exception as e:
            logger.error(
                f"Gateway registration failed for {gateway_id}: {e}",
                extra={"correlation_id": correlation_id}
            )
            return False
    
    async def generate_cross_domain_certificate(
        self, 
        device_id: str, 
        target_domain: str
    ) -> Optional[bytes]:
        """
        Generate a cross-domain authentication certificate.
        
        Args:
            device_id: Device requesting certificate
            target_domain: Target domain for access
            
        Returns:
            Certificate bytes if successful, None otherwise
        """
        correlation_id = str(uuid.uuid4())
        
        try:
            if not self._is_available:
                logger.warning(
                    f"Certificate generation failed - TA unavailable: {device_id}",
                    extra={"correlation_id": correlation_id}
                )
                return None
            
            if device_id not in self._registered_devices:
                logger.error(
                    f"Certificate requested for unregistered device: {device_id}",
                    extra={"correlation_id": correlation_id}
                )
                return None
            
            # Generate unique certificate ID
            cert_id = f"{device_id}:{target_domain}:{correlation_id}"
            
            if cert_id in self._issued_certificates:
                logger.warning(
                    f"Certificate already issued: {cert_id}",
                    extra={"correlation_id": correlation_id}
                )
                return None
            
            # Create certificate data
            device_pub_key = self._registered_devices[device_id]
            cert_data = {
                "device_id": device_id,
                "target_domain": target_domain,
                "device_public_key": device_pub_key.hex(),
                "issuer": self._authority_id,
                "certificate_id": cert_id
            }
            
            # Create certificate hash (simplified - in real implementation,
            # this would be a proper X.509 certificate)
            cert_bytes = str(cert_data).encode('utf-8')
            certificate_hash = secure_hash(cert_bytes)
            
            # Sign the certificate with TA's private key
            signature = self._private_key.sign(
                certificate_hash,
                ec.ECDSA(hashes.SHA256())
            )
            
            # Store issued certificate
            self._issued_certificates.add(cert_id)
            
            # Return certificate (hash + signature + cert_data)
            certificate = certificate_hash + signature + cert_bytes
            
            logger.info(
                f"Cross-domain certificate issued: {cert_id}",
                extra={"correlation_id": correlation_id}
            )
            
            return certificate
            
        except CryptoError as e:
            logger.error(
                f"Cryptographic error in certificate generation for {device_id}: {e}",
                extra={"correlation_id": correlation_id}
            )
            return None
        except Exception as e:
            logger.error(
                f"Certificate generation failed for {device_id}: {e}",
                extra={"correlation_id": correlation_id}
            )
            return None
    
    def is_available(self) -> bool:
        """Check if the trusted authority is available."""
        return self._is_available
    
    def set_availability(self, available: bool) -> None:
        """
        Set the availability status of the TA.
        
        Args:
            available: New availability status
        """
        old_status = self._is_available
        self._is_available = available
        
        if old_status != available:
            status = "available" if available else "unavailable"
            logger.info(
                f"Trusted Authority status changed to: {status}",
                extra={"correlation_id": "TA_STATUS_CHANGE"}
            )
    
    async def process_message(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """
        Process incoming protocol messages.
        
        Args:
            message: The incoming message
            
        Returns:
            Response message if any, None otherwise
        """
        try:
            if message.message_type == MessageType.AUTHENTICATION_REQUEST:
                # Handle authentication requests from gateways
                device_id = message.payload.get("device_id")
                gateway_id = message.sender_id
                
                if device_id and device_id in self._registered_devices:
                    if gateway_id in self._registered_gateways:
                        # Create authentication confirmation
                        response_payload = {
                            "device_public_key": self._registered_devices[device_id].hex(),
                            "device_verified": True,
                            "gateway_verified": True
                        }
                        
                        response = ProtocolMessage(
                            message_type=MessageType.AUTHENTICATION_SUCCESS,
                            sender_id=self._authority_id,
                            recipient_id=gateway_id,
                            correlation_id=message.correlation_id,
                            timestamp=asyncio.get_event_loop().time(),
                            payload=response_payload
                        )
                        
                        logger.info(
                            f"Authentication verification sent for device {device_id}",
                            extra={"correlation_id": message.correlation_id}
                        )
                        
                        return response
                
                # Authentication failed
                response_payload = {
                    "error": "Device or gateway not registered",
                    "device_verified": False,
                    "gateway_verified": gateway_id in self._registered_gateways
                }
                
                response = ProtocolMessage(
                    message_type=MessageType.AUTHENTICATION_FAILURE,
                    sender_id=self._authority_id,
                    recipient_id=message.sender_id,
                    correlation_id=message.correlation_id,
                    timestamp=asyncio.get_event_loop().time(),
                    payload=response_payload
                )
                
                return response
                
        except Exception as e:
            logger.error(
                f"Error processing message: {e}",
                extra={"correlation_id": message.correlation_id}
            )
        
        return None
    
    def get_registered_devices(self) -> Dict[str, bytes]:
        """Get all registered devices (for testing/debugging)."""
        return self._registered_devices.copy()
    
    def get_registered_gateways(self) -> Dict[str, bytes]:
        """Get all registered gateways (for testing/debugging)."""
        return self._registered_gateways.copy()
    
    def get_issued_certificates_count(self) -> int:
        """Get count of issued certificates."""
        return len(self._issued_certificates)
