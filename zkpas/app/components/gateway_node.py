"""
Gateway Node Implementation with Degraded Mode Support

The Gateway Node handles device authentication and provides graceful
degradation when the Trusted Authority becomes unavailable.
"""

import asyncio
import uuid
import time
from typing import Dict, Optional, Set, Tuple
from loguru import logger
from cryptography.hazmat.primitives.asymmetric import ec

from app.components.interfaces import (
    IGatewayNode,
    ITrustedAuthority,
    ProtocolMessage,
    AuthenticationResult
)
from shared.config import ProtocolState, MessageType, CryptoConfig
from shared.crypto_utils import (
    generate_ecc_keypair,
    serialize_public_key,
    generate_commitment,
    generate_challenge,
    verify_zkp,
    encrypt_aes_gcm,
    decrypt_aes_gcm,
    derive_key,
    secure_hash,
    constant_time_compare,
    CryptoError
)


class AuthenticationSession:
    """Represents an active authentication session."""
    
    def __init__(self, device_id: str, correlation_id: str):
        self.device_id = device_id
        self.correlation_id = correlation_id
        self.state = ProtocolState.IDLE
        self.timestamp = time.time()
        self.commitment: Optional[bytes] = None
        self.challenge: Optional[bytes] = None
        self.device_public_key: Optional[bytes] = None
        self.session_key: Optional[bytes] = None


class SlidingWindowToken:
    """Represents a sliding window authentication token."""
    
    def __init__(self, device_id: str, token: bytes, expiry: float):
        self.device_id = device_id
        self.token = token
        self.expiry = expiry
        self.created = time.time()


class GatewayNode(IGatewayNode):
    """
    Implementation of the Gateway Node with graceful degradation support.
    
    Features:
    - Zero-knowledge proof verification
    - Sliding window token validation
    - Degraded mode operation when TA is unavailable
    - Comprehensive audit logging
    """
    
    def __init__(self, gateway_id: str, trusted_authority: ITrustedAuthority):
        """
        Initialize the Gateway Node.
        
        Args:
            gateway_id: Unique identifier for this gateway
            trusted_authority: Reference to the trusted authority
        """
        self._gateway_id = gateway_id
        self._trusted_authority = trusted_authority
        self._private_key, self._public_key = generate_ecc_keypair()
        
        # Protocol state management
        self._state = ProtocolState.IDLE
        self._is_degraded_mode = False
        self._degraded_reason = ""
        
        # Session management
        self._active_sessions: Dict[str, AuthenticationSession] = {}
        self._authenticated_devices: Set[str] = set()
        
        # Sliding window token cache for degraded mode
        self._sliding_window_tokens: Dict[str, SlidingWindowToken] = {}
        self._token_window_size = 300  # 5 minutes
        
        # Authentication cache for degraded mode
        self._auth_cache: Dict[str, Tuple[bytes, float]] = {}  # device_id -> (pub_key, timestamp)
        self._cache_ttl = 3600  # 1 hour
        
        logger.info(
            f"Gateway Node {gateway_id} initialized",
            extra={"correlation_id": "GW_INIT"}
        )
    
    @property
    def entity_id(self) -> str:
        """Unique identifier for this gateway."""
        return self._gateway_id
    
    @property
    def public_key(self) -> bytes:
        """Public key of this gateway."""
        return serialize_public_key(self._public_key)
    
    @property
    def state(self) -> ProtocolState:
        """Current protocol state."""
        return self._state
    
    @property
    def is_degraded_mode(self) -> bool:
        """Check if gateway is in degraded mode."""
        return self._is_degraded_mode
    
    async def authenticate_device(self, device_id: str) -> AuthenticationResult:
        """
        Authenticate an IoT device using zero-knowledge proofs.
        
        Args:
            device_id: Device to authenticate
            
        Returns:
            Authentication result
        """
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Check if TA is available for full authentication
            if not self._trusted_authority.is_available():
                await self._check_degraded_mode()
                return await self._authenticate_device_degraded(device_id, correlation_id)
            
            # Create new authentication session
            session = AuthenticationSession(device_id, correlation_id)
            self._active_sessions[correlation_id] = session
            
            logger.info(
                f"Starting authentication for device {device_id}",
                extra={"correlation_id": correlation_id}
            )
            
            # Request device verification from TA
            auth_request = ProtocolMessage(
                message_type=MessageType.AUTHENTICATION_REQUEST,
                sender_id=self._gateway_id,
                recipient_id=self._trusted_authority.entity_id,
                correlation_id=correlation_id,
                timestamp=start_time,
                payload={"device_id": device_id}
            )
            
            response = await self._trusted_authority.process_message(auth_request)
            
            if response and response.message_type == MessageType.AUTHENTICATION_SUCCESS:
                # Get device public key from TA response
                device_pub_key_hex = response.payload.get("device_public_key")
                if device_pub_key_hex:
                    session.device_public_key = bytes.fromhex(device_pub_key_hex)
                    
                    # Cache the public key for degraded mode
                    self._auth_cache[device_id] = (session.device_public_key, start_time)
                    
                    # Proceed with ZKP authentication
                    result = await self._perform_zkp_authentication(session)
                    
                    if result.success:
                        self._authenticated_devices.add(device_id)
                        logger.info(
                            f"Device {device_id} authenticated successfully",
                            extra={"correlation_id": correlation_id}
                        )
                    
                    return result
            
            # Authentication failed
            error_msg = response.payload.get("error", "Unknown error") if response else "TA unavailable"
            
            logger.warning(
                f"Device authentication failed for {device_id}: {error_msg}",
                extra={"correlation_id": correlation_id}
            )
            
            return AuthenticationResult(
                success=False,
                correlation_id=correlation_id,
                timestamp=start_time,
                error_message=error_msg
            )
            
        except Exception as e:
            logger.error(
                f"Authentication error for device {device_id}: {e}",
                extra={"correlation_id": correlation_id}
            )
            
            return AuthenticationResult(
                success=False,
                correlation_id=correlation_id,
                timestamp=start_time,
                error_message=str(e)
            )
        finally:
            # Clean up session
            if correlation_id in self._active_sessions:
                del self._active_sessions[correlation_id]
    
    async def _authenticate_device_degraded(
        self, 
        device_id: str, 
        correlation_id: str
    ) -> AuthenticationResult:
        """
        Authenticate device in degraded mode using cached credentials.
        
        Args:
            device_id: Device to authenticate
            correlation_id: Authentication correlation ID
            
        Returns:
            Authentication result
        """
        start_time = time.time()
        
        logger.warning(
            f"Authenticating device {device_id} in DEGRADED MODE",
            extra={"correlation_id": correlation_id}
        )
        
        # Check if device public key is cached
        if device_id in self._auth_cache:
            cached_key, cache_time = self._auth_cache[device_id]
            
            # Check cache validity
            if start_time - cache_time < self._cache_ttl:
                # Create session with cached data
                session = AuthenticationSession(device_id, correlation_id)
                session.device_public_key = cached_key
                
                # Perform ZKP authentication with cached key
                result = await self._perform_zkp_authentication(session)
                
                if result.success:
                    logger.info(
                        f"Device {device_id} authenticated in degraded mode",
                        extra={"correlation_id": correlation_id}
                    )
                
                return result
            else:
                # Cache expired
                del self._auth_cache[device_id]
        
        # No valid cached credentials
        return AuthenticationResult(
            success=False,
            correlation_id=correlation_id,
            timestamp=start_time,
            error_message="Device not cached for degraded mode authentication"
        )
    
    async def _perform_zkp_authentication(self, session: AuthenticationSession) -> AuthenticationResult:
        """
        Perform zero-knowledge proof authentication.
        
        Args:
            session: Authentication session
            
        Returns:
            Authentication result
        """
        try:
            # This is a simplified ZKP protocol
            # In a real implementation, this would involve multiple message exchanges
            
            session.state = ProtocolState.AWAITING_COMMITMENT
            
            # Generate challenge
            session.challenge = generate_challenge()
            
            # For simulation, we'll assume the device provides a valid commitment and response
            # In reality, this would involve message exchanges with the device
            
            # Simulate commitment (normally received from device)
            device_secret = secure_hash(session.device_public_key + b"device_secret")
            nonce = secure_hash(session.correlation_id.encode() + str(time.time()).encode())
            session.commitment = generate_commitment(device_secret, nonce)
            
            session.state = ProtocolState.AWAITING_RESPONSE
            
            # Simulate response verification (normally received from device)
            # For now, we'll assume verification succeeds if device key is valid
            if session.device_public_key and len(session.device_public_key) == 64:
                # Generate session key
                shared_secret = secure_hash(session.device_public_key + self.public_key)
                session.session_key = derive_key(
                    shared_secret,
                    b"zkpas_session",
                    session.correlation_id.encode()
                )
                
                session.state = ProtocolState.AUTHENTICATED
                
                return AuthenticationResult(
                    success=True,
                    correlation_id=session.correlation_id,
                    timestamp=time.time(),
                    session_key=session.session_key
                )
            else:
                session.state = ProtocolState.ERROR
                return AuthenticationResult(
                    success=False,
                    correlation_id=session.correlation_id,
                    timestamp=time.time(),
                    error_message="Invalid device public key"
                )
                
        except Exception as e:
            session.state = ProtocolState.ERROR
            return AuthenticationResult(
                success=False,
                correlation_id=session.correlation_id,
                timestamp=time.time(),
                error_message=f"ZKP authentication failed: {e}"
            )
    
    async def validate_sliding_window_token(
        self, 
        device_id: str, 
        token: bytes
    ) -> bool:
        """
        Validate a sliding window authentication token.
        
        Args:
            device_id: Device presenting the token
            token: Token to validate
            
        Returns:
            True if token is valid, False otherwise
        """
        correlation_id = str(uuid.uuid4())
        current_time = time.time()
        
        try:
            # Check if we have a stored token for this device
            if device_id in self._sliding_window_tokens:
                stored_token = self._sliding_window_tokens[device_id]
                
                # Check token expiry
                if current_time <= stored_token.expiry:
                    # Validate token using constant-time comparison
                    if constant_time_compare(token, stored_token.token):
                        logger.info(
                            f"Sliding window token validated for device {device_id}",
                            extra={"correlation_id": correlation_id}
                        )
                        return True
                else:
                    # Token expired, remove it
                    del self._sliding_window_tokens[device_id]
                    logger.info(
                        f"Sliding window token expired for device {device_id}",
                        extra={"correlation_id": correlation_id}
                    )
            
            logger.warning(
                f"Sliding window token validation failed for device {device_id}",
                extra={"correlation_id": correlation_id}
            )
            return False
            
        except Exception as e:
            logger.error(
                f"Error validating sliding window token for device {device_id}: {e}",
                extra={"correlation_id": correlation_id}
            )
            return False
    
    async def enter_degraded_mode(self, reason: str) -> None:
        """
        Enter degraded operational mode.
        
        Args:
            reason: Reason for entering degraded mode
        """
        if not self._is_degraded_mode:
            self._is_degraded_mode = True
            self._degraded_reason = reason
            self._state = ProtocolState.DEGRADED_MODE
            
            logger.warning(
                f"Gateway entering DEGRADED MODE: {reason}",
                extra={"correlation_id": "DEGRADED_MODE_ENTER"}
            )
    
    async def exit_degraded_mode(self) -> bool:
        """
        Attempt to exit degraded mode.
        
        Returns:
            True if successfully exited, False otherwise
        """
        try:
            # Check if TA is available again
            if self._trusted_authority.is_available():
                self._is_degraded_mode = False
                self._degraded_reason = ""
                self._state = ProtocolState.IDLE
                
                logger.info(
                    "Gateway exited degraded mode - TA available",
                    extra={"correlation_id": "DEGRADED_MODE_EXIT"}
                )
                return True
            else:
                logger.info(
                    "Cannot exit degraded mode - TA still unavailable",
                    extra={"correlation_id": "DEGRADED_MODE_EXIT_FAILED"}
                )
                return False
                
        except Exception as e:
            logger.error(
                f"Error exiting degraded mode: {e}",
                extra={"correlation_id": "DEGRADED_MODE_EXIT_ERROR"}
            )
            return False
    
    async def _check_degraded_mode(self) -> None:
        """Check if gateway should enter degraded mode."""
        if not self._is_degraded_mode and not self._trusted_authority.is_available():
            await self.enter_degraded_mode("Trusted Authority unavailable")
    
    async def process_message(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """
        Process incoming protocol messages.
        
        Args:
            message: The incoming message
            
        Returns:
            Response message if any, None otherwise
        """
        try:
            if message.message_type == MessageType.SLIDING_WINDOW_TOKEN:
                # Handle sliding window token validation
                device_id = message.sender_id
                token = bytes.fromhex(message.payload.get("token", ""))
                
                is_valid = await self.validate_sliding_window_token(device_id, token)
                
                response_type = MessageType.AUTHENTICATION_SUCCESS if is_valid else MessageType.AUTHENTICATION_FAILURE
                response_payload = {"token_valid": is_valid}
                
                return ProtocolMessage(
                    message_type=response_type,
                    sender_id=self._gateway_id,
                    recipient_id=device_id,
                    correlation_id=message.correlation_id,
                    timestamp=time.time(),
                    payload=response_payload
                )
                
        except Exception as e:
            logger.error(
                f"Error processing message: {e}",
                extra={"correlation_id": message.correlation_id}
            )
        
        return None
    
    def get_authenticated_devices(self) -> Set[str]:
        """Get set of authenticated devices (for testing/debugging)."""
        return self._authenticated_devices.copy()
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get cache information (for testing/debugging)."""
        return {
            "auth_cache_size": len(self._auth_cache),
            "sliding_window_tokens": len(self._sliding_window_tokens),
            "active_sessions": len(self._active_sessions)
        }
