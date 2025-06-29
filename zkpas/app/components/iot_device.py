"""
IoT Device Implementation

The IoT Device component handles authentication initiation,
zero-knowledge proof generation, and mobility tracking.
"""

import asyncio
import uuid
import time
import random
from typing import List, Optional, Dict, Any
from loguru import logger

from app.components.interfaces import (
    IIoTDevice,
    IGatewayNode,
    ProtocolMessage,
    AuthenticationResult,
    DeviceLocation
)
from shared.config import ProtocolState, MessageType, CryptoConfig
from shared.crypto_utils import (
    generate_ecc_keypair,
    serialize_public_key,
    generate_commitment,
    compute_zkp_response,
    secure_hash,
    derive_key,
    encrypt_aes_gcm,
    secure_random_bytes
)


class IoTDevice(IIoTDevice):
    """
    Implementation of an IoT Device with mobility tracking and authentication.
    
    Features:
    - Zero-knowledge proof generation
    - Mobility history management
    - Sliding window token caching
    - Location-aware authentication
    """
    
    def __init__(self, device_id: str, initial_location: DeviceLocation):
        """
        Initialize the IoT Device.
        
        Args:
            device_id: Unique identifier for this device
            initial_location: Initial device location
        """
        self._device_id = device_id
        self._private_key, self._public_key = generate_ecc_keypair()
        
        # Location and mobility tracking
        self._current_location = initial_location
        self._mobility_history: List[DeviceLocation] = [initial_location]
        self._max_history_size = 100
        
        # Authentication state
        self._state = ProtocolState.IDLE
        self._current_session: Optional[str] = None
        self._session_key: Optional[bytes] = None
        
        # Sliding window token cache
        self._sliding_window_tokens: Dict[str, bytes] = {}  # gateway_id -> token
        self._token_expiry: Dict[str, float] = {}  # gateway_id -> expiry_time
        
        # ZKP session data
        self._current_commitment: Optional[bytes] = None
        self._current_nonce: Optional[bytes] = None
        self._current_challenge: Optional[bytes] = None
        
        logger.info(
            f"IoT Device {device_id} initialized at location ({initial_location.latitude}, {initial_location.longitude})",
            extra={"correlation_id": f"DEVICE_INIT_{device_id}"}
        )
    
    @property
    def entity_id(self) -> str:
        """Unique identifier for this device."""
        return self._device_id
    
    @property
    def public_key(self) -> bytes:
        """Public key of this device."""
        return serialize_public_key(self._public_key)
    
    @property
    def current_location(self) -> DeviceLocation:
        """Current location of the device."""
        return self._current_location
    
    @property
    def mobility_history(self) -> List[DeviceLocation]:
        """Historical mobility data."""
        return self._mobility_history.copy()
    
    async def update_location(self, location: DeviceLocation) -> None:
        """
        Update device location and mobility history.
        
        Args:
            location: New location data
        """
        self._current_location = location
        self._mobility_history.append(location)
        
        # Maintain history size limit
        if len(self._mobility_history) > self._max_history_size:
            self._mobility_history = self._mobility_history[-self._max_history_size:]
        
        logger.debug(
            f"Device {self._device_id} location updated to ({location.latitude}, {location.longitude})",
            extra={"correlation_id": f"LOCATION_UPDATE_{self._device_id}"}
        )
    
    async def initiate_authentication(self, gateway_id: str) -> AuthenticationResult:
        """
        Initiate authentication with a gateway.
        
        Args:
            gateway_id: Target gateway identifier
            
        Returns:
            Authentication result
        """
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(
                f"Device {self._device_id} initiating authentication with gateway {gateway_id}",
                extra={"correlation_id": correlation_id}
            )
            
            # Check if we have a valid sliding window token
            if await self._try_sliding_window_auth(gateway_id, correlation_id):
                return AuthenticationResult(
                    success=True,
                    correlation_id=correlation_id,
                    timestamp=start_time,
                    session_key=self._session_key
                )
            
            # Perform full ZKP authentication
            return await self._perform_full_authentication(gateway_id, correlation_id, start_time)
            
        except Exception as e:
            logger.error(
                f"Authentication initiation failed for device {self._device_id}: {e}",
                extra={"correlation_id": correlation_id}
            )
            
            return AuthenticationResult(
                success=False,
                correlation_id=correlation_id,
                timestamp=start_time,
                error_message=str(e)
            )
    
    async def _try_sliding_window_auth(self, gateway_id: str, correlation_id: str) -> bool:
        """
        Try authentication using sliding window token.
        
        Args:
            gateway_id: Target gateway identifier
            correlation_id: Authentication correlation ID
            
        Returns:
            True if sliding window auth succeeded, False otherwise
        """
        try:
            # Check if we have a valid token for this gateway
            if gateway_id in self._sliding_window_tokens:
                token_expiry = self._token_expiry.get(gateway_id, 0)
                
                if time.time() < token_expiry:
                    logger.info(
                        f"Using sliding window token for gateway {gateway_id}",
                        extra={"correlation_id": correlation_id}
                    )
                    
                    # In a real implementation, we would send the token to the gateway
                    # For simulation, we'll generate a session key directly
                    shared_secret = secure_hash(self.public_key + gateway_id.encode())
                    self._session_key = derive_key(
                        shared_secret,
                        b"sliding_window",
                        correlation_id.encode()
                    )
                    
                    return True
                else:
                    # Token expired, remove it
                    del self._sliding_window_tokens[gateway_id]
                    del self._token_expiry[gateway_id]
                    
                    logger.debug(
                        f"Sliding window token expired for gateway {gateway_id}",
                        extra={"correlation_id": correlation_id}
                    )
            
            return False
            
        except Exception as e:
            logger.error(
                f"Error in sliding window authentication: {e}",
                extra={"correlation_id": correlation_id}
            )
            return False
    
    async def _perform_full_authentication(
        self, 
        gateway_id: str, 
        correlation_id: str, 
        start_time: float
    ) -> AuthenticationResult:
        """
        Perform full zero-knowledge proof authentication.
        
        Args:
            gateway_id: Target gateway identifier
            correlation_id: Authentication correlation ID
            start_time: Authentication start timestamp
            
        Returns:
            Authentication result
        """
        try:
            # Generate commitment for ZKP
            self._state = ProtocolState.AWAITING_COMMITMENT
            self._current_session = correlation_id
            
            # Generate secret and nonce
            device_secret = secure_hash(self.public_key + b"device_secret")
            self._current_nonce = secure_random_bytes(32)
            self._current_commitment = generate_commitment(device_secret, self._current_nonce)
            
            logger.debug(
                f"Generated ZKP commitment for device {self._device_id}",
                extra={"correlation_id": correlation_id}
            )
            
            # In a real implementation, this would involve actual message exchanges
            # For simulation, we'll assume the gateway responds with a challenge
            self._state = ProtocolState.AWAITING_CHALLENGE
            
            # Simulate receiving challenge from gateway
            self._current_challenge = secure_random_bytes(CryptoConfig.ZKP_CHALLENGE_SIZE // 8)
            
            self._state = ProtocolState.COMPUTING_RESPONSE
            
            # Compute ZKP response
            response = compute_zkp_response(
                device_secret,
                self._current_nonce,
                self._current_challenge
            )
            
            logger.debug(
                f"Computed ZKP response for device {self._device_id}",
                extra={"correlation_id": correlation_id}
            )
            
            # Simulate successful authentication
            self._state = ProtocolState.AUTHENTICATED
            
            # Generate session key
            shared_secret = secure_hash(self.public_key + gateway_id.encode())
            self._session_key = derive_key(
                shared_secret,
                b"zkp_session",
                correlation_id.encode()
            )
            
            # Generate new sliding window token for future use
            await self._generate_sliding_window_token(gateway_id)
            
            logger.info(
                f"Device {self._device_id} authenticated successfully with gateway {gateway_id}",
                extra={"correlation_id": correlation_id}
            )
            
            return AuthenticationResult(
                success=True,
                correlation_id=correlation_id,
                timestamp=start_time,
                session_key=self._session_key
            )
            
        except Exception as e:
            self._state = ProtocolState.ERROR
            logger.error(
                f"Full authentication failed for device {self._device_id}: {e}",
                extra={"correlation_id": correlation_id}
            )
            
            return AuthenticationResult(
                success=False,
                correlation_id=correlation_id,
                timestamp=start_time,
                error_message=str(e)
            )
        finally:
            # Clean up session data
            self._current_commitment = None
            self._current_nonce = None
            self._current_challenge = None
            self._current_session = None
            
            if self._state != ProtocolState.AUTHENTICATED:
                self._state = ProtocolState.IDLE
    
    async def _generate_sliding_window_token(self, gateway_id: str) -> None:
        """
        Generate a sliding window token for future authentication.
        
        Args:
            gateway_id: Gateway identifier
        """
        try:
            # Generate token based on session key and current time
            token_data = self._session_key + gateway_id.encode() + str(time.time()).encode()
            token = secure_hash(token_data)
            
            # Store token with 5-minute expiry
            self._sliding_window_tokens[gateway_id] = token
            self._token_expiry[gateway_id] = time.time() + 300  # 5 minutes
            
            logger.debug(
                f"Generated sliding window token for gateway {gateway_id}",
                extra={"correlation_id": self._current_session or "TOKEN_GEN"}
            )
            
        except Exception as e:
            logger.error(
                f"Error generating sliding window token: {e}",
                extra={"correlation_id": self._current_session or "TOKEN_GEN_ERROR"}
            )
    
    async def get_sliding_window_token(self) -> Optional[bytes]:
        """
        Get current sliding window authentication token.
        
        Returns:
            Token bytes if available, None otherwise
        """
        # Return the most recent token (for testing purposes)
        if self._sliding_window_tokens:
            # Get the most recently created token
            latest_gateway = max(self._token_expiry.keys(), key=lambda k: self._token_expiry[k])
            
            if time.time() < self._token_expiry[latest_gateway]:
                return self._sliding_window_tokens[latest_gateway]
        
        return None
    
    async def process_message(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """
        Process incoming protocol messages.
        
        Args:
            message: The incoming message
            
        Returns:
            Response message if any, None otherwise
        """
        try:
            if message.message_type == MessageType.CHALLENGE:
                # Handle challenge from gateway during ZKP
                self._current_challenge = bytes.fromhex(message.payload.get("challenge", ""))
                
                # Generate response
                if self._current_commitment and self._current_nonce:
                    device_secret = secure_hash(self.public_key + b"device_secret")
                    response = compute_zkp_response(
                        device_secret,
                        self._current_nonce,
                        self._current_challenge
                    )
                    
                    response_message = ProtocolMessage(
                        message_type=MessageType.RESPONSE,
                        sender_id=self._device_id,
                        recipient_id=message.sender_id,
                        correlation_id=message.correlation_id,
                        timestamp=time.time(),
                        payload={"response": response.hex()}
                    )
                    
                    return response_message
                    
        except Exception as e:
            logger.error(
                f"Error processing message: {e}",
                extra={"correlation_id": message.correlation_id}
            )
        
        return None
    
    def simulate_mobility(self, time_delta: float) -> None:
        """
        Simulate device mobility for testing purposes.
        
        Args:
            time_delta: Time elapsed since last update (seconds)
        """
        # Simple random walk mobility model
        lat_change = random.uniform(-0.001, 0.001)  # Small random changes
        lon_change = random.uniform(-0.001, 0.001)
        
        new_location = DeviceLocation(
            latitude=self._current_location.latitude + lat_change,
            longitude=self._current_location.longitude + lon_change,
            timestamp=self._current_location.timestamp + time_delta,
            accuracy=random.uniform(1.0, 10.0)  # GPS accuracy in meters
        )
        
        asyncio.create_task(self.update_location(new_location))
    
    def get_token_cache_info(self) -> Dict[str, Any]:
        """Get information about cached tokens (for testing/debugging)."""
        current_time = time.time()
        return {
            "cached_tokens": len(self._sliding_window_tokens),
            "valid_tokens": sum(1 for expiry in self._token_expiry.values() if expiry > current_time),
            "expired_tokens": sum(1 for expiry in self._token_expiry.values() if expiry <= current_time),
            "mobility_history_size": len(self._mobility_history)
        }
