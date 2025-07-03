"""
Sliding Window Authentication Module for ZKPAS

This module implements advanced sliding window authentication with AES-GCM encryption
and secure fallback mechanisms for high-mobility IoT devices.

Phase 5: Advanced Authentication & Byzantine Resilience
"""

import asyncio
import time
import uuid
from typing import Dict, Optional, List, Tuple, Set
from dataclasses import dataclass
from loguru import logger
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature

from app.events import Event, EventBus, EventType
from shared.config import CryptoConfig
from shared.crypto_utils import (
    encrypt_aes_gcm,
    decrypt_aes_gcm,
    derive_key,
    secure_hash,
    constant_time_compare,
    CryptoError
)


@dataclass
class SlidingWindowToken:
    """Represents a sliding window authentication token."""
    device_id: str
    token_id: str
    encrypted_payload: bytes
    nonce: bytes
    tag: bytes
    expiry_timestamp: float
    generation_timestamp: float
    sequence_number: int
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() > self.expiry_timestamp
    
    def is_valid_sequence(self, expected_min_seq: int) -> bool:
        """Check if sequence number is valid (prevents replay attacks)."""
        return self.sequence_number >= expected_min_seq


@dataclass
class AuthenticationWindow:
    """Represents an authentication window for a device."""
    device_id: str
    window_start: float
    window_end: float
    valid_tokens: Set[str]
    sequence_counter: int
    master_key: bytes
    
    def is_active(self) -> bool:
        """Check if window is currently active."""
        current_time = time.time()
        return self.window_start <= current_time <= self.window_end


class SlidingWindowAuthenticator:
    """
    Advanced sliding window authentication system with AES-GCM encryption.
    
    Features:
    - Time-based sliding windows for token validity
    - AES-GCM encryption for secure token storage
    - Sequence numbers for replay attack prevention
    - Secure fallback mechanisms during network issues
    - Event-driven architecture integration
    """
    
    def __init__(self, event_bus: EventBus, window_duration: int = 300):
        """
        Initialize the sliding window authenticator.
        
        Args:
            event_bus: Event bus for async communication
            window_duration: Duration of each authentication window in seconds
        """
        self._event_bus = event_bus
        self._window_duration = window_duration
        self._windows: Dict[str, AuthenticationWindow] = {}
        self._token_cache: Dict[str, SlidingWindowToken] = {}
        self._fallback_mode = False
        self._fallback_reason = ""
        
        # Security parameters
        self._max_tokens_per_window = 100
        self._token_lifetime = 600  # 10 minutes
        self._sequence_window = 10  # Accept tokens within 10 sequence numbers
        
        # Register event handlers
        self._event_bus.subscribe_sync(EventType.DEVICE_AUTHENTICATED, self._handle_device_authenticated)
        self._event_bus.subscribe_sync(EventType.TOKEN_VALIDATION_REQUEST, self._handle_token_validation)
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_tokens())
    
    async def create_authentication_window(self, device_id: str, master_key: bytes) -> AuthenticationWindow:
        """
        Create a new authentication window for a device.
        
        Args:
            device_id: Unique identifier for the device
            master_key: Master key for token encryption
            
        Returns:
            AuthenticationWindow: New authentication window
        """
        current_time = time.time()
        window_start = current_time
        window_end = current_time + self._window_duration
        
        window = AuthenticationWindow(
            device_id=device_id,
            window_start=window_start,
            window_end=window_end,
            valid_tokens=set(),
            sequence_counter=0,
            master_key=master_key
        )
        
        self._windows[device_id] = window
        
        logger.info(f"Created authentication window for device {device_id}, "
                   f"valid from {window_start} to {window_end}")
        
        # Emit event
        await self._event_bus.publish(Event(
            event_type=EventType.WINDOW_CREATED,
            correlation_id=uuid.uuid4(),
            source="sliding_window_auth",
            target=device_id,
            data={
                "device_id": device_id,
                "window_start": window_start,
                "window_end": window_end
            }
        ))
        
        return window
    
    async def generate_sliding_window_token(self, device_id: str, payload: Dict) -> Optional[SlidingWindowToken]:
        """
        Generate a new sliding window token for a device.
        
        Args:
            device_id: Device identifier
            payload: Token payload data
            
        Returns:
            SlidingWindowToken: New token or None if generation fails
        """
        try:
            # Get or create authentication window
            window = self._windows.get(device_id)
            if not window or not window.is_active():
                logger.warning(f"No active window found for device {device_id}")
                return None
            
            # Check token limit
            if len(window.valid_tokens) >= self._max_tokens_per_window:
                logger.warning(f"Token limit reached for device {device_id}")
                return None
            
            # Generate token
            token_id = str(uuid.uuid4())
            sequence_number = window.sequence_counter
            window.sequence_counter += 1
            
            # Prepare token payload
            token_payload = {
                "device_id": device_id,
                "token_id": token_id,
                "sequence_number": sequence_number,
                "generation_time": time.time(),
                "payload": payload
            }
            
            # Encrypt token payload
            encrypted_payload, nonce, tag = encrypt_aes_gcm(
                data=str(token_payload).encode(),
                key=window.master_key
            )
            
            # Create token
            token = SlidingWindowToken(
                device_id=device_id,
                token_id=token_id,
                encrypted_payload=encrypted_payload,
                nonce=nonce,
                tag=tag,
                expiry_timestamp=time.time() + self._token_lifetime,
                generation_timestamp=time.time(),
                sequence_number=sequence_number
            )
            
            # Store token
            self._token_cache[token_id] = token
            window.valid_tokens.add(token_id)
            
            logger.info(f"Generated sliding window token {token_id} for device {device_id}")
            
            # Emit event
            await self._event_bus.publish(Event(
                event_type=EventType.TOKEN_GENERATED,
                correlation_id=uuid.uuid4(),
                source="sliding_window_auth",
                target=device_id,
                data={
                    "device_id": device_id,
                    "token_id": token_id,
                    "sequence_number": sequence_number
                }
            ))
            
            return token
            
        except Exception as e:
            logger.error(f"Failed to generate sliding window token for device {device_id}: {e}")
            return None
    
    async def validate_sliding_window_token(self, token_id: str, expected_device_id: str) -> Tuple[bool, Optional[Dict]]:
        """
        Validate a sliding window token.
        
        Args:
            token_id: Token identifier
            expected_device_id: Expected device identifier
            
        Returns:
            Tuple[bool, Optional[Dict]]: (is_valid, payload)
        """
        try:
            # Check if token exists
            token = self._token_cache.get(token_id)
            if not token:
                logger.warning(f"Token {token_id} not found in cache")
                return False, None
            
            # Check if token is expired
            if token.is_expired():
                logger.warning(f"Token {token_id} has expired")
                await self._remove_token(token_id)
                return False, None
            
            # Check device ID match
            if token.device_id != expected_device_id:
                logger.warning(f"Token {token_id} device ID mismatch")
                return False, None
            
            # Get authentication window
            window = self._windows.get(token.device_id)
            if not window:
                logger.warning(f"No authentication window found for device {token.device_id}")
                return False, None
            
            # Check if token is in valid tokens set
            if token_id not in window.valid_tokens:
                logger.warning(f"Token {token_id} not in valid tokens set")
                return False, None
            
            # Check sequence number (prevent replay attacks)
            min_sequence = max(0, window.sequence_counter - self._sequence_window)
            if not token.is_valid_sequence(min_sequence):
                logger.warning(f"Token {token_id} has invalid sequence number")
                return False, None
            
            # Decrypt and validate token payload
            try:
                decrypted_data = decrypt_aes_gcm(
                    ciphertext=token.encrypted_payload,
                    key=window.master_key,
                    nonce=token.nonce,
                    tag=token.tag
                )
                
                # Parse payload (simplified - in production would use proper serialization)
                import ast
                payload = ast.literal_eval(decrypted_data.decode())
                
                # Validate payload structure
                if not isinstance(payload, dict):
                    logger.warning(f"Token {token_id} has invalid payload format")
                    return False, None
                
                if payload.get("device_id") != expected_device_id:
                    logger.warning(f"Token {token_id} payload device ID mismatch")
                    return False, None
                
                logger.info(f"Successfully validated sliding window token {token_id}")
                
                # Emit event
                await self._event_bus.publish(Event(
                    event_type=EventType.TOKEN_VALIDATED,
                    correlation_id=uuid.uuid4(),
                    source="sliding_window_auth",
                    target=token.device_id,
                    data={
                        "device_id": token.device_id,
                        "token_id": token_id,
                        "validation_result": "success"
                    }
                ))
                
                return True, payload.get("payload")
                
            except Exception as e:
                logger.error(f"Failed to decrypt token {token_id}: {e}")
                return False, None
                
        except Exception as e:
            logger.error(f"Failed to validate sliding window token {token_id}: {e}")
            return False, None
    
    async def enable_fallback_mode(self, reason: str):
        """
        Enable fallback mode for degraded operation.
        
        Args:
            reason: Reason for enabling fallback mode
        """
        self._fallback_mode = True
        self._fallback_reason = reason
        
        logger.warning(f"Sliding window authenticator entering fallback mode: {reason}")
        
        # Emit event
        await self._event_bus.publish(Event(
            event_type=EventType.FALLBACK_MODE_ENABLED,
            correlation_id=uuid.uuid4(),
            source="sliding_window_auth",
            target="system",
            data={
                "reason": reason,
                "timestamp": time.time()
            }
        ))
    
    async def disable_fallback_mode(self):
        """Disable fallback mode and return to normal operation."""
        self._fallback_mode = False
        self._fallback_reason = ""
        
        logger.info("Sliding window authenticator exiting fallback mode")
        
        # Emit event
        await self._event_bus.publish(Event(
            event_type=EventType.FALLBACK_MODE_DISABLED,
            correlation_id=uuid.uuid4(),
            source="sliding_window_auth",
            target="system",
            data={"timestamp": time.time()}
        ))
    
    async def _handle_device_authenticated(self, event: Event):
        """Handle device authentication events."""
        device_id = event.data.get("device_id")
        session_key = event.data.get("session_key")
        
        if device_id and session_key:
            # Create authentication window for newly authenticated device
            await self.create_authentication_window(device_id, session_key)
    
    async def _handle_token_validation(self, event: Event):
        """Handle token validation requests."""
        token_id = event.data.get("token_id")
        device_id = event.data.get("device_id")
        
        if token_id and device_id:
            is_valid, payload = await self.validate_sliding_window_token(token_id, device_id)
            
            # Send response event
            await self._event_bus.publish(Event(
                event_type=EventType.TOKEN_VALIDATION_RESPONSE,
                correlation_id=event.correlation_id,
                source="sliding_window_auth",
                target=event.source,
                data={
                    "token_id": token_id,
                    "device_id": device_id,
                    "is_valid": is_valid,
                    "payload": payload
                }
            ))
    
    async def _remove_token(self, token_id: str):
        """Remove a token from cache and valid tokens set."""
        token = self._token_cache.pop(token_id, None)
        if token:
            window = self._windows.get(token.device_id)
            if window:
                window.valid_tokens.discard(token_id)
    
    async def _cleanup_expired_tokens(self):
        """Periodically clean up expired tokens and windows."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = time.time()
                expired_tokens = []
                expired_windows = []
                
                # Find expired tokens
                for token_id, token in self._token_cache.items():
                    if token.is_expired():
                        expired_tokens.append(token_id)
                
                # Find expired windows
                for device_id, window in self._windows.items():
                    if current_time > window.window_end:
                        expired_windows.append(device_id)
                
                # Remove expired tokens
                for token_id in expired_tokens:
                    await self._remove_token(token_id)
                
                # Remove expired windows
                for device_id in expired_windows:
                    del self._windows[device_id]
                
                if expired_tokens or expired_windows:
                    logger.info(f"Cleaned up {len(expired_tokens)} expired tokens "
                               f"and {len(expired_windows)} expired windows")
                
            except Exception as e:
                logger.error(f"Error during token cleanup: {e}")
    
    def get_statistics(self) -> Dict:
        """Get current statistics."""
        active_windows = sum(1 for w in self._windows.values() if w.is_active())
        total_tokens = len(self._token_cache)
        
        return {
            "active_windows": active_windows,
            "total_windows": len(self._windows),
            "total_tokens": total_tokens,
            "fallback_mode": self._fallback_mode,
            "fallback_reason": self._fallback_reason
        }
    
    async def shutdown(self):
        """Shutdown the authenticator and cleanup resources."""
        if hasattr(self, '_cleanup_task'):
            self._cleanup_task.cancel()
        
        self._windows.clear()
        self._token_cache.clear()
        
        logger.info("Sliding window authenticator shutdown complete")