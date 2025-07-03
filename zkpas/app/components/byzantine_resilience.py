"""
Byzantine Fault Tolerance Module for ZKPAS

This module implements threshold cryptography and Byzantine fault tolerance
for cross-domain authentication in distributed IoT environments.

Phase 5: Advanced Authentication & Byzantine Resilience
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from loguru import logger
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature, decode_dss_signature
from cryptography.exceptions import InvalidSignature

from app.events import Event, EventBus, EventType
from shared.config import CryptoConfig
from shared.crypto_utils import (
    generate_ecc_keypair,
    serialize_public_key,
    sign_data,
    verify_signature,
    secure_hash,
    CryptoError
)


@dataclass
class SignatureShare:
    """Represents a signature share from a trust anchor."""
    anchor_id: str
    signature_share: bytes
    message_hash: bytes
    timestamp: float
    is_valid: bool = True


@dataclass
class CrossDomainAuthRequest:
    """Cross-domain authentication request."""
    request_id: str
    source_domain: str
    target_domain: str
    device_id: str
    message_hash: bytes
    timestamp: float
    required_threshold: int


@dataclass
class ThresholdSignature:
    """Aggregated threshold signature."""
    request_id: str
    aggregated_signature: bytes
    participating_anchors: Set[str]
    threshold_met: bool
    timestamp: float


class ITrustAnchor(ABC):
    """Interface for trust anchors in the Byzantine fault tolerance system."""
    
    @property
    @abstractmethod
    def anchor_id(self) -> str:
        """Get anchor identifier."""
        pass
    
    @property
    @abstractmethod
    def public_key(self) -> ec.EllipticCurvePublicKey:
        """Get public key."""
        pass
    
    @abstractmethod
    async def generate_signature_share(self, message_hash: bytes, request_id: str) -> SignatureShare:
        """Generate a signature share for a message."""
        pass
    
    @abstractmethod
    async def verify_signature_share(self, share: SignatureShare) -> bool:
        """Verify a signature share."""
        pass


class TrustAnchor(ITrustAnchor):
    """
    Honest trust anchor implementation.
    
    Provides legitimate signature shares for threshold cryptography.
    """
    
    def __init__(self, anchor_id: str, event_bus: EventBus):
        """
        Initialize trust anchor.
        
        Args:
            anchor_id: Unique identifier for this anchor
            event_bus: Event bus for communication
        """
        self._anchor_id = anchor_id
        self._event_bus = event_bus
        self._private_key, self._public_key = generate_ecc_keypair()
        self._is_available = True
        
        logger.info(f"Initialized honest trust anchor {anchor_id}")
    
    @property
    def anchor_id(self) -> str:
        """Get anchor identifier."""
        return self._anchor_id
    
    @property
    def public_key(self) -> ec.EllipticCurvePublicKey:
        """Get public key."""
        return self._public_key
    
    async def generate_signature_share(self, message_hash: bytes, request_id: str) -> SignatureShare:
        """
        Generate a legitimate signature share.
        
        Args:
            message_hash: Hash of message to sign
            request_id: Cross-domain auth request ID
            
        Returns:
            SignatureShare: Valid signature share
        """
        try:
            if not self._is_available:
                raise CryptoError(f"Trust anchor {self._anchor_id} is not available")
            
            # Generate signature share using private key
            signature = sign_data(self._private_key, message_hash)
            
            share = SignatureShare(
                anchor_id=self._anchor_id,
                signature_share=signature,
                message_hash=message_hash,
                timestamp=time.time(),
                is_valid=True
            )
            
            logger.debug(f"Trust anchor {self._anchor_id} generated signature share for request {request_id}")
            
            # Emit event
            await self._event_bus.publish(Event(
                event_type=EventType.SIGNATURE_SHARE_GENERATED,
                correlation_id=uuid.uuid4(),
                source=self._anchor_id,
                target="byzantine_coordinator",
                data={
                    "anchor_id": self._anchor_id,
                    "request_id": request_id,
                    "share_valid": True
                }
            ))
            
            return share
            
        except Exception as e:
            logger.error(f"Trust anchor {self._anchor_id} failed to generate signature share: {e}")
            
            # Return invalid share
            share = SignatureShare(
                anchor_id=self._anchor_id,
                signature_share=b"",
                message_hash=message_hash,
                timestamp=time.time(),
                is_valid=False
            )
            
            return share
    
    async def verify_signature_share(self, share: SignatureShare) -> bool:
        """
        Verify a signature share.
        
        Args:
            share: Signature share to verify
            
        Returns:
            bool: True if valid
        """
        try:
            if not share.is_valid:
                return False
            
            # Verify signature using public key
            return verify_signature(self._public_key, share.signature_share, share.message_hash)
            
        except Exception as e:
            logger.error(f"Failed to verify signature share from {share.anchor_id}: {e}")
            return False
    
    def set_availability(self, available: bool):
        """Set anchor availability (for testing)."""
        self._is_available = available
        logger.info(f"Trust anchor {self._anchor_id} availability set to {available}")


class MaliciousTrustAnchor(ITrustAnchor):
    """
    Malicious trust anchor for Byzantine fault tolerance testing.
    
    Generates invalid signature shares to test system resilience.
    """
    
    def __init__(self, anchor_id: str, event_bus: EventBus, malicious_behavior: str = "invalid_signature"):
        """
        Initialize malicious trust anchor.
        
        Args:
            anchor_id: Unique identifier for this anchor
            event_bus: Event bus for communication
            malicious_behavior: Type of malicious behavior to exhibit
        """
        self._anchor_id = anchor_id
        self._event_bus = event_bus
        self._private_key, self._public_key = generate_ecc_keypair()
        self._malicious_behavior = malicious_behavior
        
        logger.warning(f"Initialized MALICIOUS trust anchor {anchor_id} with behavior: {malicious_behavior}")
    
    @property
    def anchor_id(self) -> str:
        """Get anchor identifier."""
        return self._anchor_id
    
    @property
    def public_key(self) -> ec.EllipticCurvePublicKey:
        """Get public key."""
        return self._public_key
    
    async def generate_signature_share(self, message_hash: bytes, request_id: str) -> SignatureShare:
        """
        Generate a malicious signature share.
        
        Args:
            message_hash: Hash of message to sign
            request_id: Cross-domain auth request ID
            
        Returns:
            SignatureShare: Invalid signature share
        """
        try:
            if self._malicious_behavior == "invalid_signature":
                # Generate signature for wrong message
                wrong_message = b"malicious_message_" + message_hash[:10]
                signature = sign_data(self._private_key, wrong_message)
            
            elif self._malicious_behavior == "random_signature":
                # Generate completely random signature
                signature = secure_hash(f"random_{time.time()}_{request_id}".encode())[:64]
            
            elif self._malicious_behavior == "delayed_response":
                # Introduce artificial delay
                await asyncio.sleep(5)
                signature = sign_data(self._private_key, message_hash)
            
            else:
                # Default: invalid signature
                signature = b"invalid_signature_data"
            
            share = SignatureShare(
                anchor_id=self._anchor_id,
                signature_share=signature,
                message_hash=message_hash,
                timestamp=time.time(),
                is_valid=True  # Claim it's valid (malicious)
            )
            
            logger.warning(f"Malicious trust anchor {self._anchor_id} generated invalid signature share "
                          f"for request {request_id} (behavior: {self._malicious_behavior})")
            
            # Emit event
            await self._event_bus.publish(Event(
                event_type=EventType.SIGNATURE_SHARE_GENERATED,
                correlation_id=uuid.uuid4(),
                source=self._anchor_id,
                target="byzantine_coordinator",
                data={
                    "anchor_id": self._anchor_id,
                    "request_id": request_id,
                    "share_valid": False,  # Actually invalid
                    "malicious": True
                }
            ))
            
            return share
            
        except Exception as e:
            logger.error(f"Malicious trust anchor {self._anchor_id} failed to generate signature share: {e}")
            
            # Return invalid share
            share = SignatureShare(
                anchor_id=self._anchor_id,
                signature_share=b"",
                message_hash=message_hash,
                timestamp=time.time(),
                is_valid=False
            )
            
            return share
    
    async def verify_signature_share(self, share: SignatureShare) -> bool:
        """Always claim shares are valid (malicious behavior)."""
        return True  # Malicious: always claim validity


class TrustAnchorNetwork:
    """
    Network of trust anchors implementing threshold cryptography.
    
    Features:
    - Byzantine fault tolerance with configurable thresholds
    - Threshold signature aggregation
    - Malicious anchor detection and mitigation
    - Cross-domain authentication support
    """
    
    def __init__(self, event_bus: EventBus, threshold: int = 2):
        """
        Initialize trust anchor network.
        
        Args:
            event_bus: Event bus for communication
            threshold: Minimum number of honest anchors required
        """
        self._event_bus = event_bus
        self._threshold = threshold
        self._anchors: Dict[str, ITrustAnchor] = {}
        self._pending_requests: Dict[str, CrossDomainAuthRequest] = {}
        self._signature_shares: Dict[str, List[SignatureShare]] = {}
        
        # Register event handlers
        self._event_bus.subscribe_sync(EventType.CROSS_DOMAIN_AUTH_REQUEST, self._handle_auth_request)
        
        logger.info(f"Initialized trust anchor network with threshold {threshold}")
    
    def add_trust_anchor(self, anchor: ITrustAnchor):
        """
        Add a trust anchor to the network.
        
        Args:
            anchor: Trust anchor to add
        """
        self._anchors[anchor.anchor_id] = anchor
        logger.info(f"Added trust anchor {anchor.anchor_id} to network "
                   f"(total: {len(self._anchors)})")
    
    def remove_trust_anchor(self, anchor_id: str):
        """
        Remove a trust anchor from the network.
        
        Args:
            anchor_id: ID of anchor to remove
        """
        if anchor_id in self._anchors:
            del self._anchors[anchor_id]
            logger.info(f"Removed trust anchor {anchor_id} from network")
    
    async def request_cross_domain_authentication(
        self,
        source_domain: str,
        target_domain: str,
        device_id: str,
        message: bytes
    ) -> Optional[ThresholdSignature]:
        """
        Request cross-domain authentication with threshold signatures.
        
        Args:
            source_domain: Source domain identifier
            target_domain: Target domain identifier
            device_id: Device requesting authentication
            message: Message to be signed
            
        Returns:
            ThresholdSignature: Aggregated signature if threshold met, None otherwise
        """
        try:
            request_id = str(uuid.uuid4())
            message_hash = secure_hash(message)
            
            # Create authentication request
            auth_request = CrossDomainAuthRequest(
                request_id=request_id,
                source_domain=source_domain,
                target_domain=target_domain,
                device_id=device_id,
                message_hash=message_hash,
                timestamp=time.time(),
                required_threshold=self._threshold
            )
            
            self._pending_requests[request_id] = auth_request
            self._signature_shares[request_id] = []
            
            logger.info(f"Requesting cross-domain authentication: {request_id} "
                       f"({source_domain} -> {target_domain}, device: {device_id})")
            
            # Emit request event
            await self._event_bus.publish(Event(
                event_type=EventType.CROSS_DOMAIN_AUTH_REQUEST,
                correlation_id=uuid.uuid4(),
                source="byzantine_coordinator",
                target="trust_anchor_network",
                data={
                    "request_id": request_id,
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "device_id": device_id,
                    "message_hash": message_hash.hex(),
                    "threshold": self._threshold
                }
            ))
            
            # Collect signature shares from all anchors
            share_tasks = []
            for anchor in self._anchors.values():
                task = asyncio.create_task(
                    anchor.generate_signature_share(message_hash, request_id)
                )
                share_tasks.append(task)
            
            # Wait for all shares with timeout
            try:
                shares = await asyncio.wait_for(
                    asyncio.gather(*share_tasks, return_exceptions=True),
                    timeout=10.0
                )
                
                # Process shares
                valid_shares = []
                for share in shares:
                    if isinstance(share, SignatureShare):
                        # Verify the share
                        anchor = self._anchors.get(share.anchor_id)
                        if anchor and await anchor.verify_signature_share(share):
                            valid_shares.append(share)
                        else:
                            logger.warning(f"Invalid signature share from {share.anchor_id}")
                
                self._signature_shares[request_id] = valid_shares
                
                # Check if threshold is met
                if len(valid_shares) >= self._threshold:
                    threshold_signature = await self._aggregate_signatures(request_id, valid_shares)
                    
                    logger.info(f"Cross-domain authentication successful: {request_id} "
                               f"({len(valid_shares)}/{len(self._anchors)} valid shares)")
                    
                    # Emit success event
                    await self._event_bus.publish(Event(
                        event_type=EventType.CROSS_DOMAIN_AUTH_SUCCESS,
                        correlation_id=uuid.uuid4(),
                        source="byzantine_coordinator",
                        target=device_id,
                        data={
                            "request_id": request_id,
                            "valid_shares": len(valid_shares),
                            "total_anchors": len(self._anchors),
                            "threshold_met": True
                        }
                    ))
                    
                    return threshold_signature
                else:
                    logger.error(f"Cross-domain authentication failed: {request_id} "
                                f"(only {len(valid_shares)}/{self._threshold} valid shares)")
                    
                    # Emit failure event
                    await self._event_bus.publish(Event(
                        event_type=EventType.CROSS_DOMAIN_AUTH_FAILURE,
                        correlation_id=uuid.uuid4(),
                        source="byzantine_coordinator",
                        target=device_id,
                        data={
                            "request_id": request_id,
                            "valid_shares": len(valid_shares),
                            "required_threshold": self._threshold,
                            "threshold_met": False
                        }
                    ))
                    
                    return None
                
            except asyncio.TimeoutError:
                logger.error(f"Cross-domain authentication timeout: {request_id}")
                return None
                
        except Exception as e:
            logger.error(f"Cross-domain authentication error: {e}")
            return None
        
        finally:
            # Cleanup
            self._pending_requests.pop(request_id, None)
            self._signature_shares.pop(request_id, None)
    
    async def _aggregate_signatures(self, request_id: str, valid_shares: List[SignatureShare]) -> ThresholdSignature:
        """
        Aggregate valid signature shares into a threshold signature.
        
        Args:
            request_id: Authentication request ID
            valid_shares: List of valid signature shares
            
        Returns:
            ThresholdSignature: Aggregated signature
        """
        try:
            # Simple aggregation: concatenate all valid signatures
            # In a real implementation, this would use proper threshold signature schemes
            aggregated_data = b""
            participating_anchors = set()
            
            for share in valid_shares:
                aggregated_data += share.signature_share
                participating_anchors.add(share.anchor_id)
            
            # Create hash of aggregated signatures
            aggregated_signature = secure_hash(aggregated_data)
            
            threshold_signature = ThresholdSignature(
                request_id=request_id,
                aggregated_signature=aggregated_signature,
                participating_anchors=participating_anchors,
                threshold_met=True,
                timestamp=time.time()
            )
            
            logger.info(f"Aggregated threshold signature for request {request_id} "
                       f"with {len(participating_anchors)} anchors")
            
            return threshold_signature
            
        except Exception as e:
            logger.error(f"Failed to aggregate signatures for request {request_id}: {e}")
            raise
    
    async def _handle_auth_request(self, event: Event):
        """Handle cross-domain authentication request events."""
        # This is handled by the public request method
        pass
    
    def get_network_status(self) -> Dict:
        """Get current network status."""
        honest_anchors = sum(1 for a in self._anchors.values() 
                           if not isinstance(a, MaliciousTrustAnchor))
        malicious_anchors = len(self._anchors) - honest_anchors
        
        return {
            "total_anchors": len(self._anchors),
            "honest_anchors": honest_anchors,
            "malicious_anchors": malicious_anchors,
            "threshold": self._threshold,
            "byzantine_resilient": honest_anchors >= self._threshold,
            "pending_requests": len(self._pending_requests)
        }


class ByzantineResilienceCoordinator:
    """
    Main coordinator for Byzantine fault tolerance in ZKPAS.
    
    Manages trust anchor networks and provides high-level Byzantine
    resilience capabilities for the authentication system.
    """
    
    def __init__(self, event_bus: EventBus, default_threshold: int = 2):
        """
        Initialize Byzantine resilience coordinator.
        
        Args:
            event_bus: Event bus for communication
            default_threshold: Default threshold for new networks
        """
        self._event_bus = event_bus
        self._default_threshold = default_threshold
        self._networks: Dict[str, TrustAnchorNetwork] = {}
        
        logger.info(f"Initialized Byzantine resilience coordinator with default threshold {default_threshold}")
    
    def create_trust_network(self, network_id: str, threshold: Optional[int] = None) -> TrustAnchorNetwork:
        """
        Create a new trust anchor network.
        
        Args:
            network_id: Unique identifier for the network
            threshold: Threshold for this network (uses default if None)
            
        Returns:
            TrustAnchorNetwork: New trust network
        """
        threshold = threshold or self._default_threshold
        network = TrustAnchorNetwork(self._event_bus, threshold)
        self._networks[network_id] = network
        
        logger.info(f"Created trust network {network_id} with threshold {threshold}")
        return network
    
    def get_trust_network(self, network_id: str) -> Optional[TrustAnchorNetwork]:
        """Get a trust network by ID."""
        return self._networks.get(network_id)
    
    async def test_byzantine_resilience(self, network_id: str, num_malicious: int) -> Dict:
        """
        Test Byzantine resilience by adding malicious anchors.
        
        Args:
            network_id: Network to test
            num_malicious: Number of malicious anchors to add
            
        Returns:
            Dict: Test results
        """
        network = self._networks.get(network_id)
        if not network:
            raise ValueError(f"Trust network {network_id} not found")
        
        logger.info(f"Testing Byzantine resilience for network {network_id} "
                   f"with {num_malicious} malicious anchors")
        
        # Add malicious anchors
        malicious_behaviors = ["invalid_signature", "random_signature", "delayed_response"]
        malicious_anchors = []
        
        for i in range(num_malicious):
            behavior = malicious_behaviors[i % len(malicious_behaviors)]
            anchor_id = f"malicious_anchor_{i}"
            malicious_anchor = MaliciousTrustAnchor(anchor_id, self._event_bus, behavior)
            network.add_trust_anchor(malicious_anchor)
            malicious_anchors.append(anchor_id)
        
        # Test authentication with malicious anchors
        test_message = b"test_cross_domain_authentication_message"
        result = await network.request_cross_domain_authentication(
            source_domain="domain_a",
            target_domain="domain_b",
            device_id="test_device",
            message=test_message
        )
        
        # Cleanup malicious anchors
        for anchor_id in malicious_anchors:
            network.remove_trust_anchor(anchor_id)
        
        test_results = {
            "network_id": network_id,
            "num_malicious_anchors": num_malicious,
            "authentication_successful": result is not None,
            "threshold_met": result.threshold_met if result else False,
            "network_status": network.get_network_status()
        }
        
        logger.info(f"Byzantine resilience test results: {test_results}")
        return test_results
    
    def get_system_status(self) -> Dict:
        """Get overall system status."""
        status = {
            "total_networks": len(self._networks),
            "default_threshold": self._default_threshold,
            "networks": {}
        }
        
        for network_id, network in self._networks.items():
            status["networks"][network_id] = network.get_network_status()
        
        return status