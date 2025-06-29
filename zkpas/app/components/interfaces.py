"""
Component Interfaces for ZKPAS Simulation

This module defines strict interfaces for all system components using
Python's Abstract Base Classes (ABC) to ensure consistent implementation
and enable proper testing and verification.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

from shared.config import ProtocolState, MessageType


@dataclass
class AuthenticationResult:
    """Result of an authentication attempt."""
    success: bool
    correlation_id: str
    timestamp: float
    error_message: Optional[str] = None
    session_key: Optional[bytes] = None


@dataclass
class ProtocolMessage:
    """Protocol message structure."""
    message_type: MessageType
    sender_id: str
    recipient_id: str
    correlation_id: str
    timestamp: float
    payload: Dict[str, Any]
    signature: Optional[bytes] = None


@dataclass
class DeviceLocation:
    """Device location information."""
    latitude: float
    longitude: float
    timestamp: float
    accuracy: Optional[float] = None


@dataclass
class MobilityPrediction:
    """Mobility prediction result."""
    predicted_locations: List[DeviceLocation]
    confidence: float
    prediction_horizon: int
    model_version: str


class IAuthenticationEntity(ABC):
    """Base interface for all authentication entities."""
    
    @property
    @abstractmethod
    def entity_id(self) -> str:
        """Unique identifier for this entity."""
        pass
    
    @property
    @abstractmethod
    def public_key(self) -> bytes:
        """Public key of this entity."""
        pass
    
    @abstractmethod
    async def process_message(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """
        Process an incoming protocol message.
        
        Args:
            message: The incoming message
            
        Returns:
            Response message if any, None otherwise
        """
        pass


class ITrustedAuthority(IAuthenticationEntity):
    """Interface for the Trusted Authority component."""
    
    @abstractmethod
    async def register_device(self, device_id: str, public_key: bytes) -> bool:
        """
        Register a new IoT device.
        
        Args:
            device_id: Unique device identifier
            public_key: Device's public key
            
        Returns:
            True if registration successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def register_gateway(self, gateway_id: str, public_key: bytes) -> bool:
        """
        Register a new gateway node.
        
        Args:
            gateway_id: Unique gateway identifier
            public_key: Gateway's public key
            
        Returns:
            True if registration successful, False otherwise
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the trusted authority is available."""
        pass


class IIoTDevice(IAuthenticationEntity):
    """Interface for IoT Device components."""
    
    @property
    @abstractmethod
    def current_location(self) -> DeviceLocation:
        """Current location of the device."""
        pass
    
    @property
    @abstractmethod
    def mobility_history(self) -> List[DeviceLocation]:
        """Historical mobility data."""
        pass
    
    @abstractmethod
    async def initiate_authentication(self, gateway_id: str) -> AuthenticationResult:
        """
        Initiate authentication with a gateway.
        
        Args:
            gateway_id: Target gateway identifier
            
        Returns:
            Authentication result
        """
        pass
    
    @abstractmethod
    async def update_location(self, location: DeviceLocation) -> None:
        """
        Update device location.
        
        Args:
            location: New location data
        """
        pass
    
    @abstractmethod
    async def get_sliding_window_token(self) -> Optional[bytes]:
        """
        Get current sliding window authentication token.
        
        Returns:
            Token bytes if available, None otherwise
        """
        pass


class IGatewayNode(IAuthenticationEntity):
    """Interface for Gateway Node components."""
    
    @property
    @abstractmethod
    def state(self) -> ProtocolState:
        """Current protocol state."""
        pass
    
    @property
    @abstractmethod
    def is_degraded_mode(self) -> bool:
        """Check if gateway is in degraded mode."""
        pass
    
    @abstractmethod
    async def authenticate_device(self, device_id: str) -> AuthenticationResult:
        """
        Authenticate an IoT device.
        
        Args:
            device_id: Device to authenticate
            
        Returns:
            Authentication result
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def enter_degraded_mode(self, reason: str) -> None:
        """
        Enter degraded operational mode.
        
        Args:
            reason: Reason for entering degraded mode
        """
        pass
    
    @abstractmethod
    async def exit_degraded_mode(self) -> bool:
        """
        Attempt to exit degraded mode.
        
        Returns:
            True if successfully exited, False otherwise
        """
        pass


class IMobilityPredictor(ABC):
    """Interface for mobility prediction components."""
    
    @abstractmethod
    async def predict_mobility(
        self, 
        device_id: str, 
        history: List[DeviceLocation]
    ) -> MobilityPrediction:
        """
        Predict future mobility for a device.
        
        Args:
            device_id: Device identifier
            history: Historical location data
            
        Returns:
            Mobility prediction
        """
        pass
    
    @abstractmethod
    async def train_model(self, training_data: Dict[str, List[DeviceLocation]]) -> None:
        """
        Train the mobility prediction model.
        
        Args:
            training_data: Training dataset per device
        """
        pass
    
    @abstractmethod
    async def update_model(self, device_id: str, new_data: List[DeviceLocation]) -> None:
        """
        Update model with new mobility data.
        
        Args:
            device_id: Device identifier
            new_data: New location data
        """
        pass


class ITrustAnchor(ABC):
    """Interface for Trust Anchor components (for cross-domain authentication)."""
    
    @property
    @abstractmethod
    def anchor_id(self) -> str:
        """Unique identifier for this trust anchor."""
        pass
    
    @abstractmethod
    async def generate_signature_share(
        self, 
        message: bytes, 
        threshold_params: Dict[str, Any]
    ) -> Optional[bytes]:
        """
        Generate a signature share for threshold cryptography.
        
        Args:
            message: Message to sign
            threshold_params: Threshold signature parameters
            
        Returns:
            Signature share if successful, None otherwise
        """
        pass


class ITrustAnchorNetwork(ABC):
    """Interface for Trust Anchor Network management."""
    
    @abstractmethod
    async def aggregate_signatures(
        self, 
        message: bytes, 
        signature_shares: List[Tuple[str, bytes]]
    ) -> Optional[bytes]:
        """
        Aggregate signature shares into final signature.
        
        Args:
            message: Original message
            signature_shares: List of (anchor_id, signature_share) tuples
            
        Returns:
            Aggregated signature if threshold met, None otherwise
        """
        pass
    
    @abstractmethod
    async def verify_cross_domain_signature(
        self, 
        message: bytes, 
        signature: bytes
    ) -> bool:
        """
        Verify a cross-domain signature.
        
        Args:
            message: Original message
            signature: Signature to verify
            
        Returns:
            True if signature valid, False otherwise
        """
        pass


class IEventQueue(ABC):
    """Interface for asynchronous event handling."""
    
    @abstractmethod
    async def publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Publish an event to the queue.
        
        Args:
            event_type: Type of event
            payload: Event data
        """
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: str, handler) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Event type to subscribe to
            handler: Async function to handle events
        """
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the event processing loop."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the event processing loop."""
        pass


class ISimulationEnvironment(ABC):
    """Interface for the overall simulation environment."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the simulation environment."""
        pass
    
    @abstractmethod
    async def run_simulation(self, duration: int) -> Dict[str, Any]:
        """
        Run the simulation for specified duration.
        
        Args:
            duration: Simulation duration in seconds
            
        Returns:
            Simulation results and metrics
        """
        pass
    
    @abstractmethod
    async def add_device(self, device: IIoTDevice) -> bool:
        """
        Add a device to the simulation.
        
        Args:
            device: Device to add
            
        Returns:
            True if successfully added, False otherwise
        """
        pass
    
    @abstractmethod
    async def add_gateway(self, gateway: IGatewayNode) -> bool:
        """
        Add a gateway to the simulation.
        
        Args:
            gateway: Gateway to add
            
        Returns:
            True if successfully added, False otherwise
        """
        pass
    
    @abstractmethod
    async def inject_network_fault(self, fault_type: str, parameters: Dict[str, Any]) -> None:
        """
        Inject network faults for testing resilience.
        
        Args:
            fault_type: Type of fault to inject
            parameters: Fault parameters
        """
        pass
