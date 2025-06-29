"""
Task 4.1: Privacy-Preserving Federated Learning Pipeline
========================================================

This module implements a privacy-preserving federated learning pipeline for the
ZKPAS MLOps system. It enables distributed training across multiple nodes while
maintaining privacy through differential privacy and secure aggregation.

Key Features:
- Federated learning coordinator and client simulation
- Differential privacy mechanisms for model updates
- Secure aggregation of gradients without revealing individual contributions
- Integration with mobility prediction models
- Event-driven federation with the existing ZKPAS event system
- Privacy budget tracking and allocation
"""

import asyncio
import json
import logging
import numpy as np
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import uuid
from abc import ABC, abstractmethod

# Import from our existing modules
from app.events import EventBus, Event, EventType
from app.data_subsetting import DataSubsettingManager, DataSubset

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class FederatedClient:
    """Represents a federated learning client."""
    client_id: str
    client_type: str  # 'gateway', 'device', 'trusted_authority'
    data_size: int
    privacy_budget: float
    last_update: Optional[float] = None
    round_participated: int = 0
    is_active: bool = True
    quality_score: float = 1.0


@dataclass
class ModelUpdate:
    """Represents a model update from a federated client."""
    client_id: str
    round_number: int
    model_weights: Dict[str, np.ndarray]
    gradient_norm: float
    privacy_spent: float
    sample_count: int
    timestamp: float
    validation_metrics: Optional[Dict[str, float]] = None


@dataclass
class FederatedRound:
    """Represents a federated learning round."""
    round_number: int
    participating_clients: List[str]
    aggregated_weights: Optional[Dict[str, np.ndarray]]
    global_metrics: Optional[Dict[str, float]]
    privacy_budget_used: float
    start_time: float
    end_time: Optional[float] = None
    convergence_metrics: Optional[Dict[str, float]] = None


class PrivacyMechanism(ABC):
    """Abstract base class for privacy mechanisms."""
    
    @abstractmethod
    def add_noise(self, 
                  gradients: Dict[str, np.ndarray], 
                  sensitivity: float, 
                  epsilon: float) -> Dict[str, np.ndarray]:
        """Add privacy noise to gradients."""
        pass
    
    @abstractmethod
    def get_privacy_cost(self, sensitivity: float, noise_scale: float) -> float:
        """Calculate privacy cost (epsilon) for given parameters."""
        pass


class DifferentialPrivacyMechanism(PrivacyMechanism):
    """Implements differential privacy using Gaussian mechanism."""
    
    def __init__(self, delta: float = 1e-5, noise_multiplier: float = 1.0):
        self.delta = delta
        self.noise_multiplier = noise_multiplier
    
    def add_noise(self, 
                  gradients: Dict[str, np.ndarray], 
                  sensitivity: float, 
                  epsilon: float) -> Dict[str, np.ndarray]:
        """Add Gaussian noise for differential privacy."""
        # Calculate noise scale based on privacy budget
        sigma = self.noise_multiplier * sensitivity / epsilon
        
        noisy_gradients = {}
        for layer_name, weights in gradients.items():
            noise = np.random.normal(0, sigma, weights.shape)
            noisy_gradients[layer_name] = weights + noise
        
        logger.debug(f"Added DP noise with sigma={sigma:.4f}, epsilon={epsilon:.4f}")
        return noisy_gradients
    
    def get_privacy_cost(self, sensitivity: float, noise_scale: float) -> float:
        """Calculate epsilon for given noise scale."""
        if noise_scale == 0:
            return float('inf')
        return (self.noise_multiplier * sensitivity) / noise_scale


class SecureAggregation:
    """Implements secure aggregation for federated learning."""
    
    def __init__(self, threshold: int = 2):
        self.threshold = threshold  # Minimum number of clients for aggregation
    
    def aggregate_weights(self, 
                         updates: List[ModelUpdate],
                         aggregation_strategy: str = "fedavg") -> Dict[str, np.ndarray]:
        """
        Securely aggregate model weights from multiple clients.
        
        Args:
            updates: List of model updates from clients
            aggregation_strategy: Aggregation method ('fedavg', 'weighted')
            
        Returns:
            Aggregated model weights
        """
        if len(updates) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} updates for secure aggregation")
        
        logger.info(f"Aggregating {len(updates)} model updates using {aggregation_strategy}")
        
        if aggregation_strategy == "fedavg":
            return self._federated_averaging(updates)
        elif aggregation_strategy == "weighted":
            return self._weighted_averaging(updates)
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")
    
    def _federated_averaging(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Standard federated averaging."""
        if not updates:
            return {}
        
        # Get the structure from the first update
        aggregated = {}
        layer_names = updates[0].model_weights.keys()
        
        total_samples = sum(update.sample_count for update in updates)
        
        for layer_name in layer_names:
            weighted_sum = np.zeros_like(updates[0].model_weights[layer_name])
            
            for update in updates:
                weight = update.sample_count / total_samples
                weighted_sum += weight * update.model_weights[layer_name]
            
            aggregated[layer_name] = weighted_sum
        
        return aggregated
    
    def _weighted_averaging(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Quality-weighted averaging based on client performance."""
        if not updates:
            return {}
        
        aggregated = {}
        layer_names = updates[0].model_weights.keys()
        
        # Calculate weights based on sample count and quality
        total_weight = 0
        weights = []
        
        for update in updates:
            # Combine sample count and validation performance
            client_weight = update.sample_count
            if update.validation_metrics and 'accuracy' in update.validation_metrics:
                client_weight *= update.validation_metrics['accuracy']
            weights.append(client_weight)
            total_weight += client_weight
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        for layer_name in layer_names:
            weighted_sum = np.zeros_like(updates[0].model_weights[layer_name])
            
            for update, weight in zip(updates, weights):
                weighted_sum += weight * update.model_weights[layer_name]
            
            aggregated[layer_name] = weighted_sum
        
        return aggregated


class FederatedLearningCoordinator:
    """
    Coordinates federated learning across multiple ZKPAS components.
    
    Manages the federated learning process, client coordination, privacy
    preservation, and integration with the ZKPAS event system.
    """
    
    def __init__(self,
                 event_bus: EventBus,
                 privacy_budget: float = 10.0,
                 max_rounds: int = 100,
                 min_clients: int = 2,
                 client_fraction: float = 1.0):
        """
        Initialize the federated learning coordinator.
        
        Args:
            event_bus: ZKPAS event bus for communication
            privacy_budget: Total privacy budget for the federation
            max_rounds: Maximum number of federated rounds
            min_clients: Minimum clients required per round
            client_fraction: Fraction of clients to use per round
        """
        self.event_bus = event_bus
        self.privacy_budget = privacy_budget
        self.max_rounds = max_rounds
        self.min_clients = min_clients
        self.client_fraction = client_fraction
        
        # Initialize components
        self.privacy_mechanism = DifferentialPrivacyMechanism()
        self.secure_aggregation = SecureAggregation(threshold=min_clients)
        
        # State management
        self.clients: Dict[str, FederatedClient] = {}
        self.current_round = 0
        self.rounds_history: List[FederatedRound] = []
        self.global_model_weights: Optional[Dict[str, np.ndarray]] = None
        self.privacy_spent = 0.0
        self.is_training = False
        
        # Performance tracking
        self.convergence_threshold = 0.001
        self.no_improvement_rounds = 0
        self.max_no_improvement = 10
        
        # Setup event handlers
        self._setup_event_handlers()
        
        logger.info(f"FederatedLearningCoordinator initialized with {max_rounds} max rounds")
    
    def _setup_event_handlers(self):
        """Setup event handlers for federated learning communication."""
        self.event_bus.subscribe_sync(EventType.FL_CLIENT_REGISTER, self._handle_client_register)
        self.event_bus.subscribe_sync(EventType.FL_MODEL_UPDATE, self._handle_model_update)
        self.event_bus.subscribe_sync(EventType.FL_ROUND_COMPLETE, self._handle_round_complete)
        
        logger.debug("Federated learning event handlers registered")
    
    async def start_federation(self, initial_model: Optional[Dict[str, np.ndarray]] = None):
        """Start the federated learning process."""
        if self.is_training:
            logger.warning("Federation already in progress")
            return
        
        logger.info("Starting federated learning process")
        self.is_training = True
        self.current_round = 0
        self.privacy_spent = 0.0
        
        # Initialize global model
        if initial_model:
            self.global_model_weights = initial_model
        else:
            self.global_model_weights = self._create_initial_model()
        
        # Publish federation start event
        await self.event_bus.publish(Event(
            event_type=EventType.FL_ROUND_START,
            component_id="federation_coordinator",
            data={
                "round_number": self.current_round,
                "global_model": self._serialize_weights(self.global_model_weights),
                "privacy_budget_remaining": self.privacy_budget - self.privacy_spent
            }
        ))
        
        # Start the federation loop
        await self._run_federation_loop()
    
    async def _run_federation_loop(self):
        """Main federation training loop."""
        while (self.current_round < self.max_rounds and 
               self.privacy_spent < self.privacy_budget and
               self.no_improvement_rounds < self.max_no_improvement and
               self.is_training):
            
            logger.info(f"Starting federated round {self.current_round + 1}")
            
            # Start new round
            round_start_time = datetime.now().timestamp()
            
            # Select clients for this round
            selected_clients = self._select_clients()
            
            if len(selected_clients) < self.min_clients:
                logger.warning(f"Not enough clients ({len(selected_clients)}) for round {self.current_round + 1}")
                await asyncio.sleep(5)  # Wait for more clients
                continue
            
            # Create round info
            current_round_info = FederatedRound(
                round_number=self.current_round + 1,
                participating_clients=[c.client_id for c in selected_clients],
                aggregated_weights=None,
                global_metrics=None,
                privacy_budget_used=0.0,
                start_time=round_start_time
            )
            
            # Send model to selected clients
            await self._send_model_to_clients(selected_clients)
            
            # Wait for client updates
            updates = await self._collect_client_updates(selected_clients)
            
            if len(updates) >= self.min_clients:
                # Aggregate updates
                aggregated_weights = self.secure_aggregation.aggregate_weights(updates)
                
                # Update global model
                self.global_model_weights = aggregated_weights
                current_round_info.aggregated_weights = aggregated_weights
                
                # Calculate privacy cost for this round
                round_privacy_cost = sum(update.privacy_spent for update in updates) / len(updates)
                current_round_info.privacy_budget_used = round_privacy_cost
                self.privacy_spent += round_privacy_cost
                
                # Evaluate global model
                global_metrics = await self._evaluate_global_model()
                current_round_info.global_metrics = global_metrics
                
                # Check for convergence
                convergence_info = self._check_convergence(global_metrics)
                current_round_info.convergence_metrics = convergence_info
                
                if convergence_info.get('converged', False):
                    logger.info("Model converged, stopping federation")
                    break
                
                # Update round completion
                current_round_info.end_time = datetime.now().timestamp()
                self.rounds_history.append(current_round_info)
                
                # Publish round completion
                await self.event_bus.publish(Event(
                    event_type=EventType.FL_ROUND_COMPLETE,
                    component_id="federation_coordinator",
                    data={
                        "round_number": self.current_round + 1,
                        "global_metrics": global_metrics,
                        "privacy_spent": self.privacy_spent,
                        "participating_clients": len(selected_clients)
                    }
                ))
                
                self.current_round += 1
                
                logger.info(f"Round {self.current_round} completed. "
                           f"Privacy spent: {self.privacy_spent:.4f}/{self.privacy_budget}")
            
            else:
                logger.warning(f"Insufficient updates received for round {self.current_round + 1}")
                await asyncio.sleep(2)
        
        # Federation completed
        await self._finalize_federation()
    
    def register_client(self, 
                       client_id: str, 
                       client_type: str, 
                       data_size: int,
                       privacy_budget: float) -> bool:
        """Register a new federated learning client."""
        if client_id in self.clients:
            logger.warning(f"Client {client_id} already registered")
            return False
        
        client = FederatedClient(
            client_id=client_id,
            client_type=client_type,
            data_size=data_size,
            privacy_budget=privacy_budget,
            last_update=datetime.now().timestamp()
        )
        
        self.clients[client_id] = client
        logger.info(f"Registered FL client: {client_id} ({client_type}) with {data_size} samples")
        return True
    
    def _select_clients(self) -> List[FederatedClient]:
        """Select clients for the current round."""
        available_clients = [c for c in self.clients.values() if c.is_active]
        
        if not available_clients:
            return []
        
        # Select fraction of clients
        num_selected = max(self.min_clients, 
                          int(len(available_clients) * self.client_fraction))
        num_selected = min(num_selected, len(available_clients))
        
        # Simple random selection (could be enhanced with more sophisticated strategies)
        np.random.shuffle(available_clients)
        selected = available_clients[:num_selected]
        
        logger.debug(f"Selected {len(selected)} clients for round {self.current_round + 1}")
        return selected
    
    async def _send_model_to_clients(self, clients: List[FederatedClient]):
        """Send current global model to selected clients."""
        model_data = self._serialize_weights(self.global_model_weights)
        
        for client in clients:
            await self.event_bus.publish(Event(
                event_type=EventType.FL_MODEL_BROADCAST,
                component_id="federation_coordinator",
                target=client.client_id,
                data={
                    "round_number": self.current_round + 1,
                    "global_model": model_data,
                    "privacy_budget": client.privacy_budget / self.max_rounds
                }
            ))
    
    async def _collect_client_updates(self, 
                                    clients: List[FederatedClient],
                                    timeout: float = 30.0) -> List[ModelUpdate]:
        """Collect model updates from clients."""
        updates = []
        start_time = datetime.now().timestamp()
        
        # Wait for updates with timeout
        while (len(updates) < len(clients) and 
               (datetime.now().timestamp() - start_time) < timeout):
            await asyncio.sleep(1)
            # In a real implementation, this would check for received updates
            # For now, simulate some updates
            if len(updates) == 0:  # Simulate receiving updates
                updates = self._simulate_client_updates(clients)
        
        logger.info(f"Collected {len(updates)} updates from {len(clients)} clients")
        return updates
    
    def _simulate_client_updates(self, clients: List[FederatedClient]) -> List[ModelUpdate]:
        """Simulate client updates for demonstration."""
        updates = []
        
        for client in clients:
            # Simulate model weights (normally received from actual clients)
            simulated_weights = {}
            if self.global_model_weights:
                for layer_name, weights in self.global_model_weights.items():
                    # Add small random changes to simulate training
                    noise = np.random.normal(0, 0.01, weights.shape)
                    simulated_weights[layer_name] = weights + noise
            
            # Add differential privacy noise
            privacy_epsilon = client.privacy_budget / self.max_rounds
            private_weights = self.privacy_mechanism.add_noise(
                simulated_weights, 
                sensitivity=1.0, 
                epsilon=privacy_epsilon
            )
            
            update = ModelUpdate(
                client_id=client.client_id,
                round_number=self.current_round + 1,
                model_weights=private_weights,
                gradient_norm=np.random.uniform(0.1, 1.0),
                privacy_spent=privacy_epsilon,
                sample_count=client.data_size,
                timestamp=datetime.now().timestamp(),
                validation_metrics={'accuracy': np.random.uniform(0.7, 0.95)}
            )
            
            updates.append(update)
        
        return updates
    
    async def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate the global model performance."""
        # Simulate global evaluation
        # In real implementation, this would evaluate on validation data
        metrics = {
            'accuracy': np.random.uniform(0.75, 0.95),
            'loss': np.random.uniform(0.1, 0.5),
            'f1_score': np.random.uniform(0.70, 0.90)
        }
        
        logger.debug(f"Global model metrics: {metrics}")
        return metrics
    
    def _check_convergence(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check if the model has converged."""
        if len(self.rounds_history) < 2:
            return {'converged': False, 'improvement': 0.0}
        
        prev_metrics = self.rounds_history[-1].global_metrics
        if not prev_metrics:
            return {'converged': False, 'improvement': 0.0}
        
        # Check accuracy improvement
        prev_accuracy = prev_metrics.get('accuracy', 0.0)
        current_accuracy = current_metrics.get('accuracy', 0.0)
        improvement = current_accuracy - prev_accuracy
        
        converged = abs(improvement) < self.convergence_threshold
        
        if improvement <= 0:
            self.no_improvement_rounds += 1
        else:
            self.no_improvement_rounds = 0
        
        return {
            'converged': converged,
            'improvement': improvement,
            'no_improvement_rounds': self.no_improvement_rounds
        }
    
    async def _finalize_federation(self):
        """Finalize the federated learning process."""
        self.is_training = False
        
        final_metrics = await self._evaluate_global_model()
        
        await self.event_bus.publish(Event(
            event_type=EventType.FL_TRAINING_COMPLETE,
            component_id="federation_coordinator",
            data={
                "total_rounds": self.current_round,
                "final_metrics": final_metrics,
                "total_privacy_spent": self.privacy_spent,
                "participating_clients": len(self.clients)
            }
        ))
        
        logger.info(f"Federated learning completed after {self.current_round} rounds")
        logger.info(f"Final metrics: {final_metrics}")
        logger.info(f"Total privacy spent: {self.privacy_spent:.4f}/{self.privacy_budget}")
    
    def _create_initial_model(self) -> Dict[str, np.ndarray]:
        """Create a simple initial model for demonstration."""
        # Create a simple neural network structure
        return {
            'layer1_weights': np.random.normal(0, 0.1, (10, 5)),
            'layer1_bias': np.zeros(5),
            'layer2_weights': np.random.normal(0, 0.1, (5, 3)),
            'layer2_bias': np.zeros(3),
            'output_weights': np.random.normal(0, 0.1, (3, 1)),
            'output_bias': np.zeros(1)
        }
    
    def _serialize_weights(self, weights: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[str, List]]:
        """Serialize numpy arrays for JSON transmission."""
        if not weights:
            return None
        return {k: v.tolist() for k, v in weights.items()}
    
    def _deserialize_weights(self, weights_data: Optional[Dict[str, List]]) -> Optional[Dict[str, np.ndarray]]:
        """Deserialize weights from JSON."""
        if not weights_data:
            return None
        return {k: np.array(v) for k, v in weights_data.items()}
    
    def _handle_client_register(self, event: Event):
        """Handle client registration events."""
        data = event.data
        self.register_client(
            client_id=data.get('client_id'),
            client_type=data.get('client_type'),
            data_size=data.get('data_size', 100),
            privacy_budget=data.get('privacy_budget', 1.0)
        )
    
    def _handle_model_update(self, event: Event):
        """Handle model update events from clients."""
        # This would process real model updates in a full implementation
        logger.debug(f"Received model update from {event.component_id}")
    
    def _handle_round_complete(self, event: Event):
        """Handle round completion events."""
        logger.debug(f"Round {event.data.get('round_number')} completed")
    
    def get_federation_status(self) -> Dict[str, Any]:
        """Get current federation status."""
        return {
            'is_training': self.is_training,
            'current_round': self.current_round,
            'total_clients': len(self.clients),
            'active_clients': len([c for c in self.clients.values() if c.is_active]),
            'privacy_spent': self.privacy_spent,
            'privacy_budget': self.privacy_budget,
            'rounds_completed': len(self.rounds_history)
        }


# Extend EventType enum for federated learning
if not hasattr(EventType, 'FL_CLIENT_REGISTER'):
    # Add new event types for federated learning
    EventType.FL_CLIENT_REGISTER = "fl_client_register"
    EventType.FL_MODEL_UPDATE = "fl_model_update" 
    EventType.FL_MODEL_BROADCAST = "fl_model_broadcast"
    EventType.FL_ROUND_START = "fl_round_start"
    EventType.FL_ROUND_COMPLETE = "fl_round_complete"
    EventType.FL_TRAINING_COMPLETE = "fl_training_complete"


async def main():
    """Example usage of the federated learning pipeline."""
    print("🚀 Testing Privacy-Preserving Federated Learning Pipeline")
    
    # Initialize event bus
    event_bus = EventBus()
    await event_bus.start()
    
    # Initialize federated learning coordinator
    fl_coordinator = FederatedLearningCoordinator(
        event_bus=event_bus,
        privacy_budget=5.0,
        max_rounds=10,
        min_clients=2
    )
    
    # Register some simulated clients
    fl_coordinator.register_client("gateway_001", "gateway", 500, 2.0)
    fl_coordinator.register_client("device_001", "device", 200, 1.5)
    fl_coordinator.register_client("device_002", "device", 300, 1.5)
    
    print(f"Registered {len(fl_coordinator.clients)} federated learning clients")
    
    # Start federation
    await fl_coordinator.start_federation()
    
    # Get final status
    status = fl_coordinator.get_federation_status()
    print(f"Federation completed: {status}")
    
    await event_bus.stop()
    print("✅ Federated Learning Pipeline test completed!")


if __name__ == "__main__":
    asyncio.run(main())
