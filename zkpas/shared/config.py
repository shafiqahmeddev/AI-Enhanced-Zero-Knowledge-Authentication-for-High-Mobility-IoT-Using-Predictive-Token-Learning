"""
Centralized Configuration for ZKPAS Simulation

This module provides all configuration constants and parameters for the
Zero-Knowledge Proof Authentication System simulation.
"""

import os
from enum import Enum
from typing import Final

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CryptoConfig:
    """Cryptographic configuration constants."""
    
    # Elliptic Curve Configuration
    ECC_CURVE: Final[str] = os.getenv("ECC_CURVE", "secp256r1")
    
    # Hash Algorithm
    HASH_ALGORITHM: Final[str] = os.getenv("HASH_ALGORITHM", "sha256")
    
    # Key Derivation
    HKDF_SALT: Final[bytes] = os.getenv("HKDF_SALT", "zkpas_simulation_2025").encode()
    
    # Key Sizes (in bits)
    PRIVATE_KEY_SIZE: Final[int] = 256
    PUBLIC_KEY_SIZE: Final[int] = 512
    SYMMETRIC_KEY_SIZE: Final[int] = 256
    
    # ZKP Parameters
    ZKP_CHALLENGE_SIZE: Final[int] = 256
    ZKP_RESPONSE_SIZE: Final[int] = 256
    
    # Post-Quantum Preparation
    PQ_KEY_SIZE: Final[int] = 1024  # Placeholder for future PQ algorithms


class SimulationConfig:
    """Simulation runtime configuration."""
    
    # Device Limits
    MAX_DEVICES: Final[int] = int(os.getenv("MAX_DEVICES", "100"))
    
    # Timing Parameters
    SIMULATION_DURATION: Final[int] = int(os.getenv("SIMULATION_DURATION", "3600"))
    AUTHENTICATION_TIMEOUT: Final[int] = 30
    HANDSHAKE_TIMEOUT: Final[int] = 10
    
    # Network Parameters
    NETWORK_LATENCY_MS: Final[int] = int(os.getenv("NETWORK_LATENCY_MS", "50"))
    PACKET_DROP_RATE: Final[float] = float(os.getenv("PACKET_DROP_RATE", "0.01"))
    
    # Memory Constraints (for MacBook Pro 2017)
    MAX_MEMORY_GB: Final[int] = int(os.getenv("MAX_MEMORY_GB", "6"))
    BATCH_SIZE: Final[int] = int(os.getenv("BATCH_SIZE", "32"))


class MLConfig:
    """Machine Learning configuration."""
    
    # Model Parameters
    MODEL_QUANTIZATION: Final[bool] = os.getenv("MODEL_QUANTIZATION", "True").lower() == "true"
    SEQUENCE_LENGTH: Final[int] = 10
    PREDICTION_HORIZON: Final[int] = 5
    
    # Training Parameters
    LEARNING_RATE: Final[float] = 0.001
    EPOCHS: Final[int] = 50
    VALIDATION_SPLIT: Final[float] = 0.2
    
    # Federated Learning
    FEDERATED_ROUNDS: Final[int] = 10
    MIN_CLIENTS_PER_ROUND: Final[int] = 5


class DatabaseConfig:
    """Database configuration."""
    
    DATABASE_URL: Final[str] = os.getenv("DATABASE_URL", "sqlite:///zkpas_simulation.db")


class LoggingConfig:
    """Logging configuration."""
    
    LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: Final[str] = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<blue>{extra[correlation_id]}</blue> | "
        "<level>{message}</level>"
    )


class ExperimentConfig:
    """Experiment tracking configuration."""
    
    MLFLOW_TRACKING_URI: Final[str] = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    EXPERIMENT_NAME: Final[str] = os.getenv("EXPERIMENT_NAME", "ZKPAS_Mobility_Prediction")


class ProtocolState(Enum):
    """Protocol state machine states."""
    
    IDLE = "idle"
    AWAITING_COMMITMENT = "awaiting_commitment"
    AWAITING_RESPONSE = "awaiting_response"
    AUTHENTICATED = "authenticated"
    DEGRADED_MODE = "degraded_mode"
    ERROR = "error"


class MessageType(Enum):
    """Protocol message types."""
    
    AUTHENTICATION_REQUEST = "auth_request"
    COMMITMENT = "commitment"
    CHALLENGE = "challenge"
    RESPONSE = "response"
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    SLIDING_WINDOW_TOKEN = "sliding_token"
    CROSS_DOMAIN_REQUEST = "cross_domain_request"


# Global configuration instance
CONFIG = {
    'crypto': CryptoConfig,
    'simulation': SimulationConfig,
    'ml': MLConfig,
    'database': DatabaseConfig,
    'logging': LoggingConfig,
    'experiment': ExperimentConfig,
}
