"""
Centralized Configuration for ZKPAS Simulation

This module provides all configuration constants and parameters for the
Zero-Knowledge Proof Authentication System simulation.

Enhanced with comprehensive error handling and fallback mechanisms.
"""

import os
import logging
from enum import Enum
from typing import Final, Dict, Any, Optional

# Safe import of dotenv with fallback
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logging.warning("python-dotenv not available, using environment variables only")

logger = logging.getLogger(__name__)


def safe_getenv(key: str, default: Any, convert_type: type = str) -> Any:
    """Safely get environment variable with type conversion and fallback."""
    try:
        value = os.getenv(key, str(default))
        if convert_type == str:
            return value
        elif convert_type == int:
            return int(value)
        elif convert_type == float:
            return float(value)
        elif convert_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif convert_type == bytes:
            return value.encode()
        else:
            return convert_type(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert {key}={os.getenv(key)} to {convert_type}, using default: {default}")
        return default


class CryptoConfig:
    """Cryptographic configuration constants with safe fallbacks."""
    
    # Elliptic Curve Configuration
    ECC_CURVE: Final[str] = safe_getenv("ECC_CURVE", "secp256r1")
    
    # Hash Algorithm
    HASH_ALGORITHM: Final[str] = safe_getenv("HASH_ALGORITHM", "sha256")
    
    # Key Derivation
    HKDF_SALT: Final[bytes] = safe_getenv("HKDF_SALT", "zkpas_simulation_2025", bytes)
    
    # Key Sizes (in bits)
    PRIVATE_KEY_SIZE: Final[int] = safe_getenv("PRIVATE_KEY_SIZE", 256, int)
    PUBLIC_KEY_SIZE: Final[int] = safe_getenv("PUBLIC_KEY_SIZE", 512, int)
    SYMMETRIC_KEY_SIZE: Final[int] = safe_getenv("SYMMETRIC_KEY_SIZE", 256, int)
    
    # ZKP Parameters
    ZKP_CHALLENGE_SIZE: Final[int] = safe_getenv("ZKP_CHALLENGE_SIZE", 256, int)
    ZKP_RESPONSE_SIZE: Final[int] = safe_getenv("ZKP_RESPONSE_SIZE", 256, int)
    
    # Post-Quantum Preparation
    PQ_KEY_SIZE: Final[int] = safe_getenv("PQ_KEY_SIZE", 1024, int)  # Placeholder for future PQ algorithms


def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary with safe fallbacks."""
    try:
        config = {
            # Crypto configuration
            "ECC_CURVE": CryptoConfig.ECC_CURVE,
            "HASH_ALGO": CryptoConfig.HASH_ALGORITHM,
            "HKDF_SALT": CryptoConfig.HKDF_SALT,
            "PRIVATE_KEY_SIZE": CryptoConfig.PRIVATE_KEY_SIZE,
            "PUBLIC_KEY_SIZE": CryptoConfig.PUBLIC_KEY_SIZE,
            "SYMMETRIC_KEY_SIZE": CryptoConfig.SYMMETRIC_KEY_SIZE,
            "ZKP_CHALLENGE_SIZE": CryptoConfig.ZKP_CHALLENGE_SIZE,
            "ZKP_RESPONSE_SIZE": CryptoConfig.ZKP_RESPONSE_SIZE,
            "PQ_KEY_SIZE": CryptoConfig.PQ_KEY_SIZE,
            
            # System configuration
            "dotenv_available": DOTENV_AVAILABLE,
            "config_status": "loaded"
        }
        
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Return minimal safe configuration
        return {
            "ECC_CURVE": "secp256r1",
            "HASH_ALGO": "sha256",
            "HKDF_SALT": b"zkpas_simulation_2025",
            "config_status": "fallback",
            "error": str(e)
        }


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
