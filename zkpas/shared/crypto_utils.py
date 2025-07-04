"""
Cryptographic Utilities for ZKPAS Simulation

This module provides secure, constant-time cryptographic primitives
for the Zero-Knowledge Proof Authentication System.

Enhanced with comprehensive error handling and fallback mechanisms.
All functions are pure and stateless for maximum security and testability.
"""

import hashlib
import hmac
import secrets
import logging
from typing import Tuple, Optional, Union, Any
from dataclasses import dataclass

# Safe import of cryptography with fallback
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning("cryptography library not available, using fallback implementations")

# Safe import of config
try:
    from shared.config import CryptoConfig, get_config
except ImportError:
    # Fallback configuration
    class CryptoConfig:
        ECC_CURVE = "secp256r1"
        HASH_ALGORITHM = "sha256"
        HKDF_SALT = b"zkpas_simulation_2025"
    
    def get_config():
        return {"ECC_CURVE": "secp256r1", "HASH_ALGO": "sha256"}

logger = logging.getLogger(__name__)


class CryptoError(Exception):
    """Base exception for cryptographic operations."""
    pass


class KeyGenerationError(CryptoError):
    """Exception raised during key generation."""
    pass


class SignatureError(CryptoError):
    """Exception raised during signature operations."""
    pass


class EncryptionError(CryptoError):
    """Exception raised during encryption/decryption operations."""
    pass


@dataclass
class SimpleKeyPair:
    """Simple key pair for fallback implementations."""
    private_key: bytes
    public_key: bytes
    key_type: str


def generate_key_pair() -> Union[Tuple[Any, Any], SimpleKeyPair]:
    """
    Generate a key pair using the best available method.
    
    Returns:
        Either a cryptography ECC key pair or a simple fallback key pair
    """
    if CRYPTOGRAPHY_AVAILABLE:
        return generate_ecc_keypair()
    else:
        return generate_simple_keypair()


def generate_simple_keypair() -> SimpleKeyPair:
    """
    Generate a simple key pair using basic cryptographic primitives.
    
    Returns:
        SimpleKeyPair with random keys
    """
    try:
        # Generate 32-byte private key
        private_key = secrets.token_bytes(32)
        
        # Generate public key as hash of private key (simplified)
        public_key = hashlib.sha256(private_key + b"public").digest()
        
        return SimpleKeyPair(
            private_key=private_key,
            public_key=public_key,
            key_type="simple_hash"
        )
    except Exception as e:
        logger.error(f"Error generating simple key pair: {e}")
        raise KeyGenerationError(f"Failed to generate simple key pair: {e}")


def generate_ecc_keypair() -> Tuple[Any, Any]:
    """
    Generate an ECC keypair using the configured curve.
    
    Returns:
        Tuple of (private_key, public_key)
        
    Raises:
        KeyGenerationError: If key generation fails
    """
    if not CRYPTOGRAPHY_AVAILABLE:
        raise KeyGenerationError("cryptography library not available")
    
    try:
        # Map curve name to cryptography curve object
        curve_map = {
            'secp256r1': ec.SECP256R1(),
            'secp384r1': ec.SECP384R1(),
            'secp521r1': ec.SECP521R1(),
        }
        
        curve_name = getattr(CryptoConfig, 'ECC_CURVE', 'secp256r1')
        if curve_name not in curve_map:
            logger.warning(f"Unknown curve {curve_name}, using secp256r1")
            curve_name = 'secp256r1'
        
        curve = curve_map[curve_name]
        private_key = ec.generate_private_key(curve, default_backend())
        public_key = private_key.public_key()
        
        return private_key, public_key
        
    except Exception as e:
        logger.error(f"Error generating ECC key pair: {e}")
        raise KeyGenerationError(f"Failed to generate ECC key pair: {e}")


def sign_message(private_key: Union[Any, SimpleKeyPair], message: bytes) -> bytes:
    """
    Sign a message using the provided private key.
    
    Args:
        private_key: Either ECC private key or SimpleKeyPair
        message: Message to sign
        
    Returns:
        Signature bytes
    """
    try:
        if isinstance(private_key, SimpleKeyPair):
            return simple_sign_message(private_key, message)
        elif CRYPTOGRAPHY_AVAILABLE:
            return ecc_sign_message(private_key, message)
        else:
            raise SignatureError("No signing method available")
    except Exception as e:
        logger.error(f"Error signing message: {e}")
        raise SignatureError(f"Failed to sign message: {e}")


def simple_sign_message(key_pair: SimpleKeyPair, message: bytes) -> bytes:
    """Simple HMAC-based message signing."""
    try:
        signature = hmac.new(key_pair.private_key, message, hashlib.sha256).digest()
        return signature
    except Exception as e:
        raise SignatureError(f"Simple signing failed: {e}")


def ecc_sign_message(private_key: Any, message: bytes) -> bytes:
    """ECC-based message signing."""
    if not CRYPTOGRAPHY_AVAILABLE:
        raise SignatureError("cryptography library not available")
    
    try:
        signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        return signature
    except Exception as e:
        raise SignatureError(f"ECC signing failed: {e}")


def verify_signature(public_key: Union[Any, SimpleKeyPair], message: bytes, signature: bytes) -> bool:
    """
    Verify a message signature.
    
    Args:
        public_key: Either ECC public key or SimpleKeyPair
        message: Original message
        signature: Signature to verify
        
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        if isinstance(public_key, SimpleKeyPair):
            return simple_verify_signature(public_key, message, signature)
        elif CRYPTOGRAPHY_AVAILABLE:
            return ecc_verify_signature(public_key, message, signature)
        else:
            logger.warning("No verification method available")
            return False
    except Exception as e:
        logger.error(f"Error verifying signature: {e}")
        return False


def simple_verify_signature(key_pair: SimpleKeyPair, message: bytes, signature: bytes) -> bool:
    """Simple HMAC-based signature verification."""
    try:
        expected_signature = hmac.new(key_pair.private_key, message, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        logger.error(f"Simple verification failed: {e}")
        return False


def ecc_verify_signature(public_key: Any, message: bytes, signature: bytes) -> bool:
    """ECC-based signature verification."""
    if not CRYPTOGRAPHY_AVAILABLE:
        return False
    
    try:
        public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
        return True
    except Exception as e:
        logger.debug(f"ECC verification failed: {e}")
        return False


def generate_random_bytes(length: int) -> bytes:
    """
    Generate cryptographically secure random bytes.
    
    Args:
        length: Number of bytes to generate
        
    Returns:
        Random bytes
    """
    try:
        return secrets.token_bytes(length)
    except Exception as e:
        logger.error(f"Error generating random bytes: {e}")
        # Fallback to os.urandom
        import os
        return os.urandom(length)


def hash_data(data: bytes, algorithm: str = "sha256") -> bytes:
    """
    Hash data using the specified algorithm.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm to use
        
    Returns:
        Hash digest
    """
    try:
        if algorithm.lower() == "sha256":
            return hashlib.sha256(data).digest()
        elif algorithm.lower() == "sha384":
            return hashlib.sha384(data).digest()
        elif algorithm.lower() == "sha512":
            return hashlib.sha512(data).digest()
        else:
            logger.warning(f"Unknown hash algorithm {algorithm}, using SHA256")
            return hashlib.sha256(data).digest()
    except Exception as e:
        logger.error(f"Error hashing data: {e}")
        # Final fallback
        return hashlib.md5(data).digest()


def is_crypto_available() -> bool:
    """Check if full cryptography is available."""
    return CRYPTOGRAPHY_AVAILABLE


def get_crypto_status() -> dict:
    """Get cryptography system status."""
    return {
        "cryptography_available": CRYPTOGRAPHY_AVAILABLE,
        "config_loaded": hasattr(CryptoConfig, 'ECC_CURVE'),
        "curves_supported": ['secp256r1', 'secp384r1', 'secp521r1'] if CRYPTOGRAPHY_AVAILABLE else [],
        "hash_algorithms": ['sha256', 'sha384', 'sha512'],
        "fallback_available": True
    }


# Additional utility functions for demonstration compatibility

def compute_zkp_response(secret: bytes, nonce: bytes, challenge: bytes) -> bytes:
    """Compute zero-knowledge proof response using available hash functions."""
    try:
        combined = secret + nonce + challenge
        return hash_data(combined)
    except Exception as e:
        logger.error(f"Error computing ZKP response: {e}")
        return hashlib.sha256(secret + nonce + challenge).digest()


def generate_commitment(secret: bytes, nonce: bytes) -> bytes:
    """Generate a commitment for zero-knowledge proof."""
    try:
        return hash_data(secret + nonce)
    except Exception as e:
        logger.error(f"Error generating commitment: {e}")
        return hashlib.sha256(secret + nonce).digest()


def generate_challenge() -> bytes:
    """Generate a random challenge for zero-knowledge proof."""
    try:
        return generate_random_bytes(32)  # 256 bits
    except Exception as e:
        logger.error(f"Error generating challenge: {e}")
        return secrets.token_bytes(32)


# For compatibility with existing demos
def secure_random_bytes(length: int) -> bytes:
    """Generate cryptographically secure random bytes."""
    return generate_random_bytes(length)


def constant_time_compare(a: bytes, b: bytes) -> bool:
    """Constant-time comparison of two byte strings."""
    return hmac.compare_digest(a, b)
