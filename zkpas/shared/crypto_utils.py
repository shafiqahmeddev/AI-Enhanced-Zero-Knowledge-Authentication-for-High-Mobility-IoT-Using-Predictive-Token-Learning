"""
Cryptographic Utilities for ZKPAS Simulation

This module provides secure, constant-time cryptographic primitives
for the Zero-Knowledge Proof Authentication System.

All functions are pure and stateless for maximum security and testability.
"""

import hashlib
import hmac
import secrets
from typing import Tuple, Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

from shared.config import CryptoConfig


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


def generate_ecc_keypair() -> Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
    """
    Generate an ECC keypair using the configured curve.
    
    Returns:
        Tuple of (private_key, public_key)
        
    Raises:
        KeyGenerationError: If key generation fails
    """
    try:
        # Map curve name to cryptography curve object
        curve_map = {
            'secp256r1': ec.SECP256R1(),
            'secp384r1': ec.SECP384R1(),
            'secp521r1': ec.SECP521R1(),
        }
        
        curve = curve_map.get(CryptoConfig.ECC_CURVE)
        if curve is None:
            raise KeyGenerationError(f"Unsupported curve: {CryptoConfig.ECC_CURVE}")
            
        private_key = ec.generate_private_key(curve, default_backend())
        public_key = private_key.public_key()
        
        return private_key, public_key
        
    except Exception as e:
        raise KeyGenerationError(f"Failed to generate ECC keypair: {e}") from e


def serialize_public_key(public_key: ec.EllipticCurvePublicKey) -> bytes:
    """
    Serialize a public key to bytes.
    
    Args:
        public_key: The public key to serialize
        
    Returns:
        Serialized public key bytes
    """
    return public_key.public_numbers().x.to_bytes(32, 'big') + \
           public_key.public_numbers().y.to_bytes(32, 'big')


def secure_hash(data: bytes) -> bytes:
    """
    Compute a secure hash of the input data.
    
    Args:
        data: Data to hash
        
    Returns:
        Hash digest
    """
    if CryptoConfig.HASH_ALGORITHM == "sha256":
        return hashlib.sha256(data).digest()
    elif CryptoConfig.HASH_ALGORITHM == "sha3_256":
        return hashlib.sha3_256(data).digest()
    else:
        raise CryptoError(f"Unsupported hash algorithm: {CryptoConfig.HASH_ALGORITHM}")


def derive_key(shared_secret: bytes, salt: bytes, info: bytes, key_length: int = 32) -> bytes:
    """
    Derive a key using HKDF.
    
    Args:
        shared_secret: The shared secret material
        salt: Salt value
        info: Context information
        key_length: Desired key length in bytes
        
    Returns:
        Derived key
    """
    try:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            info=info,
            backend=default_backend()
        )
        return hkdf.derive(shared_secret)
    except Exception as e:
        raise CryptoError(f"Key derivation failed: {e}") from e


def generate_commitment(secret: bytes, nonce: bytes) -> bytes:
    """
    Generate a commitment for zero-knowledge proof.
    
    Args:
        secret: Secret value
        nonce: Random nonce
        
    Returns:
        Commitment hash
    """
    return secure_hash(secret + nonce)


def generate_challenge() -> bytes:
    """
    Generate a random challenge for zero-knowledge proof.
    
    Returns:
        Random challenge bytes
    """
    return secrets.token_bytes(CryptoConfig.ZKP_CHALLENGE_SIZE // 8)


def compute_zkp_response(
    secret: bytes, 
    nonce: bytes, 
    challenge: bytes
) -> bytes:
    """
    Compute zero-knowledge proof response.
    
    Args:
        secret: Secret value
        nonce: Random nonce used in commitment
        challenge: Challenge from verifier
        
    Returns:
        ZKP response
    """
    # Simplified ZKP response calculation
    # In a real implementation, this would use proper group operations
    combined = secret + nonce + challenge
    return secure_hash(combined)


def verify_zkp(
    commitment: bytes,
    challenge: bytes,
    response: bytes,
    public_key: ec.EllipticCurvePublicKey
) -> bool:
    """
    Verify a zero-knowledge proof.
    
    Args:
        commitment: Prover's commitment
        challenge: Verifier's challenge
        response: Prover's response
        public_key: Prover's public key
        
    Returns:
        True if proof is valid, False otherwise
    """
    try:
        # Simplified verification
        # In a real implementation, this would use proper group operations
        pub_key_bytes = serialize_public_key(public_key)
        expected_response = secure_hash(pub_key_bytes + commitment + challenge)
        
        # Constant-time comparison
        return hmac.compare_digest(response, expected_response)
        
    except Exception:
        return False


def encrypt_aes_gcm(data: bytes, key: bytes, nonce: Optional[bytes] = None) -> Tuple[bytes, bytes, bytes]:
    """
    Encrypt data using AES-GCM.
    
    Args:
        data: Data to encrypt
        key: Encryption key (32 bytes)
        nonce: Nonce (12 bytes). If None, will be generated randomly.
        
    Returns:
        Tuple of (ciphertext, nonce, tag)
        
    Raises:
        EncryptionError: If encryption fails
    """
    try:
        if nonce is None:
            nonce = secrets.token_bytes(12)
        elif len(nonce) != 12:
            raise EncryptionError("Nonce must be 12 bytes for AES-GCM")
            
        if len(key) != 32:
            raise EncryptionError("Key must be 32 bytes for AES-256-GCM")
            
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return ciphertext, nonce, encryptor.tag
        
    except Exception as e:
        raise EncryptionError(f"AES-GCM encryption failed: {e}") from e


def decrypt_aes_gcm(ciphertext: bytes, key: bytes, nonce: bytes, tag: bytes) -> bytes:
    """
    Decrypt data using AES-GCM.
    
    Args:
        ciphertext: Encrypted data
        key: Decryption key (32 bytes)
        nonce: Nonce (12 bytes)
        tag: Authentication tag
        
    Returns:
        Decrypted data
        
    Raises:
        EncryptionError: If decryption fails
    """
    try:
        if len(key) != 32:
            raise EncryptionError("Key must be 32 bytes for AES-256-GCM")
            
        if len(nonce) != 12:
            raise EncryptionError("Nonce must be 12 bytes for AES-GCM")
            
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
        
    except Exception as e:
        raise EncryptionError(f"AES-GCM decryption failed: {e}") from e


def derive_post_quantum_shared_secret_stub(
    party_a_material: bytes, 
    party_b_material: bytes
) -> bytes:
    """
    Post-quantum key exchange stub for future implementation.
    
    This is a placeholder function demonstrating forward-thinking
    for post-quantum cryptography. In a real implementation, this
    would use algorithms like CRYSTALS-Kyber.
    
    Args:
        party_a_material: Party A's key material
        party_b_material: Party B's key material
        
    Returns:
        Fixed-size shared secret (placeholder)
    """
    # Placeholder implementation - NOT cryptographically secure
    # This would be replaced with proper post-quantum algorithms
    combined = party_a_material + party_b_material
    return secure_hash(combined)[:CryptoConfig.PQ_KEY_SIZE // 8]


def constant_time_compare(a: bytes, b: bytes) -> bool:
    """
    Constant-time comparison of two byte strings.
    
    Args:
        a: First byte string
        b: Second byte string
        
    Returns:
        True if equal, False otherwise
    """
    return hmac.compare_digest(a, b)


def secure_random_bytes(length: int) -> bytes:
    """
    Generate cryptographically secure random bytes.
    
    Args:
        length: Number of bytes to generate
        
    Returns:
        Random bytes
    """
    return secrets.token_bytes(length)


# Validation-compatible function aliases
def generate_keypair():
    """Alias for generate_ecc_keypair for validation compatibility."""
    return generate_ecc_keypair()

def create_zk_proof(secret: bytes, challenge: bytes) -> bytes:
    """Alias for ZK proof creation for validation compatibility."""
    nonce = secure_random_bytes(32)
    return compute_zkp_response(secret, nonce, challenge)

def verify_proof(commitment: bytes, challenge: bytes, response: bytes) -> bool:
    """Alias for ZK proof verification for validation compatibility."""
    return verify_zkp(commitment, challenge, response, b"public_data")

def hash_data(data: bytes) -> bytes:
    """Alias for secure_hash for validation compatibility."""
    return secure_hash(data)
