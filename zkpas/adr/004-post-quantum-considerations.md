# ADR-004: Post-Quantum Cryptography Considerations

**Status:** Proposed  
**Date:** 2025-06-29  
**Deciders:** Lead Systems Engineer

## Context

The ZKPAS (Zero-Knowledge Proof Authentication System) for IoT devices must consider the long-term security implications of quantum computing advances. While practical quantum computers capable of breaking current elliptic curve cryptography are not yet available, IoT devices deployed today may operate for 10-20 years, potentially outlasting the current cryptographic assumptions.

## Decision

We will implement a forward-thinking approach to post-quantum cryptography:

1. **Immediate Implementation**: Include a placeholder function `derive_post_quantum_shared_secret_stub()` that simulates post-quantum key exchange
2. **Architecture Design**: Structure the cryptographic module to allow easy swapping of algorithms
3. **Research Consideration**: Identify CRYSTALS-Kyber as a leading post-quantum key encapsulation mechanism for future implementation
4. **Documentation**: Clearly mark the placeholder as non-cryptographically secure

## Rationale

### Why Post-Quantum Resistance Matters for IoT

1. **Device Longevity**: IoT devices often have long operational lifespans (10-20 years)
2. **Update Challenges**: Many IoT devices cannot be easily updated in the field
3. **Quantum Timeline**: NIST estimates quantum computers capable of breaking RSA-2048 and ECC-256 could emerge within 10-15 years
4. **Regulatory Pressure**: Government agencies are already mandating post-quantum readiness

### Why a Stub Implementation Now

1. **Architecture Preparedness**: Ensures our system design can accommodate post-quantum algorithms
2. **Interface Stability**: Defines the expected function signature and behavior
3. **Testing Infrastructure**: Allows us to test quantum-resistant code paths
4. **Future-Proofing**: Minimizes refactoring needed when real PQ algorithms are integrated

### Why CRYSTALS-Kyber

1. **NIST Standardization**: Selected as NIST's primary post-quantum KEM standard
2. **Performance**: Relatively efficient compared to other post-quantum algorithms
3. **Security Analysis**: Extensive cryptanalysis and security evaluation
4. **Industry Adoption**: Growing support in major cryptographic libraries

## Implementation Details

```python
def derive_post_quantum_shared_secret_stub(
    party_a_material: bytes,
    party_b_material: bytes
) -> bytes:
    """
    Post-quantum key exchange stub for future implementation.

    WARNING: This is a placeholder - NOT cryptographically secure!
    """
    # Current: Simple hash-based placeholder
    combined = party_a_material + party_b_material
    return secure_hash(combined)[:PQ_KEY_SIZE // 8]

# Future implementation would replace with:
# def derive_kyber_shared_secret(
#     kyber_ciphertext: bytes,
#     kyber_private_key: bytes
# ) -> bytes:
#     return kyber_kem_decrypt(kyber_ciphertext, kyber_private_key)
```

## Consequences

### Positive

- **Future-Ready Architecture**: Easy to upgrade when PQ algorithms mature
- **Security Awareness**: Demonstrates consideration of long-term security
- **Testing Capability**: Can test hybrid classical/quantum-resistant scenarios
- **Risk Mitigation**: Reduces future technical debt

### Negative

- **Current Overhead**: Slight complexity increase in current implementation
- **False Security**: Risk of confusion about placeholder vs real security
- **Algorithm Selection Risk**: CRYSTALS-Kyber may not be the final choice

### Neutral

- **Performance Impact**: Minimal in simulation environment
- **Standards Evolution**: Will need updates as NIST standards finalize

## Compliance Considerations

- **NIST SP 800-208**: Guidelines for post-quantum cryptography migration
- **NSA/CISA Guidance**: Federal agency quantum-readiness requirements
- **ISO/IEC 23837**: Emerging international standards for quantum-safe cryptography

## Migration Path

1. **Phase 1** (Current): Implement stub with clear documentation
2. **Phase 2** (6-12 months): Integrate real CRYSTALS-Kyber implementation
3. **Phase 3** (1-2 years): Hybrid classical/post-quantum mode
4. **Phase 4** (2-5 years): Full post-quantum transition

## Monitoring

- Track NIST post-quantum standardization progress
- Monitor quantum computing advances (IBM, Google, IonQ quantum roadmaps)
- Evaluate performance of real PQ implementations in similar systems
- Review IoT-specific post-quantum guidance from security agencies

## References

- NIST Post-Quantum Cryptography Standards (FIPS 203, 204, 205)
- CRYSTALS-Kyber Algorithm Specification
- "Post-Quantum Cryptography for IoT: A Survey" (IEEE IoT Journal, 2024)
- NSA Quantum-Safe Cryptography Guidelines
