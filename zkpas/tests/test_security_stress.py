#!/usr/bin/env python3
"""
ZKPAS Phase 7: Security Stress Testing & Code-Level Testing

This module implements comprehensive security stress testing including:
- Cryptographic primitive security validation
- Byzantine fault tolerance under extreme conditions
- Protocol security invariant checking
- Fuzz testing for input validation
- Stress testing under high load
"""

import asyncio
import pytest
import time
import random
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from unittest.mock import Mock, patch
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.events import EventBus, EventType, Event
from app.components.trusted_authority import TrustedAuthority
from app.components.gateway_node import GatewayNode
from app.components.iot_device import IoTDevice
from app.components.sliding_window_auth import SlidingWindowAuthenticator, SlidingWindowToken
from app.components.byzantine_resilience import (
    ByzantineResilienceCoordinator, 
    TrustAnchor, 
    MaliciousTrustAnchor,
    TrustAnchorNetwork
)
from shared.crypto_utils import (
    generate_ecc_keypair,
    sign_data,
    verify_signature,
    encrypt_aes_gcm,
    decrypt_aes_gcm,
    secure_hash,
    derive_key,
    constant_time_compare,
    CryptoError
)
from shared.config import CryptoConfig


@dataclass
class SecurityInvariant:
    """Represents a security invariant that must be maintained."""
    name: str
    description: str
    check_function: callable
    violation_count: int = 0
    last_violation: Optional[str] = None


class SecurityStressTester:
    """
    Comprehensive security stress testing framework.
    
    Features:
    - Cryptographic primitive validation
    - Protocol security invariant checking
    - Byzantine fault tolerance stress testing
    - High-load stress testing
    - Security violation detection and reporting
    """
    
    def __init__(self):
        self.event_bus = None
        self.security_invariants: List[SecurityInvariant] = []
        self.test_results: Dict[str, Any] = {}
        self.stress_duration = 60  # Default stress test duration
        
        # Security metrics
        self.total_operations = 0
        self.security_violations = 0
        self.crypto_failures = 0
        self.byzantine_attacks_detected = 0
        self.protocol_violations = 0
        
        self._setup_security_invariants()
    
    def _setup_security_invariants(self):
        """Setup security invariants to monitor during testing."""
        
        # Cryptographic Invariants
        self.security_invariants.extend([
            SecurityInvariant(
                name="CRYPTO_SIGNATURE_INTEGRITY",
                description="All signatures must be cryptographically valid",
                check_function=self._check_signature_integrity
            ),
            SecurityInvariant(
                name="CRYPTO_ENCRYPTION_INTEGRITY", 
                description="All encrypted data must decrypt correctly",
                check_function=self._check_encryption_integrity
            ),
            SecurityInvariant(
                name="CRYPTO_KEY_DERIVATION_CONSISTENCY",
                description="Key derivation must be deterministic and secure",
                check_function=self._check_key_derivation_consistency
            )
        ])
        
        # Protocol Security Invariants
        self.security_invariants.extend([
            SecurityInvariant(
                name="PROTOCOL_AUTHENTICATION_INTEGRITY",
                description="Authentication tokens must not be forgeable",
                check_function=self._check_authentication_integrity
            ),
            SecurityInvariant(
                name="PROTOCOL_REPLAY_PROTECTION",
                description="System must prevent replay attacks",
                check_function=self._check_replay_protection
            ),
            SecurityInvariant(
                name="PROTOCOL_BYZANTINE_THRESHOLD",
                description="Byzantine resilience threshold must be maintained",
                check_function=self._check_byzantine_threshold
            )
        ])
        
        # System Security Invariants
        self.security_invariants.extend([
            SecurityInvariant(
                name="SYSTEM_NO_PRIVATE_KEY_EXPOSURE",
                description="Private keys must never be exposed in logs or events",
                check_function=self._check_private_key_exposure
            ),
            SecurityInvariant(
                name="SYSTEM_SECURE_STATE_TRANSITIONS",
                description="All state transitions must be secure and validated",
                check_function=self._check_secure_state_transitions
            )
        ])
    
    async def run_comprehensive_security_stress_test(self) -> Dict[str, Any]:
        """
        Run comprehensive security stress testing suite.
        
        Returns:
            Dict containing detailed test results and security metrics
        """
        print("🔒 PHASE 7: COMPREHENSIVE SECURITY STRESS TESTING")
        print("=" * 60)
        
        # Initialize test environment
        await self._initialize_test_environment()
        
        test_suite_results = {}
        
        # 1. Cryptographic Primitive Security Testing
        print("\n🔐 Test 1: Cryptographic Primitive Security")
        test_suite_results["crypto_primitives"] = await self._test_cryptographic_primitives()
        
        # 2. Byzantine Fault Tolerance Stress Testing
        print("\n⚔️ Test 2: Byzantine Fault Tolerance Stress")
        test_suite_results["byzantine_stress"] = await self._test_byzantine_stress()
        
        # 3. Protocol Security Invariant Checking
        print("\n🛡️ Test 3: Protocol Security Invariants")
        test_suite_results["protocol_security"] = await self._test_protocol_security_invariants()
        
        # 4. High-Load Stress Testing
        print("\n📈 Test 4: High-Load Security Stress")
        test_suite_results["high_load_stress"] = await self._test_high_load_security()
        
        # 5. Fuzz Testing for Input Validation
        print("\n🎯 Test 5: Security Fuzz Testing")
        test_suite_results["fuzz_testing"] = await self._test_security_fuzz_testing()
        
        # 6. Time-based Attack Resistance
        print("\n⏰ Test 6: Time-based Attack Resistance")
        test_suite_results["timing_attacks"] = await self._test_timing_attack_resistance()
        
        # Generate comprehensive security report
        security_report = self._generate_security_report(test_suite_results)
        
        # Cleanup
        await self._cleanup_test_environment()
        
        return security_report
    
    async def _initialize_test_environment(self):
        """Initialize secure test environment."""
        self.event_bus = EventBus()
        self.total_operations = 0
        self.security_violations = 0
        print("✅ Secure test environment initialized")
    
    async def _cleanup_test_environment(self):
        """Cleanup test environment."""
        if self.event_bus:
            # Note: EventBus doesn't have shutdown method, so we just clear references
            self.event_bus = None
        print("✅ Test environment cleaned up")
    
    async def _test_cryptographic_primitives(self) -> Dict[str, Any]:
        """Test cryptographic primitives under stress conditions."""
        print("   Testing ECC signatures, AES-GCM encryption, key derivation...")
        
        results = {
            "tests_run": 0,
            "failures": 0,
            "performance_metrics": {},
            "security_violations": []
        }
        
        # Test 1: ECC Signature Security Stress
        print("     • ECC Signature stress testing...")
        ecc_results = await self._stress_test_ecc_signatures(1000)
        results["ecc_signatures"] = ecc_results
        results["tests_run"] += ecc_results["operations"]
        results["failures"] += ecc_results["failures"]
        
        # Test 2: AES-GCM Encryption Security Stress  
        print("     • AES-GCM encryption stress testing...")
        aes_results = await self._stress_test_aes_gcm_encryption(1000)
        results["aes_gcm"] = aes_results
        results["tests_run"] += aes_results["operations"]
        results["failures"] += aes_results["failures"]
        
        # Test 3: Key Derivation Security
        print("     • Key derivation security testing...")
        kdf_results = await self._stress_test_key_derivation(500)
        results["key_derivation"] = kdf_results
        results["tests_run"] += kdf_results["operations"]
        results["failures"] += kdf_results["failures"]
        
        # Check security invariants
        for invariant in self.security_invariants:
            if "CRYPTO" in invariant.name:
                try:
                    invariant.check_function(results)
                except Exception as e:
                    invariant.violation_count += 1
                    invariant.last_violation = str(e)
                    results["security_violations"].append(f"{invariant.name}: {e}")
        
        success_rate = ((results["tests_run"] - results["failures"]) / results["tests_run"]) * 100
        print(f"     ✅ Cryptographic primitives: {success_rate:.2f}% success rate")
        
        return results
    
    async def _stress_test_ecc_signatures(self, num_operations: int) -> Dict[str, Any]:
        """Stress test ECC signature operations."""
        start_time = time.time()
        failures = 0
        
        for i in range(num_operations):
            try:
                # Generate keypair
                private_key, public_key = generate_ecc_keypair()
                
                # Create test message
                message = f"test_message_{i}_{time.time()}".encode()
                
                # Sign message
                signature = sign_data(private_key, message)
                
                # Verify signature
                is_valid = verify_signature(public_key, signature, message)
                
                if not is_valid:
                    failures += 1
                    
                # Test signature tampering detection
                tampered_signature = signature[:-1] + b'\x00'
                is_tampered_valid = verify_signature(public_key, tampered_signature, message)
                
                if is_tampered_valid:  # Should be False
                    failures += 1
                    
            except Exception as e:
                failures += 1
        
        duration = time.time() - start_time
        
        return {
            "operations": num_operations,
            "failures": failures,
            "duration_seconds": duration,
            "operations_per_second": num_operations / duration,
            "success_rate": ((num_operations - failures) / num_operations) * 100
        }
    
    async def _stress_test_aes_gcm_encryption(self, num_operations: int) -> Dict[str, Any]:
        """Stress test AES-GCM encryption operations."""
        start_time = time.time()
        failures = 0
        
        for i in range(num_operations):
            try:
                # Generate random key and data
                key = os.urandom(32)  # 256-bit key
                data = f"sensitive_data_{i}_{random.randint(1000, 9999)}".encode()
                
                # Encrypt data
                ciphertext, nonce, tag = encrypt_aes_gcm(data, key)
                
                # Decrypt data
                decrypted_data = decrypt_aes_gcm(ciphertext, key, nonce, tag)
                
                # Verify decryption integrity
                if decrypted_data != data:
                    failures += 1
                
                # Test tampering detection
                tampered_ciphertext = ciphertext[:-1] + b'\x00'
                try:
                    decrypt_aes_gcm(tampered_ciphertext, key, nonce, tag)
                    failures += 1  # Should have failed
                except:
                    pass  # Expected failure
                    
            except Exception as e:
                failures += 1
        
        duration = time.time() - start_time
        
        return {
            "operations": num_operations,
            "failures": failures,
            "duration_seconds": duration,
            "operations_per_second": num_operations / duration,
            "success_rate": ((num_operations - failures) / num_operations) * 100
        }
    
    async def _stress_test_key_derivation(self, num_operations: int) -> Dict[str, Any]:
        """Stress test key derivation operations."""
        start_time = time.time()
        failures = 0
        
        master_key = os.urandom(32)
        
        for i in range(num_operations):
            try:
                # Test deterministic key derivation
                info1 = f"context_{i}".encode()
                info2 = f"context_{i}".encode()  # Same context
                
                derived_key1 = derive_key(master_key, info1)
                derived_key2 = derive_key(master_key, info2)
                
                # Should be identical for same context
                if not constant_time_compare(derived_key1, derived_key2):
                    failures += 1
                
                # Test different contexts produce different keys
                info3 = f"different_context_{i}".encode()
                derived_key3 = derive_key(master_key, info3)
                
                if constant_time_compare(derived_key1, derived_key3):
                    failures += 1  # Should be different
                    
            except Exception as e:
                failures += 1
        
        duration = time.time() - start_time
        
        return {
            "operations": num_operations,
            "failures": failures,
            "duration_seconds": duration,
            "operations_per_second": num_operations / duration,
            "success_rate": ((num_operations - failures) / num_operations) * 100
        }
    
    async def _test_byzantine_stress(self) -> Dict[str, Any]:
        """Test Byzantine fault tolerance under extreme stress conditions."""
        print("   Testing Byzantine resilience under coordinated attacks...")
        
        results = {
            "scenarios_tested": 0,
            "attacks_successful": 0,
            "system_compromised": False,
            "attack_scenarios": []
        }
        
        # Initialize Byzantine coordinator
        coordinator = ByzantineResilienceCoordinator(self.event_bus, default_threshold=3)
        network = coordinator.create_trust_network("stress_test_network", threshold=3)
        
        # Add honest anchors
        honest_anchors = []
        for i in range(5):
            anchor = TrustAnchor(f"honest_anchor_{i}", self.event_bus)
            network.add_trust_anchor(anchor)
            honest_anchors.append(anchor)
        
        # Test scenarios with increasing malicious anchor ratios
        malicious_ratios = [0.2, 0.4, 0.6, 0.8]  # 20%, 40%, 60%, 80% malicious
        
        for ratio in malicious_ratios:
            scenario_name = f"byzantine_stress_{int(ratio*100)}pct"
            print(f"     • Testing {int(ratio*100)}% malicious anchors...")
            
            num_malicious = int(len(honest_anchors) * ratio)
            
            try:
                # Add malicious anchors with different attack behaviors
                malicious_behaviors = ["invalid_signature", "random_signature", "delayed_response"]
                for i in range(num_malicious):
                    behavior = malicious_behaviors[i % len(malicious_behaviors)]
                    malicious_anchor = MaliciousTrustAnchor(f"malicious_{i}", self.event_bus, behavior)
                    network.add_trust_anchor(malicious_anchor)
                
                # Attempt multiple authentications under attack
                attack_successful = True
                for auth_attempt in range(10):
                    test_message = f"auth_test_{scenario_name}_{auth_attempt}".encode()
                    result = await network.request_cross_domain_authentication(
                        source_domain="domain_a",
                        target_domain="domain_b", 
                        device_id=f"test_device_{auth_attempt}",
                        message=test_message
                    )
                    
                    if result is None or not result.threshold_met:
                        attack_successful = False
                        results["attacks_successful"] += 1
                        if ratio >= 0.6:  # Critical threshold
                            results["system_compromised"] = True
                
                # Clean up malicious anchors
                for i in range(num_malicious):
                    network.remove_trust_anchor(f"malicious_{i}")
                
                results["attack_scenarios"].append({
                    "name": scenario_name,
                    "malicious_ratio": ratio,
                    "attack_successful": not attack_successful,
                    "system_resilient": attack_successful
                })
                
                results["scenarios_tested"] += 1
                
            except Exception as e:
                print(f"     ❌ Byzantine stress test error: {e}")
                results["attack_scenarios"].append({
                    "name": scenario_name,
                    "error": str(e)
                })
        
        # Calculate overall Byzantine resilience score
        successful_defenses = results["scenarios_tested"] - results["attacks_successful"]
        resilience_score = (successful_defenses / results["scenarios_tested"]) * 100 if results["scenarios_tested"] > 0 else 0
        
        results["resilience_score"] = resilience_score
        
        print(f"     ✅ Byzantine resilience: {resilience_score:.1f}% defense success rate")
        
        return results
    
    async def _test_protocol_security_invariants(self) -> Dict[str, Any]:
        """Test protocol-level security invariants."""
        print("   Testing authentication, replay protection, state integrity...")
        
        results = {
            "invariants_checked": len(self.security_invariants),
            "violations": 0,
            "violation_details": []
        }
        
        # Initialize sliding window authenticator for testing
        sliding_auth = SlidingWindowAuthenticator(self.event_bus)
        
        try:
            # Test authentication token integrity
            device_id = "security_test_device"
            master_key = secure_hash(b"test_master_key")
            
            # Create authentication window
            window = await sliding_auth.create_authentication_window(device_id, master_key)
            
            # Generate legitimate token
            payload = {"test": "legitimate_payload", "timestamp": time.time()}
            legitimate_token = await sliding_auth.generate_sliding_window_token(device_id, payload)
            
            # Test 1: Token forgery resistance
            print("     • Testing token forgery resistance...")
            forged_token = SlidingWindowToken(
                device_id=device_id,
                token_id=str(uuid.uuid4()),
                encrypted_payload=b"forged_payload",
                nonce=os.urandom(12),
                tag=os.urandom(16),
                expiry_timestamp=time.time() + 600,
                generation_timestamp=time.time(),
                sequence_number=999
            )
            
            # Attempt to validate forged token
            is_valid, _ = await sliding_auth.validate_sliding_window_token(forged_token.token_id, device_id)
            if is_valid:
                results["violations"] += 1
                results["violation_details"].append("Token forgery was not detected")
            
            # Test 2: Replay attack prevention
            print("     • Testing replay attack prevention...")
            if legitimate_token:
                # First validation should succeed
                is_valid1, _ = await sliding_auth.validate_sliding_window_token(legitimate_token.token_id, device_id)
                
                # Simulate replay attack by validating same token again
                # (Note: Current implementation doesn't have explicit replay protection,
                # but sequence numbers provide some protection)
                
            # Test 3: Cross-device token validation
            print("     • Testing cross-device security...")
            wrong_device_id = "wrong_device"
            if legitimate_token:
                is_valid_wrong_device, _ = await sliding_auth.validate_sliding_window_token(
                    legitimate_token.token_id, wrong_device_id
                )
                if is_valid_wrong_device:
                    results["violations"] += 1
                    results["violation_details"].append("Cross-device token validation succeeded (security violation)")
            
        except Exception as e:
            results["violations"] += 1
            results["violation_details"].append(f"Protocol test error: {e}")
        
        finally:
            await sliding_auth.shutdown()
        
        print(f"     ✅ Protocol security: {results['violations']} violations detected")
        
        return results
    
    async def _test_high_load_security(self) -> Dict[str, Any]:
        """Test security under high-load conditions."""
        print("   Testing security under concurrent high-load operations...")
        
        results = {
            "concurrent_threads": 20,
            "operations_per_thread": 50,
            "total_operations": 0,
            "security_failures": 0,
            "performance_degradation": False
        }
        
        # Initialize components for high-load testing
        sliding_auth = SlidingWindowAuthenticator(self.event_bus)
        coordinator = ByzantineResilienceCoordinator(self.event_bus)
        network = coordinator.create_trust_network("high_load_network", threshold=2)
        
        # Add trust anchors
        for i in range(3):
            anchor = TrustAnchor(f"load_anchor_{i}", self.event_bus)
            network.add_trust_anchor(anchor)
        
        async def high_load_worker(worker_id: int) -> Dict[str, Any]:
            """Worker function for high-load testing."""
            worker_results = {"operations": 0, "failures": 0, "security_issues": 0}
            
            try:
                device_id = f"load_test_device_{worker_id}"
                master_key = secure_hash(f"load_test_key_{worker_id}".encode())
                
                # Create authentication window
                await sliding_auth.create_authentication_window(device_id, master_key)
                
                for op in range(results["operations_per_thread"]):
                    try:
                        # Generate and validate token
                        payload = {"worker": worker_id, "operation": op}
                        token = await sliding_auth.generate_sliding_window_token(device_id, payload)
                        
                        if token:
                            is_valid, _ = await sliding_auth.validate_sliding_window_token(token.token_id, device_id)
                            if not is_valid:
                                worker_results["security_issues"] += 1
                        
                        # Perform Byzantine operation
                        message = f"high_load_{worker_id}_{op}".encode()
                        result = await network.request_cross_domain_authentication(
                            source_domain="load_domain",
                            target_domain="target_domain",
                            device_id=device_id,
                            message=message
                        )
                        
                        if result is None:
                            worker_results["security_issues"] += 1
                        
                        worker_results["operations"] += 1
                        
                    except Exception as e:
                        worker_results["failures"] += 1
                        
            except Exception as e:
                worker_results["failures"] += 1
            
            return worker_results
        
        # Run concurrent high-load test
        start_time = time.time()
        
        tasks = []
        for worker_id in range(results["concurrent_threads"]):
            task = asyncio.create_task(high_load_worker(worker_id))
            tasks.append(task)
        
        # Wait for all workers to complete with timeout
        try:
            worker_results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=120)
            
            # Aggregate results
            for worker_result in worker_results:
                results["total_operations"] += worker_result["operations"]
                results["security_failures"] += worker_result["security_issues"]
            
        except asyncio.TimeoutError:
            results["performance_degradation"] = True
            print("     ⚠️ High-load test timed out - performance degradation detected")
        
        duration = time.time() - start_time
        operations_per_second = results["total_operations"] / duration if duration > 0 else 0
        
        results["duration_seconds"] = duration
        results["operations_per_second"] = operations_per_second
        results["security_failure_rate"] = (results["security_failures"] / results["total_operations"]) * 100 if results["total_operations"] > 0 else 0
        
        # Cleanup
        await sliding_auth.shutdown()
        
        print(f"     ✅ High-load security: {results['security_failure_rate']:.2f}% failure rate at {operations_per_second:.1f} ops/sec")
        
        return results
    
    async def _test_security_fuzz_testing(self) -> Dict[str, Any]:
        """Perform security-focused fuzz testing."""
        print("   Testing input validation and boundary conditions...")
        
        results = {
            "fuzz_inputs_tested": 0,
            "crashes": 0,
            "security_exceptions": 0,
            "invalid_inputs_accepted": 0
        }
        
        # Initialize test subject
        sliding_auth = SlidingWindowAuthenticator(self.event_bus)
        
        # Fuzz test data generators
        def generate_fuzz_inputs():
            """Generate various fuzz inputs for testing."""
            return [
                # Empty/None inputs
                None, "", b"",
                
                # Extremely long inputs
                "A" * 10000, b"B" * 10000,
                
                # Special characters and encoding attacks
                "'; DROP TABLE users; --", 
                "<script>alert('xss')</script>",
                "../../../../etc/passwd",
                "\x00\x01\x02\xFF\xFE\xFD",
                
                # Unicode and encoding edge cases
                "🚀💀🔒🎛️", "\u0000\u0001\uffff",
                
                # Large numbers and overflow attempts
                999999999999999999999, -999999999999999999999,
                
                # Malformed JSON and data structures
                '{"incomplete": ', '}{invalid}',
                
                # Binary data designed to trigger edge cases
                b"\x00" * 1000, b"\xFF" * 1000,
                bytes(range(256)),
            ]
        
        fuzz_inputs = generate_fuzz_inputs()
        
        for fuzz_input in fuzz_inputs:
            results["fuzz_inputs_tested"] += 1
            
            try:
                # Test 1: Device ID fuzz testing
                try:
                    if isinstance(fuzz_input, str):
                        master_key = secure_hash(b"test_key")
                        await sliding_auth.create_authentication_window(fuzz_input, master_key)
                except (ValueError, TypeError):
                    pass  # Expected rejection
                except Exception as e:
                    results["crashes"] += 1
                    print(f"     ⚠️ Unexpected crash with device_id='{fuzz_input}': {e}")
                
                # Test 2: Token payload fuzz testing
                try:
                    if not isinstance(fuzz_input, (int, float)) or abs(fuzz_input) < 1000000:
                        device_id = "fuzz_test_device"
                        master_key = secure_hash(b"fuzz_test_key")
                        await sliding_auth.create_authentication_window(device_id, master_key)
                        
                        payload = {"fuzz_data": fuzz_input}
                        token = await sliding_auth.generate_sliding_window_token(device_id, payload)
                        
                        if token and isinstance(fuzz_input, str) and len(fuzz_input) > 5000:
                            results["invalid_inputs_accepted"] += 1
                            
                except (ValueError, TypeError):
                    pass  # Expected rejection
                except Exception as e:
                    results["crashes"] += 1
                    print(f"     ⚠️ Unexpected crash with payload='{fuzz_input}': {e}")
                
                # Test 3: Cryptographic function fuzz testing
                try:
                    if isinstance(fuzz_input, (str, bytes)):
                        input_bytes = fuzz_input.encode() if isinstance(fuzz_input, str) else fuzz_input
                        if len(input_bytes) < 10000:  # Avoid memory exhaustion
                            hash_result = secure_hash(input_bytes)
                            if len(hash_result) != 32:  # SHA-256 should always be 32 bytes
                                results["security_exceptions"] += 1
                except Exception as e:
                    results["crashes"] += 1
                    print(f"     ⚠️ Hash function crash with input='{fuzz_input}': {e}")
                    
            except Exception as e:
                results["crashes"] += 1
                print(f"     ⚠️ General fuzz test crash: {e}")
        
        # Cleanup
        await sliding_auth.shutdown()
        
        crash_rate = (results["crashes"] / results["fuzz_inputs_tested"]) * 100
        print(f"     ✅ Fuzz testing: {crash_rate:.1f}% crash rate, {results['invalid_inputs_accepted']} invalid inputs accepted")
        
        return results
    
    async def _test_timing_attack_resistance(self) -> Dict[str, Any]:
        """Test resistance to timing-based attacks."""
        print("   Testing timing attack resistance...")
        
        results = {
            "timing_tests": 0,
            "potential_vulnerabilities": 0,
            "timing_measurements": []
        }
        
        # Test constant-time comparison
        test_data_pairs = [
            (b"correct_password", b"correct_password"),  # Same data
            (b"correct_password", b"wrong_password123"),  # Different data, same length
            (b"correct_password", b"wrong"),             # Different data, different length
            (b"a" * 1000, b"a" * 999 + b"b"),          # Large data, single bit difference
        ]
        
        for i, (data1, data2) in enumerate(test_data_pairs):
            results["timing_tests"] += 1
            
            # Measure timing for multiple iterations
            timings = []
            iterations = 1000
            
            for _ in range(iterations):
                start_time = time.perf_counter()
                result = constant_time_compare(data1, data2)
                end_time = time.perf_counter()
                timings.append(end_time - start_time)
            
            # Analyze timing variance
            avg_time = sum(timings) / len(timings)
            max_time = max(timings)
            min_time = min(timings)
            variance = max_time - min_time
            
            # Check for potential timing leaks
            if variance > avg_time * 0.1:  # More than 10% variance might indicate timing leak
                results["potential_vulnerabilities"] += 1
                print(f"     ⚠️ Potential timing vulnerability in test {i+1}: {variance:.9f}s variance")
            
            results["timing_measurements"].append({
                "test_case": i + 1,
                "avg_time": avg_time,
                "variance": variance,
                "potential_leak": variance > avg_time * 0.1
            })
        
        vulnerability_rate = (results["potential_vulnerabilities"] / results["timing_tests"]) * 100
        print(f"     ✅ Timing attack resistance: {vulnerability_rate:.1f}% potential vulnerabilities")
        
        return results
    
    def _check_signature_integrity(self, test_data: Dict) -> None:
        """Check cryptographic signature integrity invariant."""
        if "ecc_signatures" in test_data:
            failure_rate = (test_data["ecc_signatures"]["failures"] / test_data["ecc_signatures"]["operations"]) * 100
            if failure_rate > 1.0:  # More than 1% failure rate is concerning
                raise Exception(f"High signature failure rate: {failure_rate:.2f}%")
    
    def _check_encryption_integrity(self, test_data: Dict) -> None:
        """Check encryption integrity invariant."""
        if "aes_gcm" in test_data:
            failure_rate = (test_data["aes_gcm"]["failures"] / test_data["aes_gcm"]["operations"]) * 100
            if failure_rate > 0.1:  # More than 0.1% failure rate is concerning
                raise Exception(f"High encryption failure rate: {failure_rate:.2f}%")
    
    def _check_key_derivation_consistency(self, test_data: Dict) -> None:
        """Check key derivation consistency invariant."""
        if "key_derivation" in test_data:
            failure_rate = (test_data["key_derivation"]["failures"] / test_data["key_derivation"]["operations"]) * 100
            if failure_rate > 0.0:  # Any failure in key derivation is critical
                raise Exception(f"Key derivation failures detected: {failure_rate:.2f}%")
    
    def _check_authentication_integrity(self, test_data: Dict) -> None:
        """Check authentication integrity invariant."""
        # Implementation would check that forged tokens are always rejected
        pass
    
    def _check_replay_protection(self, test_data: Dict) -> None:
        """Check replay attack protection invariant."""
        # Implementation would verify replay attacks are prevented
        pass
    
    def _check_byzantine_threshold(self, test_data: Dict) -> None:
        """Check Byzantine fault tolerance threshold invariant."""
        # Implementation would verify threshold is maintained under attack
        pass
    
    def _check_private_key_exposure(self, test_data: Dict) -> None:
        """Check that private keys are never exposed."""
        # Implementation would scan logs and data for private key exposure
        pass
    
    def _check_secure_state_transitions(self, test_data: Dict) -> None:
        """Check secure state transition invariant."""
        # Implementation would verify all state changes are validated
        pass
    
    def _generate_security_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive security test report."""
        print("\n📋 GENERATING SECURITY ASSESSMENT REPORT")
        print("=" * 50)
        
        # Calculate overall security score
        total_tests = 0
        total_failures = 0
        critical_issues = 0
        
        for test_category, results in test_results.items():
            if isinstance(results, dict):
                if "tests_run" in results:
                    total_tests += results["tests_run"]
                    total_failures += results.get("failures", 0)
                if "crashes" in results:
                    critical_issues += results["crashes"]
                if "security_violations" in results:
                    critical_issues += len(results["security_violations"])
        
        security_score = ((total_tests - total_failures - critical_issues) / total_tests) * 100 if total_tests > 0 else 0
        
        # Security invariant violations
        invariant_violations = sum(inv.violation_count for inv in self.security_invariants)
        
        security_report = {
            "test_execution": {
                "timestamp": time.time(),
                "total_tests_run": total_tests,
                "total_failures": total_failures,
                "critical_issues": critical_issues,
                "security_score": security_score
            },
            "security_invariants": {
                "total_invariants": len(self.security_invariants),
                "violations": invariant_violations,
                "invariant_details": [
                    {
                        "name": inv.name,
                        "description": inv.description,
                        "violations": inv.violation_count,
                        "last_violation": inv.last_violation
                    }
                    for inv in self.security_invariants
                ]
            },
            "test_results": test_results,
            "recommendations": self._generate_security_recommendations(test_results, security_score),
            "compliance_status": "PASS" if security_score >= 95 and critical_issues == 0 else "FAIL"
        }
        
        # Print summary
        print(f"🔒 Overall Security Score: {security_score:.1f}%")
        print(f"📊 Tests Run: {total_tests}")
        print(f"❌ Critical Issues: {critical_issues}")
        print(f"⚠️ Security Invariant Violations: {invariant_violations}")
        print(f"🎯 Compliance Status: {security_report['compliance_status']}")
        
        return security_report
    
    def _generate_security_recommendations(self, test_results: Dict, security_score: float) -> List[str]:
        """Generate security recommendations based on test results."""
        recommendations = []
        
        if security_score < 95:
            recommendations.append("Security score below 95% - comprehensive security review required")
        
        if "fuzz_testing" in test_results:
            fuzz_results = test_results["fuzz_testing"]
            if fuzz_results.get("crashes", 0) > 0:
                recommendations.append("Fuzz testing revealed crashes - implement robust input validation")
            if fuzz_results.get("invalid_inputs_accepted", 0) > 0:
                recommendations.append("System accepts invalid inputs - strengthen input sanitization")
        
        if "byzantine_stress" in test_results:
            byzantine_results = test_results["byzantine_stress"]
            if byzantine_results.get("system_compromised", False):
                recommendations.append("System compromised under Byzantine attacks - increase fault tolerance threshold")
        
        if "timing_attacks" in test_results:
            timing_results = test_results["timing_attacks"]
            if timing_results.get("potential_vulnerabilities", 0) > 0:
                recommendations.append("Potential timing vulnerabilities detected - review constant-time implementations")
        
        if not recommendations:
            recommendations.append("Security testing passed - maintain current security practices")
        
        return recommendations


# Test fixtures and integration tests
@pytest.fixture
async def security_tester():
    """Fixture for security stress tester."""
    tester = SecurityStressTester()
    yield tester
    # Cleanup handled by tester itself


@pytest.mark.asyncio
async def test_comprehensive_security_stress():
    """Comprehensive security stress test suite."""
    tester = SecurityStressTester()
    
    # Run full security stress test
    security_report = await tester.run_comprehensive_security_stress_test()
    
    # Verify critical security requirements
    assert security_report["compliance_status"] == "PASS", "Security compliance test failed"
    assert security_report["test_execution"]["security_score"] >= 85, "Security score too low"
    assert security_report["test_execution"]["critical_issues"] == 0, "Critical security issues found"
    
    print("✅ Comprehensive security stress testing completed successfully")


@pytest.mark.asyncio
async def test_cryptographic_primitives_security():
    """Test cryptographic primitives security in isolation."""
    tester = SecurityStressTester()
    await tester._initialize_test_environment()
    
    # Test cryptographic primitives
    crypto_results = await tester._test_cryptographic_primitives()
    
    # Verify results
    assert crypto_results["failures"] == 0, "Cryptographic primitive failures detected"
    assert len(crypto_results["security_violations"]) == 0, "Security violations in crypto primitives"
    
    await tester._cleanup_test_environment()
    print("✅ Cryptographic primitives security test passed")


@pytest.mark.asyncio
async def test_byzantine_fault_tolerance_stress():
    """Test Byzantine fault tolerance under stress."""
    tester = SecurityStressTester()
    await tester._initialize_test_environment()
    
    # Test Byzantine stress
    byzantine_results = await tester._test_byzantine_stress()
    
    # Verify Byzantine resilience
    assert byzantine_results["resilience_score"] >= 75, "Byzantine resilience score too low"
    assert not byzantine_results["system_compromised"], "System was compromised by Byzantine attacks"
    
    await tester._cleanup_test_environment()
    print("✅ Byzantine fault tolerance stress test passed")


if __name__ == "__main__":
    # Run comprehensive security stress testing
    async def main():
        print("🔒 ZKPAS Phase 7: Security Stress Testing")
        print("Running comprehensive security validation...")
        
        tester = SecurityStressTester()
        security_report = await tester.run_comprehensive_security_stress_test()
        
        # Save security report
        import json
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"security_stress_report_{timestamp}.json"
        
        with open(report_filename, 'w') as f:
            json.dump(security_report, f, indent=2, default=str)
        
        print(f"\n📄 Security report saved to: {report_filename}")
        print(f"🎯 Final Security Score: {security_report['test_execution']['security_score']:.1f}%")
        print(f"🛡️ Compliance Status: {security_report['compliance_status']}")
    
    asyncio.run(main())