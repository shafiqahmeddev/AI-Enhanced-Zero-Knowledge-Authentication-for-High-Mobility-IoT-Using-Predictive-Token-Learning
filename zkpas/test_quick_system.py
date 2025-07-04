"""
Quick ZKPAS system test to identify and fix issues
"""

import asyncio
import sys
sys.path.append('.')

from app.events import EventBus, EventType
from app.components.trusted_authority import TrustedAuthority
from app.components.sliding_window_auth import SlidingWindowAuthenticator
from app.components.byzantine_resilience import ByzantineResilienceCoordinator, TrustAnchor
from shared.crypto_utils import generate_ecc_keypair, serialize_public_key


async def main():
    print("🔍 Quick ZKPAS System Check")
    
    # Test EventBus
    event_bus = EventBus()
    await event_bus.start()
    print("✅ EventBus started")
    
    # Test TrustedAuthority
    ta = TrustedAuthority("test_authority")
    print("✅ TrustedAuthority created")
    
    # Test device registration
    private_key, public_key = generate_ecc_keypair()
    serialized_key = serialize_public_key(public_key)
    result = await ta.register_device("test_device", serialized_key)
    print(f"✅ Device registration: {result}")
    
    # Test SlidingWindowAuthenticator
    sliding_auth = SlidingWindowAuthenticator(event_bus)
    print("✅ SlidingWindowAuthenticator created")
    
    # Test ByzantineResilienceCoordinator
    coordinator = ByzantineResilienceCoordinator(event_bus)
    network = coordinator.create_trust_network("test_network")
    anchor = TrustAnchor("test_anchor", event_bus)
    network.add_trust_anchor(anchor)
    print("✅ Byzantine resilience components working")
    
    # Cleanup
    await sliding_auth.shutdown()
    await event_bus.stop()
    
    print("🎉 All basic components working!")


if __name__ == "__main__":
    asyncio.run(main())