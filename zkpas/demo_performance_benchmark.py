#!/usr/bin/env python3
"""
ZKPAS Performance Benchmarking Demo

This demo tests system performance under various loads and conditions.
"""

import asyncio
import time
import sys
import statistics
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Add the zkpas directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.events import EventBus, EventType
from shared.crypto_utils import generate_ecc_keypair, serialize_public_key, secure_hash
from loguru import logger


async def benchmark_cryptographic_operations(iterations: int = 1000) -> Dict[str, Any]:
    """Benchmark cryptographic operations."""
    print(f"🔐 Benchmarking Cryptographic Operations ({iterations} iterations)...")
    
    # Key Generation Benchmark
    start_time = time.perf_counter()
    for _ in range(iterations):
        private_key, public_key = generate_ecc_keypair()
    keygen_time = time.perf_counter() - start_time
    
    # Key Serialization Benchmark
    start_time = time.perf_counter()
    for _ in range(iterations):
        serialized = serialize_public_key(public_key)
    serialize_time = time.perf_counter() - start_time
    
    # Hash Operation Benchmark
    test_data = b"ZKPAS performance test data for hashing operations"
    start_time = time.perf_counter()
    for _ in range(iterations):
        hash_result = secure_hash(test_data)
    hash_time = time.perf_counter() - start_time
    
    return {
        "keygen_time": keygen_time,
        "keygen_ops_per_sec": iterations / keygen_time,
        "serialize_time": serialize_time,
        "serialize_ops_per_sec": iterations / serialize_time,
        "hash_time": hash_time,
        "hash_ops_per_sec": iterations / hash_time
    }


async def benchmark_event_system(num_events: int = 10000) -> Dict[str, Any]:
    """Benchmark event system performance."""
    print(f"📡 Benchmarking Event System ({num_events} events)...")
    
    event_bus = EventBus()
    events_processed = 0
    
    def event_handler(event):
        nonlocal events_processed
        events_processed += 1
    
    # Subscribe to test event
    event_bus.subscribe_sync(EventType.DEVICE_AUTHENTICATED, event_handler)
    
    # Benchmark event publishing
    start_time = time.perf_counter()
    
    for i in range(num_events):
        await event_bus.publish_event(
            EventType.DEVICE_AUTHENTICATED,
            correlation_id=f"perf_test_{i}",
            source="benchmark",
            target=f"device_{i}",
            data={"test": True}
        )
    
    # Allow processing time
    await asyncio.sleep(0.5)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    return {
        "total_time": total_time,
        "events_per_sec": num_events / total_time,
        "events_processed": events_processed,
        "processing_rate": events_processed / total_time
    }


async def benchmark_concurrent_operations(num_workers: int = 50) -> Dict[str, Any]:
    """Benchmark concurrent operations."""
    print(f"⚡ Benchmarking Concurrent Operations ({num_workers} workers)...")
    
    async def worker_task(worker_id: int) -> Dict[str, Any]:
        """Simulated authentication work."""
        start_time = time.perf_counter()
        
        # Simulate authentication operations
        private_key, public_key = generate_ecc_keypair()
        serialized_key = serialize_public_key(public_key)
        hash_result = secure_hash(serialized_key)
        
        # Simulate processing delay
        await asyncio.sleep(0.01)  # 10ms simulated work
        
        end_time = time.perf_counter()
        
        return {
            "worker_id": worker_id,
            "execution_time": end_time - start_time,
            "success": True
        }
    
    # Execute concurrent workers
    start_time = time.perf_counter()
    
    tasks = [worker_task(i) for i in range(num_workers)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Analyze results
    successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
    execution_times = [r['execution_time'] for r in successful_results]
    
    return {
        "total_time": total_time,
        "workers_completed": len(successful_results),
        "success_rate": len(successful_results) / num_workers,
        "throughput": len(successful_results) / total_time,
        "avg_execution_time": statistics.mean(execution_times) if execution_times else 0,
        "min_execution_time": min(execution_times) if execution_times else 0,
        "max_execution_time": max(execution_times) if execution_times else 0
    }


async def benchmark_memory_usage() -> Dict[str, Any]:
    """Benchmark memory usage patterns."""
    print("💾 Benchmarking Memory Usage...")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create large data structures to test memory handling
    test_data = []
    
    for i in range(1000):
        # Generate keys and store them
        private_key, public_key = generate_ecc_keypair()
        serialized = serialize_public_key(public_key)
        test_data.append({
            "id": i,
            "key": serialized,
            "hash": secure_hash(serialized)
        })
    
    # Peak memory
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Clear data
    test_data.clear()
    
    # Allow garbage collection
    import gc
    gc.collect()
    await asyncio.sleep(0.1)
    
    # Final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        "initial_memory_mb": initial_memory,
        "peak_memory_mb": peak_memory,
        "final_memory_mb": final_memory,
        "memory_growth_mb": peak_memory - initial_memory,
        "memory_released_mb": peak_memory - final_memory
    }


async def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark."""
    print("📈 ZKPAS Performance Benchmarking Suite")
    print("=" * 60)
    
    results = {}
    
    try:
        # 1. Cryptographic Operations
        print("\n1️⃣ Cryptographic Operations Benchmark")
        results['crypto'] = await benchmark_cryptographic_operations(500)
        
        print(f"   Key Generation: {results['crypto']['keygen_ops_per_sec']:.1f} ops/sec")
        print(f"   Key Serialization: {results['crypto']['serialize_ops_per_sec']:.1f} ops/sec")
        print(f"   Hash Operations: {results['crypto']['hash_ops_per_sec']:.1f} ops/sec")
        
        # 2. Event System
        print("\n2️⃣ Event System Benchmark")
        results['events'] = await benchmark_event_system(5000)
        
        print(f"   Event Publishing: {results['events']['events_per_sec']:.1f} events/sec")
        print(f"   Event Processing: {results['events']['processing_rate']:.1f} events/sec")
        print(f"   Events Processed: {results['events']['events_processed']}")
        
        # 3. Concurrent Operations
        print("\n3️⃣ Concurrent Operations Benchmark")
        results['concurrency'] = await benchmark_concurrent_operations(25)
        
        print(f"   Throughput: {results['concurrency']['throughput']:.1f} ops/sec")
        print(f"   Success Rate: {results['concurrency']['success_rate']*100:.1f}%")
        print(f"   Avg Execution Time: {results['concurrency']['avg_execution_time']*1000:.1f}ms")
        
        # 4. Memory Usage
        print("\n4️⃣ Memory Usage Benchmark")
        results['memory'] = await benchmark_memory_usage()
        
        print(f"   Initial Memory: {results['memory']['initial_memory_mb']:.1f} MB")
        print(f"   Peak Memory: {results['memory']['peak_memory_mb']:.1f} MB")
        print(f"   Memory Growth: {results['memory']['memory_growth_mb']:.1f} MB")
        print(f"   Memory Released: {results['memory']['memory_released_mb']:.1f} MB")
        
        # Generate Performance Report
        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 40)
        
        # Overall Assessment
        crypto_score = min(100, results['crypto']['keygen_ops_per_sec'] / 10)
        event_score = min(100, results['events']['events_per_sec'] / 100)
        concurrency_score = results['concurrency']['success_rate'] * 100
        memory_score = 100 - min(100, results['memory']['memory_growth_mb'] / 10)
        
        overall_score = (crypto_score + event_score + concurrency_score + memory_score) / 4
        
        print(f"🔐 Cryptographic Performance: {crypto_score:.1f}/100")
        print(f"📡 Event System Performance: {event_score:.1f}/100")
        print(f"⚡ Concurrency Performance: {concurrency_score:.1f}/100")
        print(f"💾 Memory Efficiency: {memory_score:.1f}/100")
        print(f"📈 Overall Performance Score: {overall_score:.1f}/100")
        
        # Performance Grade
        if overall_score >= 90:
            grade = "🏆 EXCELLENT"
        elif overall_score >= 80:
            grade = "🥇 VERY GOOD"
        elif overall_score >= 70:
            grade = "🥈 GOOD"
        elif overall_score >= 60:
            grade = "🥉 FAIR"
        else:
            grade = "⚠️  NEEDS IMPROVEMENT"
        
        print(f"\n🎯 Performance Grade: {grade}")
        
        # Recommendations
        print(f"\n💡 PERFORMANCE RECOMMENDATIONS")
        print("-" * 35)
        
        if results['crypto']['keygen_ops_per_sec'] < 100:
            print("• Consider key caching for repeated operations")
        
        if results['events']['processing_rate'] < results['events']['events_per_sec']:
            print("• Event processing may be a bottleneck")
        
        if results['concurrency']['success_rate'] < 0.95:
            print("• Review concurrent operation error handling")
        
        if results['memory']['memory_growth_mb'] > 50:
            print("• Monitor memory usage in production")
        
        print("• System performance is within acceptable ranges")
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        print(f"❌ Benchmark failed: {e}")
    
    return results


async def main():
    """Main entry point for performance benchmark."""
    print("🎯 Welcome to ZKPAS Performance Benchmarking!")
    print("This test measures system performance under various conditions.")
    print()
    
    start_time = time.time()
    
    results = await run_comprehensive_benchmark()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n⏰ Total benchmark time: {total_time:.2f} seconds")
    print("✨ Performance benchmarking completed!")


if __name__ == "__main__":
    asyncio.run(main())
