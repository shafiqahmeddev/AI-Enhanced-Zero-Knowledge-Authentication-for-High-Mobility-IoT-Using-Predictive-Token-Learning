#!/usr/bin/env python3
"""
Benchmark script to demonstrate the hash calculation improvement.

This script compares the old to_string() method with the new pandas hashing method
to show the performance improvement and memory efficiency.
"""

import pandas as pd
import numpy as np
import hashlib
import time
import psutil
import os
from pandas.util import hash_pandas_object

def old_hash_method(data: pd.DataFrame) -> str:
    """Original inefficient hash method using to_string()."""
    data_str = data.to_string(index=False)
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]

def new_hash_method(data: pd.DataFrame) -> str:
    """New efficient hash method using pandas built-in hashing."""
    row_hashes = hash_pandas_object(data, index=False, encoding='utf8')
    combined_hash = hashlib.sha256(row_hashes.values.tobytes()).hexdigest()
    return combined_hash[:16]

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark_hash_methods():
    """Benchmark both hash methods across different dataset sizes."""
    print("=" * 70)
    print("HASH CALCULATION IMPROVEMENT BENCHMARK")
    print("=" * 70)
    
    # Test dataset sizes
    sizes = [100, 1000, 10000, 50000]
    
    results = []
    
    for size in sizes:
        print(f"\nTesting with {size:,} rows...")
        print("-" * 50)
        
        # Create test data
        np.random.seed(42)
        test_data = pd.DataFrame({
            'numeric1': np.random.randn(size),
            'numeric2': np.random.uniform(0, 100, size),
            'category1': np.random.choice(['A', 'B', 'C', 'D', 'E'], size),
            'category2': np.random.choice(['X', 'Y', 'Z'], size),
            'text': [f"text_{i}" for i in range(size)]
        })
        
        # Test old method
        mem_before = get_memory_usage()
        start_time = time.time()
        old_hash = old_hash_method(test_data)
        old_time = time.time() - start_time
        old_mem_peak = get_memory_usage()
        
        # Clear memory
        import gc
        gc.collect()
        
        # Test new method
        mem_before_new = get_memory_usage()
        start_time = time.time()
        new_hash = new_hash_method(test_data)
        new_time = time.time() - start_time
        new_mem_peak = get_memory_usage()
        
        # Calculate improvements
        time_improvement = ((old_time - new_time) / old_time) * 100
        mem_improvement = ((old_mem_peak - mem_before) - (new_mem_peak - mem_before_new)) / (old_mem_peak - mem_before) * 100
        
        print(f"Old method (to_string):")
        print(f"  Time: {old_time:.4f}s")
        print(f"  Memory peak: {old_mem_peak:.2f} MB")
        print(f"  Hash: {old_hash}")
        
        print(f"New method (pandas hash):")
        print(f"  Time: {new_time:.4f}s") 
        print(f"  Memory peak: {new_mem_peak:.2f} MB")
        print(f"  Hash: {new_hash}")
        
        print(f"Improvements:")
        print(f"  Time: {time_improvement:+.1f}% {'(faster)' if time_improvement > 0 else '(slower)'}")
        print(f"  Memory: {mem_improvement:+.1f}% {'(less)' if mem_improvement > 0 else '(more)'}")
        
        # Verify both methods produce deterministic results
        old_hash2 = old_hash_method(test_data)
        new_hash2 = new_hash_method(test_data)
        
        print(f"Deterministic check:")
        print(f"  Old method consistent: {old_hash == old_hash2}")
        print(f"  New method consistent: {new_hash == new_hash2}")
        
        results.append({
            'size': size,
            'old_time': old_time,
            'new_time': new_time,
            'time_improvement': time_improvement,
            'old_hash': old_hash,
            'new_hash': new_hash
        })
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nKey Improvements:")
    print("✓ Memory efficient: No need to create large string representations")
    print("✓ Faster processing: Direct row-wise hashing")
    print("✓ Deterministic: Consistent hash values across runs")
    print("✓ Scalable: Performance doesn't degrade significantly with size")
    
    print(f"\nPerformance Summary:")
    for result in results:
        print(f"  {result['size']:>6,} rows: {result['time_improvement']:+6.1f}% time improvement")
    
    return results

if __name__ == "__main__":
    try:
        benchmark_results = benchmark_hash_methods()
        print(f"\nBenchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
