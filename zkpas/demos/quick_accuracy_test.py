#!/usr/bin/env python3
"""
Quick Accuracy Test for Enhanced LSTM System

This demo provides a fast validation that our enhanced LSTM system
can achieve the target 92%+ accuracy.
"""

import asyncio
import time
import numpy as np
from pathlib import Path

# Import our enhanced components
from app.pytorch_lstm_predictor import AdvancedConfig, EnsembleMobilityPredictor
from app.dataset_loader import DatasetLoader, DatasetConfig, TrajectoryData
from app.mobility_predictor import LocationPoint, MobilityPattern
from app.events import EventBus
import random

print("🚀 ZKPAS Quick Accuracy Test")
print("=" * 50)
print("🎯 TARGET: Validate 92%+ accuracy capability")
print()

def create_test_data():
    """Create high-quality synthetic test data."""
    print("📊 Creating synthetic test data...")
    
    trajectories = []
    
    # Create realistic mobility patterns
    for user_id in range(50):  # 50 users for quick test
        for traj_id in range(10):  # 10 trajectories per user
            points = []
            
            # Starting location (NYC area)
            base_lat, base_lon = 40.7128 + random.uniform(-0.05, 0.05), -74.0060 + random.uniform(-0.05, 0.05)
            current_lat, current_lon = base_lat, base_lon
            start_time = time.time() - random.randint(0, 24*3600)
            
            # Generate 50-80 points per trajectory
            num_points = random.randint(50, 80)
            
            for i in range(num_points):
                timestamp = start_time + i * random.uniform(30, 60)  # 30-60s intervals
                
                # Realistic movement pattern
                if i < num_points // 3:  # First third: moving away
                    lat_delta = random.uniform(-0.001, 0.001)
                    lon_delta = random.uniform(-0.001, 0.001)
                elif i < 2 * num_points // 3:  # Middle: stable area
                    lat_delta = random.uniform(-0.0005, 0.0005)
                    lon_delta = random.uniform(-0.0005, 0.0005)
                else:  # Last third: returning
                    lat_delta = (base_lat - current_lat) * 0.1 + random.uniform(-0.0003, 0.0003)
                    lon_delta = (base_lon - current_lon) * 0.1 + random.uniform(-0.0003, 0.0003)
                
                current_lat += lat_delta
                current_lon += lon_delta
                
                # Add realistic GPS noise
                current_lat += random.uniform(-0.00001, 0.00001)
                current_lon += random.uniform(-0.00001, 0.00001)
                
                point = LocationPoint(
                    latitude=current_lat,
                    longitude=current_lon,
                    timestamp=timestamp
                )
                points.append(point)
            
            # Calculate trajectory stats
            total_distance = sum([
                np.sqrt((p2.latitude - p1.latitude)**2 + (p2.longitude - p1.longitude)**2) * 111
                for p1, p2 in zip(points[:-1], points[1:])
            ])
            
            duration = points[-1].timestamp - points[0].timestamp
            avg_speed = (total_distance / (duration / 3600)) if duration > 0 else 0
            
            trajectory = TrajectoryData(
                user_id=f"test_user_{user_id}",
                trajectory_id=f"traj_{traj_id}",
                points=points,
                start_time=points[0].timestamp,
                end_time=points[-1].timestamp,
                duration_seconds=duration,
                total_distance_km=total_distance,
                avg_speed_kmh=avg_speed,
                mobility_pattern=MobilityPattern.COMMUTER
            )
            
            trajectories.append(trajectory)
    
    print(f"✅ Created {len(trajectories)} test trajectories")
    return trajectories

def test_accuracy():
    """Test the enhanced LSTM system accuracy."""
    print("\n🧠 Testing Enhanced LSTM System")
    print("-" * 40)
    
    # Create test data
    trajectories = create_test_data()
    
    # Configuration optimized for accuracy
    config = AdvancedConfig(
        sequence_length=25,
        lstm_units=128,
        num_layers=3,
        attention_heads=4,
        dropout_rate=0.1,
        learning_rate=0.001,
        epochs=50,  # Reduced for quick test
        batch_size=128,
        patience=10,
        num_ensemble_models=3,  # Reduced for speed
        data_augmentation_factor=1.5,
    )
    
    # Initialize predictor
    predictor = EnsembleMobilityPredictor(config)
    
    print(f"🎯 Training ensemble with {config.num_ensemble_models} models...")
    start_time = time.time()
    
    # Train the model
    results = predictor.train(trajectories)
    
    training_time = time.time() - start_time
    
    print(f"\n📊 RESULTS:")
    print(f"   ⏱️ Training Time: {training_time:.1f}s")
    print(f"   🎯 Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"   📏 Error: {results['avg_error_km']*1000:.1f}m")
    print(f"   🤖 Models: {results['num_models']}")
    print(f"   📊 Features: {results['num_features']}")
    print(f"   📈 Training Samples: {results['training_samples']}")
    
    # Accuracy assessment
    if results['accuracy'] >= 0.92:
        status = "🟢 SUCCESS"
        print(f"\n{status}: Target accuracy achieved!")
        print(f"✅ Achieved {results['accuracy']*100:.1f}% accuracy (target: 92%)")
        print(f"🎉 Exceeded target by {(results['accuracy'] - 0.92)*100:.1f} percentage points")
        print("🚀 System ready for production!")
        
    elif results['accuracy'] >= 0.85:
        status = "🟡 CLOSE"
        gap = (0.92 - results['accuracy']) * 100
        print(f"\n{status}: Very close to target")
        print(f"📊 Gap: {gap:.1f} percentage points to reach 92%")
        print("🔧 Minor improvements needed")
        
    else:
        status = "🔴 NEEDS WORK"
        gap = (0.92 - results['accuracy']) * 100
        print(f"\n{status}: More improvement needed")
        print(f"📊 Gap: {gap:.1f} percentage points to reach 92%")
    
    return results['accuracy'] >= 0.92, results

if __name__ == "__main__":
    try:
        success, results = test_accuracy()
        
        print(f"\n{'='*50}")
        print("🏆 FINAL ASSESSMENT")
        print(f"{'='*50}")
        
        if success:
            print("✅ ZKPAS Enhanced LSTM system successfully achieves 92%+ accuracy!")
            print("🚀 Ready for integration and deployment")
        else:
            print("🔧 System shows strong potential but needs optimization")
            print("📈 Consider longer training or larger ensemble")
        
        print(f"\n⏱️ Total test time: {time.time():.1f}s")
        print("✅ Quick accuracy test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()