#!/usr/bin/env python3
"""
ZKPAS Real Data Integration Demo

This demo shows the complete "train once, use many times" system with real mobility datasets.

Features demonstrated:
1. Load real Geolife and Beijing Taxi datasets
2. Train ML models once and save to disk
3. Load pre-trained models on subsequent runs
4. Make mobility predictions using real trajectory patterns
5. Show performance improvements over synthetic data

Usage:
    python demo_real_data_system.py [--force-retrain]
"""

import asyncio
import argparse
import time
import sys
from pathlib import Path

# Add the zkpas directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.dataset_loader import load_real_mobility_data, get_default_dataset_config
from app.model_trainer import train_or_load_models, get_default_training_config
from app.mobility_predictor import MobilityPredictor, LocationPoint
from app.events import EventBus


async def main():
    """Demonstrate the complete real data integration system."""
    parser = argparse.ArgumentParser(description="ZKPAS Real Data Integration Demo")
    parser.add_argument("--force-retrain", action="store_true", 
                       help="Force retraining of models even if they exist")
    parser.add_argument("--max-users", type=int, default=10,
                       help="Maximum number of users to load (default: 10)")
    args = parser.parse_args()
    
    print("🚀 ZKPAS Real Data Integration Demo")
    print("=" * 50)
    
    # Create event bus for system communication
    event_bus = EventBus()
    
    # Step 1: Load Real Datasets
    print("\n📂 Step 1: Loading Real Mobility Datasets...")
    dataset_config = get_default_dataset_config()
    dataset_config.max_users = args.max_users
    dataset_config.force_retrain = args.force_retrain
    
    try:
        loader, datasets = await load_real_mobility_data(dataset_config, event_bus)
        
        # Print dataset statistics
        stats = loader.get_dataset_statistics()
        total_trajectories = sum(stat["num_trajectories"] for stat in stats.values())
        total_points = sum(stat["total_points"] for stat in stats.values())
        
        print(f"✅ Loaded {len(datasets)} datasets with {total_trajectories} trajectories")
        print(f"   Total GPS points: {total_points:,}")
        
        for dataset_name, stat in stats.items():
            print(f"\n   {dataset_name.upper()}:")
            print(f"     - Trajectories: {stat['num_trajectories']}")
            print(f"     - Users: {stat['num_users']}")
            print(f"     - Distance: {stat['total_distance_km']:.1f} km")
            print(f"     - Avg Speed: {stat['avg_speed_kmh']:.1f} km/h")
            print(f"     - Patterns: {stat['mobility_patterns']}")
        
    except Exception as e:
        print(f"❌ Failed to load datasets: {e}")
        print("\n💡 Make sure the datasets are available in ../Datasets/")
        print("   - Geolife Trajectories 1.3")
        print("   - Beijing Taxi Logs 2008")
        return
    
    # Step 2: Train or Load ML Models
    print(f"\n🤖 Step 2: {'Training' if args.force_retrain else 'Loading/Training'} ML Models...")
    
    training_config = get_default_training_config()
    training_config.force_retrain = args.force_retrain
    training_config.max_depth = 15  # Optimize for real data
    training_config.n_estimators = 50  # Balance performance and speed
    
    start_time = time.time()
    try:
        model_trainer = await train_or_load_models(loader, training_config, event_bus)
        training_time = time.time() - start_time
        
        available_models = model_trainer.list_available_models()
        print(f"✅ Models ready in {training_time:.1f}s: {available_models}")
        
        # Show model performance
        for model_name in available_models:
            metadata = model_trainer.get_metadata(model_name)
            if metadata:
                print(f"\n   {model_name.upper()}:")
                print(f"     - Version: {metadata.version}")
                print(f"     - Trained: {metadata.training_date[:19]}")
                perf = metadata.performance_metrics
                if 'accuracy' in perf:
                    print(f"     - Accuracy: {perf['accuracy']:.3f}")
                if 'avg_distance_error_km' in perf:
                    print(f"     - Avg Error: {perf['avg_distance_error_km']:.2f} km")
        
    except Exception as e:
        print(f"❌ Failed to train/load models: {e}")
        return
    
    # Step 3: Initialize Enhanced Mobility Predictor
    print(f"\n🎯 Step 3: Initializing Enhanced Mobility Predictor...")
    
    predictor = MobilityPredictor(event_bus, model_trainer)
    
    if predictor.is_ready_for_prediction():
        print("✅ Mobility predictor ready with pre-trained models")
        performance = predictor.get_model_performance()
        print(f"   Model performance: {performance}")
    else:
        print("❌ Mobility predictor not ready - using fallback heuristics")
    
    # Step 4: Demonstrate Predictions with Real Data
    print(f"\n🔮 Step 4: Demonstrating Mobility Predictions...")
    
    # Get a sample trajectory for demonstration
    sample_trajectories = loader.get_trajectories_for_training(max_trajectories=3)
    
    if sample_trajectories:
        # Use first trajectory for demo
        demo_trajectory = sample_trajectories[0]
        demo_points = demo_trajectory.points[:20]  # Use first 20 points
        
        print(f"   Using trajectory from user {demo_trajectory.user_id}")
        print(f"   Pattern: {demo_trajectory.mobility_pattern.name}")
        print(f"   Duration: {demo_trajectory.duration_seconds/60:.1f} minutes")
        print(f"   Distance: {demo_trajectory.total_distance_km:.2f} km")
        print(f"   Avg Speed: {demo_trajectory.avg_speed_kmh:.1f} km/h")
        
        # Simulate real-time predictions
        device_id = f"demo_device_{demo_trajectory.user_id}"
        
        for i, point in enumerate(demo_points[5:15]):  # Use middle points
            await predictor.update_location(device_id, point)
            
            if i >= 5:  # Start predicting after some history
                predictions = await predictor.predict_mobility(device_id)
                
                if predictions:
                    pred = predictions[0]  # Shortest horizon prediction
                    actual_next = demo_points[i + 6] if i + 6 < len(demo_points) else None
                    
                    print(f"\n   Prediction {i-4}:")
                    print(f"     Current: ({point.latitude:.5f}, {point.longitude:.5f})")
                    print(f"     Predicted: ({pred.predicted_location.latitude:.5f}, "
                          f"{pred.predicted_location.longitude:.5f})")
                    print(f"     Confidence: {pred.confidence:.3f}")
                    print(f"     Pattern: {pred.mobility_pattern.name}")
                    print(f"     Horizon: {pred.time_horizon}s")
                    
                    if actual_next:
                        # Calculate prediction error
                        error = predictor._calculate_distance(
                            pred.predicted_location.latitude,
                            pred.predicted_location.longitude,
                            actual_next.latitude,
                            actual_next.longitude
                        )
                        print(f"     Actual error: {error:.1f}m")
        
        # Show device statistics
        stats = predictor.get_device_stats(device_id)
        print(f"\n   Device Statistics:")
        print(f"     - Total locations: {stats.get('total_locations', 0)}")
        print(f"     - Avg speed: {stats.get('avg_speed', 0):.1f} m/s")
        print(f"     - Max speed: {stats.get('max_speed', 0):.1f} m/s")
        print(f"     - Total distance: {stats.get('total_distance', 0):.1f}m")
    
    # Step 5: Performance Comparison
    print(f"\n📊 Step 5: System Performance Summary...")
    
    if predictor.models_loaded:
        print("✅ Pre-trained Model System:")
        print("   - Instant startup (models loaded from disk)")
        print("   - Trained on real Geolife + Taxi trajectory data")
        print("   - Consistent predictions across application restarts")
        print("   - No retraining required during operation")
        
        # Model file sizes
        model_dir = Path("data/trained_models")
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in model_files if f.exists())
            print(f"   - Model storage: {total_size / 1024 / 1024:.1f} MB")
    else:
        print("⚠️  Fallback Heuristic System:")
        print("   - Simple linear extrapolation")
        print("   - No learning from historical patterns")
        print("   - Lower prediction accuracy")
    
    print(f"\n🎉 Demo Complete!")
    print("\n💡 Key Benefits of Real Data Integration:")
    print("   1. Authentic IoT device movement patterns")
    print("   2. Better prediction accuracy for real-world scenarios")
    print("   3. Faster application startup with pre-trained models")
    print("   4. Scalable training process on large datasets")
    print("   5. Persistent model improvements across sessions")


if __name__ == "__main__":
    asyncio.run(main())