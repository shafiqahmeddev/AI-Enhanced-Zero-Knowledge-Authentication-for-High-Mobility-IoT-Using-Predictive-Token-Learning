#!/usr/bin/env python3
"""
ZKPAS LSTM-Enhanced Mobility Prediction Demo

This demo shows the LSTM-based mobility prediction system with real mobility datasets.

Features demonstrated:
1. LSTM neural networks for sequential trajectory prediction
2. "Train once, use many times" with TensorFlow/Keras models
3. Enhanced accuracy for temporal mobility patterns
4. Real-world dataset integration with deep learning

Usage:
    python demo_lstm_system.py [--force-retrain] [--max-users N]
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
    """Demonstrate the LSTM-enhanced mobility prediction system."""
    parser = argparse.ArgumentParser(description="ZKPAS LSTM Mobility Prediction Demo")
    parser.add_argument("--force-retrain", action="store_true", 
                       help="Force retraining of LSTM models even if they exist")
    parser.add_argument("--max-users", type=int, default=5,
                       help="Maximum number of users to load (default: 5)")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs for LSTM (default: 20)")
    parser.add_argument("--sequence-length", type=int, default=15,
                       help="LSTM sequence length (default: 15)")
    args = parser.parse_args()
    
    print("🧠 ZKPAS LSTM-Enhanced Mobility Prediction Demo")
    print("=" * 55)
    
    # Create event bus for system communication
    event_bus = EventBus()
    
    # Step 1: Load Real Datasets
    print("\n📂 Step 1: Loading Real Mobility Datasets...")
    dataset_config = get_default_dataset_config()
    dataset_config.max_users = args.max_users
    
    try:
        loader, datasets = await load_real_mobility_data(dataset_config, event_bus)
        
        # Print dataset statistics
        stats = loader.get_dataset_statistics()
        total_trajectories = sum(stat["num_trajectories"] for stat in stats.values())
        total_points = sum(stat["total_points"] for stat in stats.values())
        
        print(f"✅ Loaded {len(datasets)} datasets with {total_trajectories} trajectories")
        print(f"   Total GPS points: {total_points:,}")
        
        # Show trajectory statistics for LSTM suitability
        suitable_trajectories = loader.get_trajectories_for_training(max_trajectories=1000)
        long_trajectories = [t for t in suitable_trajectories if len(t.points) >= 20]
        
        print(f"   Trajectories suitable for LSTM: {len(long_trajectories)}")
        if long_trajectories:
            avg_length = sum(len(t.points) for t in long_trajectories) / len(long_trajectories)
            print(f"   Average trajectory length: {avg_length:.1f} points")
        
    except Exception as e:
        print(f"❌ Failed to load datasets: {e}")
        return
    
    # Step 2: Configure LSTM Training
    print(f"\n🧠 Step 2: Configuring LSTM Model...")
    
    training_config = get_default_training_config()
    training_config.force_retrain = args.force_retrain
    training_config.epochs = args.epochs
    training_config.sequence_length = args.sequence_length
    training_config.lstm_units = 64
    training_config.lstm_layers = 2
    training_config.dropout_rate = 0.3
    training_config.learning_rate = 0.001
    training_config.batch_size = 32
    training_config.patience = 8
    
    print(f"   LSTM Configuration:")
    print(f"     - Sequence Length: {training_config.sequence_length} time steps")
    print(f"     - LSTM Units: {training_config.lstm_units}")
    print(f"     - LSTM Layers: {training_config.lstm_layers}")
    print(f"     - Dropout Rate: {training_config.dropout_rate}")
    print(f"     - Learning Rate: {training_config.learning_rate}")
    print(f"     - Max Epochs: {training_config.epochs}")
    print(f"     - Batch Size: {training_config.batch_size}")
    
    # Step 3: Train or Load LSTM Models
    print(f"\n🤖 Step 3: {'Training' if args.force_retrain else 'Loading/Training'} LSTM Models...")
    
    start_time = time.time()
    try:
        model_trainer = await train_or_load_models(loader, training_config, event_bus)
        training_time = time.time() - start_time
        
        available_models = model_trainer.list_available_models()
        print(f"✅ Models ready in {training_time:.1f}s: {available_models}")
        
        # Show LSTM model performance
        lstm_metadata = model_trainer.get_metadata("mobility_prediction_lstm")
        if lstm_metadata:
            print(f"\n   🧠 LSTM MOBILITY PREDICTION MODEL:")
            print(f"     - Model Type: {lstm_metadata.model_type}")
            print(f"     - Version: {lstm_metadata.version}")
            print(f"     - Trained: {lstm_metadata.training_date[:19]}")
            
            perf = lstm_metadata.performance_metrics
            print(f"     - Average Distance Error: {perf.get('avg_distance_error_km', 0):.3f} km")
            print(f"     - Training Loss: {perf.get('final_loss', 0):.6f}")
            print(f"     - Validation Loss: {perf.get('final_val_loss', 0):.6f}")
            print(f"     - Training Samples: {perf.get('training_samples', 0):,}")
            print(f"     - Epochs Trained: {perf.get('epochs_trained', 0)}")
            print(f"     - Sequence Length: {perf.get('sequence_length', 0)}")
            
            # Model architecture details
            params = lstm_metadata.model_params
            print(f"     - LSTM Units: {params.get('lstm_units', 'N/A')}")
            print(f"     - LSTM Layers: {params.get('lstm_layers', 'N/A')}")
            print(f"     - Dropout Rate: {params.get('dropout_rate', 'N/A')}")
        
        # Show other models
        for model_name in ["pattern_classification", "risk_assessment"]:
            metadata = model_trainer.get_metadata(model_name)
            if metadata:
                print(f"\n   📊 {model_name.upper()}:")
                print(f"     - Version: {metadata.version}")
                perf = metadata.performance_metrics
                if 'accuracy' in perf:
                    print(f"     - Accuracy: {perf['accuracy']:.3f}")
                if 'mae' in perf:
                    print(f"     - MAE: {perf['mae']:.3f}")
        
    except Exception as e:
        print(f"❌ Failed to train/load models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Initialize LSTM-Enhanced Mobility Predictor
    print(f"\n🎯 Step 4: Initializing LSTM-Enhanced Mobility Predictor...")
    
    predictor = MobilityPredictor(event_bus, model_trainer)
    
    if predictor.is_ready_for_prediction():
        print("✅ LSTM mobility predictor ready")
        performance = predictor.get_model_performance()
        
        # Check if using LSTM
        if "mobility_prediction_lstm" in performance:
            print("🧠 Using LSTM neural network for mobility prediction")
            lstm_perf = performance["mobility_prediction_lstm"]
            print(f"   Average prediction error: {lstm_perf.get('avg_distance_error_km', 0):.3f} km")
        else:
            print("⚠️ Fallback to traditional models (LSTM not available)")
    else:
        print("❌ Mobility predictor not ready")
        return
    
    # Step 5: Demonstrate LSTM Predictions
    print(f"\n🔮 Step 5: Demonstrating LSTM Mobility Predictions...")
    
    # Get a sample trajectory for demonstration
    sample_trajectories = loader.get_trajectories_for_training(max_trajectories=5)
    
    if sample_trajectories:
        # Use trajectory with enough points for LSTM
        demo_trajectory = None
        for traj in sample_trajectories:
            if len(traj.points) >= 30:  # Need enough for LSTM sequence
                demo_trajectory = traj
                break
        
        if not demo_trajectory:
            demo_trajectory = sample_trajectories[0]
        
        demo_points = demo_trajectory.points[:30]  # Use first 30 points
        
        print(f"   Using trajectory from user {demo_trajectory.user_id}")
        print(f"   Pattern: {demo_trajectory.mobility_pattern.name}")
        print(f"   Points available: {len(demo_points)}")
        print(f"   Duration: {demo_trajectory.duration_seconds/60:.1f} minutes")
        print(f"   Distance: {demo_trajectory.total_distance_km:.2f} km")
        print(f"   Avg Speed: {demo_trajectory.avg_speed_kmh:.1f} km/h")
        
        # Simulate real-time LSTM predictions
        device_id = f"lstm_demo_device_{demo_trajectory.user_id}"
        
        # Build up sequence history first
        for i, point in enumerate(demo_points[:20]):  # Build sequence
            await predictor.update_location(device_id, point)
        
        print(f"\n   🧠 LSTM Predictions (using {training_config.sequence_length}-step sequences):")
        
        prediction_errors = []
        for i, point in enumerate(demo_points[20:25]):  # Make 5 predictions
            await predictor.update_location(device_id, point)
            
            predictions = await predictor.predict_mobility(device_id)
            
            if predictions:
                pred = predictions[0]  # Shortest horizon prediction
                actual_next = demo_points[20 + i + 1] if 20 + i + 1 < len(demo_points) else None
                
                print(f"\n     Prediction {i+1}:")
                print(f"       Current: ({point.latitude:.5f}, {point.longitude:.5f})")
                print(f"       Predicted: ({pred.predicted_location.latitude:.5f}, "
                      f"{pred.predicted_location.longitude:.5f})")
                print(f"       Confidence: {pred.confidence:.3f}")
                print(f"       Pattern: {pred.mobility_pattern.name}")
                print(f"       Horizon: {pred.time_horizon}s")
                
                if actual_next:
                    # Calculate prediction error
                    error = predictor._calculate_distance(
                        pred.predicted_location.latitude,
                        pred.predicted_location.longitude,
                        actual_next.latitude,
                        actual_next.longitude
                    )
                    prediction_errors.append(error)
                    print(f"       Actual error: {error:.1f}m")
        
        # Show prediction statistics
        if prediction_errors:
            avg_error = sum(prediction_errors) / len(prediction_errors)
            print(f"\n   📊 LSTM Prediction Performance:")
            print(f"     - Average error: {avg_error:.1f}m")
            print(f"     - Min error: {min(prediction_errors):.1f}m")
            print(f"     - Max error: {max(prediction_errors):.1f}m")
            print(f"     - Predictions made: {len(prediction_errors)}")
        
        # Show device statistics
        stats = predictor.get_device_stats(device_id)
        print(f"\n   📈 Device Statistics:")
        print(f"     - Total locations processed: {stats.get('total_locations', 0)}")
        print(f"     - Average speed: {stats.get('avg_speed', 0):.1f} m/s")
        print(f"     - Total distance: {stats.get('total_distance', 0):.1f}m")
    
    # Step 6: LSTM vs Traditional Comparison
    print(f"\n📊 Step 6: LSTM Model Advantages...")
    
    print("✅ LSTM Neural Network Benefits:")
    print("   🧠 Sequential Learning: Captures temporal dependencies in movement")
    print("   🔄 Memory: Remembers long-term mobility patterns")
    print("   📈 Accuracy: Better prediction for complex trajectories")
    print("   🎯 Adaptation: Learns user-specific movement behaviors")
    print("   ⚡ Efficiency: Fast inference once trained")
    
    # Show model comparison
    lstm_metadata = model_trainer.get_metadata("mobility_prediction_lstm")
    traditional_metadata = model_trainer.get_metadata("mobility_prediction")
    
    if lstm_metadata and traditional_metadata:
        lstm_error = lstm_metadata.performance_metrics.get('avg_distance_error_km', 0)
        traditional_error = traditional_metadata.performance_metrics.get('avg_distance_error_km', 0)
        
        if lstm_error > 0 and traditional_error > 0:
            improvement = ((traditional_error - lstm_error) / traditional_error) * 100
            print(f"\n   📊 Performance Comparison:")
            print(f"     - LSTM Error: {lstm_error:.3f} km")
            print(f"     - Random Forest Error: {traditional_error:.3f} km")
            print(f"     - LSTM Improvement: {improvement:.1f}%")
    
    print(f"\n🎉 LSTM Demo Complete!")
    print("\n💡 Key LSTM Implementation Features:")
    print("   1. TensorFlow/Keras deep learning models")
    print("   2. Sequential trajectory pattern learning")
    print("   3. Temporal feature extraction with velocity and acceleration")
    print("   4. MinMax scaling optimized for neural networks")
    print("   5. Early stopping and model checkpointing")
    print("   6. Persistent H5 model storage for fast loading")
    print("   7. Real-time sequence preparation for predictions")


if __name__ == "__main__":
    asyncio.run(main())