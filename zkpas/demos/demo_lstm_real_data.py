#!/usr/bin/env python3
"""
Enhanced LSTM Mobility Prediction Demo with Real Datasets

This demo shows the improved LSTM system using real-world mobility datasets
(Geolife and Beijing Taxi) for training and prediction.

Features:
- Real TensorFlow LSTM implementation
- Training on real mobility datasets
- Advanced feature engineering
- Improved accuracy metrics
- Data preprocessing for GPS noise handling

Author: Shafiq Ahmed <s.ahmed@essex.ac.uk>
"""

import asyncio
import time
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ZKPAS components
from app.dataset_loader import DatasetLoader, DatasetConfig, get_default_dataset_config, load_real_mobility_data
from app.model_trainer import ModelTrainer, TrainingConfig, get_default_training_config, train_or_load_models
from app.mobility_predictor import MobilityPredictor, LocationPoint
from app.events import EventBus, EventType
from shared.config import get_config


class EnhancedLSTMDemo:
    """Enhanced LSTM demo using real datasets."""
    
    def __init__(self):
        """Initialize the enhanced LSTM demo."""
        self.config = get_config()
        self.event_bus = EventBus()
        self.dataset_loader = None
        self.model_trainer = None
        self.mobility_predictor = None
        
    async def run_demo(self):
        """Run the complete enhanced LSTM demo."""
        print("🚀 Enhanced LSTM Mobility Prediction with Real Datasets")
        print("=" * 60)
        print("📋 Features:")
        print("   • Real TensorFlow LSTM implementation")
        print("   • Training on Geolife & Beijing Taxi datasets") 
        print("   • Advanced feature engineering")
        print("   • GPS noise handling")
        print("   • Improved accuracy metrics")
        print()
        
        try:
            # Step 1: Load real datasets
            await self._load_real_datasets()
            
            # Step 2: Train/load LSTM models
            await self._train_lstm_models()
            
            # Step 3: Initialize enhanced mobility predictor
            await self._initialize_predictor()
            
            # Step 4: Run prediction demonstrations
            await self._demonstrate_predictions()
            
            # Step 5: Performance analysis
            await self._analyze_performance()
            
            print("\n✅ Enhanced LSTM demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            print(f"\n❌ Demo failed: {e}")
    
    async def _load_real_datasets(self):
        """Load real-world mobility datasets."""
        print("📊 Loading Real Mobility Datasets...")
        
        # Configure dataset loading for improved performance
        dataset_config = DatasetConfig(
            max_users=100,  # Load more users for better training
            max_trajectories_per_user=20,  # More trajectories per user
            min_trajectory_points=30,  # Longer trajectories for better sequences
            temporal_resolution_seconds=30,  # 30-second intervals for better resolution
            apply_noise=True,  # Apply privacy noise
            noise_level=0.0001,  # Smaller noise for better accuracy
            cache_processed_data=True
        )
        
        try:
            # Load datasets
            self.dataset_loader, datasets = await load_real_mobility_data(dataset_config, self.event_bus)
            
            # Print dataset statistics
            stats = self.dataset_loader.get_dataset_statistics()
            print("\n📈 Dataset Statistics:")
            
            total_trajectories = 0
            total_points = 0
            
            for dataset_name, stat in stats.items():
                print(f"\n   {dataset_name.upper()} Dataset:")
                print(f"   • Trajectories: {stat['num_trajectories']:,}")
                print(f"   • Users: {stat['num_users']:,}")
                print(f"   • Data Points: {stat['total_points']:,}")
                print(f"   • Total Distance: {stat['total_distance_km']:,.1f} km")
                print(f"   • Average Speed: {stat['avg_speed_kmh']:.1f} km/h")
                print(f"   • Mobility Patterns: {stat['mobility_patterns']}")
                
                total_trajectories += stat['num_trajectories']
                total_points += stat['total_points']
            
            print(f"\n📊 Total Data: {total_trajectories:,} trajectories, {total_points:,} GPS points")
            
            if total_trajectories == 0:
                print("⚠️  No real datasets found. Creating synthetic data...")
                await self._create_synthetic_data()
            else:
                print("✅ Real datasets loaded successfully!")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load real datasets: {e}")
            print(f"⚠️ Failed to load real datasets: {e}")
            print("🔧 Creating synthetic data for demonstration...")
            await self._create_synthetic_data()
    
    async def _create_synthetic_data(self):
        """Create high-quality synthetic data if real datasets unavailable."""
        print("🔧 Creating synthetic mobility data...")
        
        # For demonstration purposes, create realistic synthetic trajectories
        # This is a fallback when real datasets are not available
        from app.dataset_loader import TrajectoryData
        import random
        from datetime import datetime, timedelta
        
        synthetic_trajectories = []
        
        # Create multiple users with different mobility patterns
        for user_id in range(20):
            for traj_id in range(5):
                # Generate realistic trajectory
                points = []
                start_time = time.time() - random.randint(0, 86400)  # Random time in last day
                
                # Random starting location (simulate city area)
                base_lat = 39.9042 + random.uniform(-0.1, 0.1)  # Around Beijing
                base_lon = 116.4074 + random.uniform(-0.1, 0.1)
                
                current_lat, current_lon = base_lat, base_lon
                
                # Generate trajectory points with realistic movement
                for i in range(50):  # 50 points per trajectory
                    timestamp = start_time + i * 60  # 1-minute intervals
                    
                    # Add realistic movement with some pattern
                    if i < 20:  # First part: gradual movement
                        lat_delta = random.uniform(-0.001, 0.001)
                        lon_delta = random.uniform(-0.001, 0.001)
                    else:  # Second part: return movement (simulating round trip)
                        lat_delta = (base_lat - current_lat) * 0.1 + random.uniform(-0.0005, 0.0005)
                        lon_delta = (base_lon - current_lon) * 0.1 + random.uniform(-0.0005, 0.0005)
                    
                    current_lat += lat_delta
                    current_lon += lon_delta
                    
                    point = LocationPoint(
                        latitude=current_lat,
                        longitude=current_lon,
                        timestamp=timestamp
                    )
                    points.append(point)
                
                # Calculate trajectory statistics
                total_distance = sum([
                    np.sqrt((p2.latitude - p1.latitude)**2 + (p2.longitude - p1.longitude)**2) * 111
                    for p1, p2 in zip(points[:-1], points[1:])
                ])
                
                duration = points[-1].timestamp - points[0].timestamp
                avg_speed = (total_distance / (duration / 3600)) if duration > 0 else 0
                
                trajectory = TrajectoryData(
                    user_id=f"synthetic_user_{user_id}",
                    trajectory_id=f"traj_{traj_id}",
                    points=points,
                    start_time=points[0].timestamp,
                    end_time=points[-1].timestamp,
                    duration_seconds=duration,
                    total_distance_km=total_distance,
                    avg_speed_kmh=avg_speed
                )
                
                synthetic_trajectories.append(trajectory)
        
        # Store synthetic data in dataset loader
        if not hasattr(self.dataset_loader, 'trajectories'):
            self.dataset_loader.trajectories = {}
        self.dataset_loader.trajectories['synthetic'] = synthetic_trajectories
        
        print(f"✅ Created {len(synthetic_trajectories)} synthetic trajectories")
    
    async def _train_lstm_models(self):
        """Train enhanced LSTM models on real data."""
        print("\n🧠 Training Enhanced LSTM Models...")
        
        # Enhanced training configuration for better accuracy
        training_config = TrainingConfig(
            # LSTM architecture improvements
            sequence_length=25,     # Longer sequences for better pattern learning
            lstm_units=128,         # More units for better capacity
            lstm_layers=3,          # Deeper network
            dropout_rate=0.15,      # Less dropout for better learning
            
            # Training improvements
            learning_rate=0.0005,   # Lower learning rate for better convergence
            epochs=100,             # More epochs for better training
            batch_size=64,          # Larger batches for stable training
            patience=15,            # More patience for convergence
            
            # Data improvements
            test_size=0.15,         # Smaller test set, more training data
            time_window_hours=12,   # Longer time windows
            prediction_horizon_minutes=30,  # Shorter prediction horizon for better accuracy
            
            # Force retraining to use new configuration
            force_retrain=False,    # Use cached models if available
            models_dir="data/enhanced_models"
        )
        
        try:
            # Initialize model trainer
            self.model_trainer = await train_or_load_models(
                self.dataset_loader, 
                training_config, 
                self.event_bus
            )
            
            # Display model information
            models = self.model_trainer.list_available_models()
            print(f"\n📊 Available Models: {models}")
            
            for model_name in models:
                metadata = self.model_trainer.get_metadata(model_name)
                if metadata and 'lstm' in model_name.lower():
                    print(f"\n   🎯 {model_name.upper()}:")
                    print(f"   • Model Type: {'TensorFlow LSTM' if metadata.model_params.get('tensorflow_available') else 'Neural Network'}")
                    print(f"   • Sequence Length: {metadata.model_params.get('sequence_length', 'N/A')}")
                    print(f"   • LSTM Units: {metadata.model_params.get('lstm_units', 'N/A')}")
                    print(f"   • Layers: {metadata.model_params.get('lstm_layers', 'N/A')}")
                    
                    perf = metadata.performance_metrics
                    if perf:
                        print(f"   • Training Samples: {perf.get('training_samples', 'N/A'):,}")
                        print(f"   • Average Error: {perf.get('avg_distance_error_km', 'N/A'):.3f} km")
                        print(f"   • Latitude MAE: {perf.get('latitude_mae', 'N/A'):.6f}°")
                        print(f"   • Longitude MAE: {perf.get('longitude_mae', 'N/A'):.6f}°")
            
            print("✅ LSTM models ready!")
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            raise
    
    async def _initialize_predictor(self):
        """Initialize enhanced mobility predictor."""
        print("\n🎯 Initializing Enhanced Mobility Predictor...")
        
        # Initialize predictor with trained models
        self.mobility_predictor = MobilityPredictor(self.event_bus, self.model_trainer)
        
        if self.mobility_predictor.is_ready_for_prediction():
            print("✅ Enhanced mobility predictor initialized with trained models")
        else:
            print("⚠️ Predictor initialized with fallback heuristics (models not available)")
    
    async def _demonstrate_predictions(self):
        """Demonstrate enhanced prediction capabilities."""
        print("\n🔮 Demonstrating Enhanced Predictions...")
        
        # Get sample trajectories for demonstration
        trajectories = self.dataset_loader.get_trajectories_for_training(max_trajectories=5)
        
        if not trajectories:
            print("⚠️ No trajectories available for prediction demonstration")
            return
        
        total_predictions = 0
        accurate_predictions = 0
        total_error_km = 0
        
        for i, trajectory in enumerate(trajectories[:3]):  # Test on 3 trajectories
            print(f"\n   📍 Trajectory {i+1}: User {trajectory.user_id}")
            
            points = trajectory.points
            if len(points) < 30:  # Need enough points for sequence
                continue
            
            # Use first 80% of trajectory for history, predict last 20%
            split_point = int(len(points) * 0.8)
            history_points = points[:split_point]
            test_points = points[split_point:]
            
            # Load trajectory history into predictor
            device_id = f"demo_device_{i}"
            for point in history_points:
                await self.mobility_predictor.update_location(device_id, point)
            
            # Make predictions and compare with actual locations
            for j, actual_point in enumerate(test_points[:5]):  # Test 5 predictions
                try:
                    # Predict next location
                    predictions = await self.mobility_predictor.predict_mobility(device_id)
                    
                    if predictions:
                        # Use shortest horizon prediction
                        prediction = min(predictions, key=lambda p: p.time_horizon)
                        predicted_loc = prediction.predicted_location
                        
                        # Calculate error
                        error_km = self._calculate_distance_km(
                            actual_point.latitude, actual_point.longitude,
                            predicted_loc.latitude, predicted_loc.longitude
                        )
                        
                        # Check if prediction is accurate (within 200m)
                        is_accurate = error_km < 0.2
                        
                        total_predictions += 1
                        if is_accurate:
                            accurate_predictions += 1
                        total_error_km += error_km
                        
                        print(f"      Step {j+1}: Error = {error_km*1000:.1f}m, " + 
                              f"Accurate = {'✅' if is_accurate else '❌'}, " +
                              f"Confidence = {prediction.confidence:.3f}")
                    
                    # Update predictor with actual location for next prediction
                    await self.mobility_predictor.update_location(device_id, actual_point)
                    
                except Exception as e:
                    logger.warning(f"Prediction failed: {e}")
        
        # Calculate and display overall accuracy
        if total_predictions > 0:
            accuracy = (accurate_predictions / total_predictions) * 100
            avg_error = (total_error_km / total_predictions) * 1000  # Convert to meters
            
            print(f"\n📊 Prediction Results:")
            print(f"   • Total Predictions: {total_predictions}")
            print(f"   • Accurate Predictions (±200m): {accurate_predictions}")
            print(f"   • Accuracy: {accuracy:.1f}%")
            print(f"   • Average Error: {avg_error:.1f}m")
        else:
            print("⚠️ No predictions could be made")
    
    async def _analyze_performance(self):
        """Analyze and display performance metrics."""
        print("\n📊 Performance Analysis...")
        
        if self.mobility_predictor and self.mobility_predictor.model_performance:
            print("\n   🎯 Model Performance Metrics:")
            
            for model_name, metrics in self.mobility_predictor.model_performance.items():
                if 'lstm' in model_name.lower() or 'mobility' in model_name.lower():
                    print(f"\n   📈 {model_name.upper()}:")
                    
                    if 'avg_distance_error_km' in metrics:
                        print(f"   • Average Distance Error: {metrics['avg_distance_error_km']:.3f} km")
                    
                    if 'latitude_mae' in metrics:
                        print(f"   • Latitude MAE: {metrics['latitude_mae']:.6f}°")
                        print(f"   • Longitude MAE: {metrics['longitude_mae']:.6f}°")
                    
                    if 'training_samples' in metrics:
                        print(f"   • Training Samples: {metrics['training_samples']:,}")
                    
                    if 'test_samples' in metrics:
                        print(f"   • Test Samples: {metrics['test_samples']:,}")
                    
                    if 'final_loss' in metrics:
                        print(f"   • Final Training Loss: {metrics['final_loss']:.6f}")
        
        # Memory and performance stats
        print(f"\n   💾 System Performance:")
        print(f"   • Models Loaded: {self.mobility_predictor.models_loaded if self.mobility_predictor else False}")
        print(f"   • Prediction Mode: {'Enhanced LSTM' if self.mobility_predictor and self.mobility_predictor.models_loaded else 'Fallback Heuristics'}")
        
        # Dataset utilization
        if self.dataset_loader:
            stats = self.dataset_loader.get_dataset_statistics()
            total_trajectories = sum(stat['num_trajectories'] for stat in stats.values())
            total_points = sum(stat['total_points'] for stat in stats.values())
            print(f"   • Training Data: {total_trajectories:,} trajectories, {total_points:,} GPS points")
    
    def _calculate_distance_km(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in kilometers."""
        # Haversine formula
        R = 6371  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c


async def main():
    """Main demo execution."""
    demo = EnhancedLSTMDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())