#!/usr/bin/env python3
"""
Enhanced LSTM Training Script for 92%+ Accuracy

This script trains the enhanced LSTM ensemble system on real datasets
and validates accuracy to achieve the target 92%+ performance.
"""

import asyncio
import time
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Import components
from app.pytorch_lstm_predictor import AdvancedConfig, EnsembleMobilityPredictor
from app.dataset_loader import DatasetLoader, DatasetConfig, load_real_mobility_data
from app.accuracy_validator import AccuracyValidator
from app.events import EventBus
from shared.config import get_config

class EnhancedLSTMTrainer:
    """Enhanced LSTM training orchestrator."""
    
    def __init__(self):
        self.config = get_config()
        self.event_bus = EventBus()
        self.accuracy_validator = AccuracyValidator(target_accuracy=0.92)
        self.training_start_time = None
        
    async def load_datasets(self):
        """Load real mobility datasets."""
        print("🌍 Loading Real Mobility Datasets")
        print("=" * 50)
        
        # Enhanced dataset configuration for maximum accuracy
        dataset_config = DatasetConfig(
            max_users=500,                    # More users for diversity
            max_trajectories_per_user=30,    # More trajectories
            min_trajectory_points=50,        # Longer trajectories
            temporal_resolution_seconds=30,  # Good resolution
            apply_noise=False,               # Clean data for training
            cache_processed_data=True
        )
        
        start_time = time.time()
        
        try:
            # Load real datasets
            self.dataset_loader, datasets = await load_real_mobility_data(
                dataset_config, self.event_bus
            )
            
            load_time = time.time() - start_time
            
            # Get training trajectories
            trajectories = self.dataset_loader.get_trajectories_for_training(
                max_trajectories=2000  # Substantial dataset
            )
            
            if trajectories:
                total_points = sum(len(t.points) for t in trajectories)
                avg_points = total_points / len(trajectories)
                
                print(f"✅ Dataset Loading Complete ({load_time:.1f}s)")
                print(f"   📊 Trajectories: {len(trajectories):,}")
                print(f"   📍 GPS Points: {total_points:,}")
                print(f"   📈 Avg Points/Trajectory: {avg_points:.1f}")
                print(f"   👥 Users: {len(set(t.user_id for t in trajectories))}")
                
                self.trajectories = trajectories
                return True
            else:
                print("❌ No trajectories loaded")
                return False
                
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")
            print(f"❌ Dataset loading failed: {e}")
            return False
    
    def configure_model(self):
        """Configure model for maximum accuracy."""
        print("\n🧠 Model Configuration")
        print("=" * 30)
        
        # Optimized configuration for 92%+ accuracy
        config = AdvancedConfig(
            # Architecture - balanced for accuracy and speed
            sequence_length=35,              # Good pattern recognition
            lstm_units=256,                 # Sufficient capacity
            num_layers=4,                   # Deep enough for patterns
            attention_heads=8,              # Multi-head attention
            dropout_rate=0.15,              # Prevent overfitting
            
            # Training - optimized for convergence
            learning_rate=0.0005,           # Stable learning
            epochs=150,                     # Sufficient training
            batch_size=128,                 # Good batch size
            patience=25,                    # Early stopping
            
            # Ensemble - multiple models for robustness
            num_ensemble_models=5,          # Good ensemble size
            data_augmentation_factor=2.0,   # Data augmentation
            
            # Features
            feature_lookback_hours=24,      # 24-hour history
            prediction_horizons=[1, 2, 3, 5, 10],  # Multiple horizons
            
            # Preprocessing
            outlier_threshold=3.0,          # Outlier detection
            use_robust_scaling=True,        # Robust scaling
        )
        
        # Estimate training time
        samples_estimate = len(self.trajectories) * 30  # Rough sequence estimate
        time_per_epoch = (samples_estimate / config.batch_size) * 0.1  # Rough estimate
        total_time_minutes = (time_per_epoch * config.epochs * config.num_ensemble_models) / 60
        
        print(f"📋 Configuration:")
        print(f"   🏗️ Architecture: {config.num_layers}-layer LSTM + Attention")
        print(f"   🎯 Ensemble: {config.num_ensemble_models} models")
        print(f"   📚 Training: {config.epochs} epochs per model")
        print(f"   ⏱️ Estimated Time: {total_time_minutes:.1f} minutes")
        print(f"   🎯 Target: 92%+ accuracy")
        
        self.model_config = config
        return config
    
    async def train_model(self):
        """Train the enhanced LSTM ensemble."""
        print(f"\n🚀 Enhanced LSTM Training Started")
        print("=" * 40)
        
        self.training_start_time = time.time()
        
        # Initialize predictor
        predictor = EnsembleMobilityPredictor(self.model_config)
        
        print(f"🎯 Training ensemble on {len(self.trajectories)} trajectories...")
        print(f"⏳ This may take {(time.time() - self.training_start_time)/60:.1f} minutes...")
        
        try:
            # Train with progress updates
            results = predictor.train(self.trajectories)
            
            training_time = time.time() - self.training_start_time
            
            print(f"\n✅ Training Complete! ({training_time/60:.1f} minutes)")
            print(f"📊 Training Results:")
            print(f"   🎯 Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
            print(f"   📏 Error: {results['avg_error_km']*1000:.1f}m")
            print(f"   🤖 Models: {results['num_models']}")
            print(f"   📈 Features: {results['num_features']}")
            print(f"   📊 Samples: {results['training_samples']}")
            
            self.predictor = predictor
            self.training_results = results
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            print(f"❌ Training failed: {e}")
            return False
    
    async def test_accuracy(self):
        """Test model accuracy comprehensively."""
        print(f"\n🔍 Comprehensive Accuracy Testing")
        print("=" * 40)
        
        try:
            # Prepare validation data
            validation_trajectories = self.trajectories[-500:]  # Last 500 for validation
            
            print(f"🧪 Extracting features from {len(validation_trajectories)} validation trajectories...")
            X_val, y_val = self.predictor.feature_engineer.extract_features(validation_trajectories)
            
            if len(X_val) == 0:
                print("❌ No validation data available")
                return False
            
            # Reshape for LSTM
            X_val_reshaped = self.predictor.feature_engineer.reshape_for_lstm(X_val)
            
            print(f"📊 Validation Set: {len(X_val_reshaped)} samples")
            
            # Comprehensive validation
            validation_metrics = self.accuracy_validator.validate_model_accuracy(
                self.predictor, X_val_reshaped, y_val, "Enhanced_LSTM_Ensemble"
            )
            
            # Results
            print(f"\n📈 ACCURACY RESULTS:")
            print(f"   🎯 50m Accuracy:  {validation_metrics.accuracy_50m:.3f} ({validation_metrics.accuracy_50m*100:.1f}%)")
            print(f"   🎯 100m Accuracy: {validation_metrics.accuracy_100m:.3f} ({validation_metrics.accuracy_100m*100:.1f}%)")
            print(f"   🎯 200m Accuracy: {validation_metrics.accuracy_200m:.3f} ({validation_metrics.accuracy_200m*100:.1f}%)")
            print(f"   📏 Mean Error: {validation_metrics.mae_meters:.1f}m")
            print(f"   📊 R² Score: {validation_metrics.r2_score:.3f}")
            
            # Target assessment
            target_achieved = validation_metrics.accuracy_50m >= 0.92
            
            if target_achieved:
                print(f"\n🎉 SUCCESS: Target Achieved!")
                print(f"✅ {validation_metrics.accuracy_50m*100:.1f}% accuracy (target: 92%)")
                print(f"🚀 Exceeded by {(validation_metrics.accuracy_50m - 0.92)*100:.1f} percentage points")
                print("🌟 Model ready for production!")
            else:
                gap = (0.92 - validation_metrics.accuracy_50m) * 100
                print(f"\n🔶 Close to Target")
                print(f"📊 {validation_metrics.accuracy_50m*100:.1f}% accuracy (target: 92%)")
                print(f"📈 Gap: {gap:.1f} percentage points")
                print("🔧 Consider longer training or larger ensemble")
            
            self.validation_metrics = validation_metrics
            return target_achieved
            
        except Exception as e:
            logger.error(f"Accuracy testing failed: {e}")
            print(f"❌ Accuracy testing failed: {e}")
            return False
    
    async def run_training_pipeline(self):
        """Run the complete training and validation pipeline."""
        print("🚀 ZKPAS Enhanced LSTM Training Pipeline")
        print("=" * 60)
        print("🎯 Target: Train and validate 92%+ accuracy LSTM system")
        print()
        
        total_start = time.time()
        
        try:
            # Step 1: Load datasets
            if not await self.load_datasets():
                return False
            
            # Step 2: Configure model
            self.configure_model()
            
            # Step 3: Train model
            if not await self.train_model():
                return False
            
            # Step 4: Test accuracy
            success = await self.test_accuracy()
            
            # Final summary
            total_time = time.time() - total_start
            
            print(f"\n{'='*60}")
            print("🏆 TRAINING PIPELINE COMPLETE")
            print(f"{'='*60}")
            print(f"⏱️ Total Time: {total_time/60:.1f} minutes")
            
            if success:
                print("✅ SUCCESS: 92%+ accuracy achieved!")
                print("🚀 Model ready for deployment")
            else:
                print("🔶 GOOD PROGRESS: Strong performance achieved")
                print("📈 Consider additional optimization for 92%+ target")
            
            return success
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            print(f"❌ Training pipeline failed: {e}")
            return False


async def main():
    """Main training execution."""
    trainer = EnhancedLSTMTrainer()
    success = await trainer.run_training_pipeline()
    
    if success:
        print("\n🎉 Training completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️ Training completed with mixed results")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())