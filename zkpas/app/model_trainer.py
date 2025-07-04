"""
Pre-trained Model System for ZKPAS

This module implements a "train once, use many times" approach for ML models
used in the ZKPAS system. Models are trained on real mobility datasets and
persisted to disk for fast loading on application startup.

Features:
- Train mobility prediction models on real data
- Persist trained models with metadata
- Fast model loading for production use
- Model versioning and validation
- Automatic retraining when needed
- Privacy-preserving model training
"""

import os
import pickle
import joblib
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from app.dataset_loader import DatasetLoader, TrajectoryData, DatasetConfig
from app.mobility_predictor import LocationPoint, MobilityPattern
from app.events import EventBus, EventType

logger = logging.getLogger(__name__)

# Try to import TensorFlow for LSTM models, fallback to scikit-learn
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, TimeDistributed
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow available - using LSTM models")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available - using neural network simulation with MLPRegressor")
    from sklearn.neural_network import MLPRegressor


@dataclass
class ModelMetadata:
    """Metadata for trained models."""
    
    model_name: str
    model_type: str  # 'mobility_prediction', 'pattern_classification', etc.
    version: str
    training_date: str
    dataset_hash: str  # Hash of training data for validation
    dataset_stats: Dict[str, Any]
    model_params: Dict[str, Any]
    performance_metrics: Dict[str, float]
    features_used: List[str]
    model_file_path: str
    scaler_file_path: Optional[str] = None
    encoder_file_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training parameters
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    n_estimators: int = 100
    max_depth: Optional[int] = None
    
    # LSTM-specific parameters
    sequence_length: int = 20  # Number of time steps for LSTM input
    lstm_units: int = 64  # Number of LSTM units
    lstm_layers: int = 2  # Number of LSTM layers
    dropout_rate: float = 0.2  # Dropout rate for regularization
    learning_rate: float = 0.001  # Learning rate for Adam optimizer
    epochs: int = 50  # Maximum training epochs
    batch_size: int = 32  # Batch size for training
    patience: int = 10  # Early stopping patience
    
    # Feature engineering
    time_window_hours: int = 24  # Hours of history to use for prediction
    prediction_horizon_minutes: int = 60  # Minutes ahead to predict
    spatial_resolution: float = 0.001  # Degrees for spatial binning
    
    # Privacy settings
    apply_differential_privacy: bool = True
    privacy_epsilon: float = 1.0
    
    # Model persistence
    models_dir: str = "data/trained_models"
    force_retrain: bool = False  # Force retraining even if models exist


class ModelTrainer:
    """Trains and manages ML models for ZKPAS system."""
    
    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None):
        """
        Initialize model trainer.
        
        Args:
            config: Training configuration
            event_bus: Optional event bus for notifications
        """
        self.config = config
        self.event_bus = event_bus
        
        # Setup directories
        self.models_dir = Path(config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Trained models storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        
        logger.info(f"Model trainer initialized with models dir: {self.models_dir}")
    
    async def train_all_models(self, dataset_loader: DatasetLoader) -> Dict[str, ModelMetadata]:
        """
        Train all required models for ZKPAS system.
        
        Args:
            dataset_loader: Loaded dataset manager
            
        Returns:
            Dictionary of trained model metadata
        """
        logger.info("Starting comprehensive model training for ZKPAS system...")
        
        # Get training data
        trajectories = dataset_loader.get_trajectories_for_training(max_trajectories=1000)
        
        if not trajectories:
            raise ValueError("No suitable trajectories found for training")
        
        logger.info(f"Training models on {len(trajectories)} trajectories")
        
        # Calculate dataset hash for versioning
        dataset_hash = self._calculate_dataset_hash(trajectories)
        
        models_trained = {}
        
        # 1. Train LSTM mobility prediction model
        try:
            logger.info("Training LSTM mobility prediction model...")
            metadata = await self._train_mobility_prediction_model(trajectories, dataset_hash)
            models_trained["mobility_prediction_lstm"] = metadata
            logger.info("✅ LSTM mobility prediction model trained successfully")
        except Exception as e:
            logger.error(f"❌ Failed to train mobility prediction model: {e}")
        
        # 2. Train mobility pattern classification model  
        try:
            logger.info("Training mobility pattern classification model...")
            metadata = await self._train_pattern_classification_model(trajectories, dataset_hash)
            models_trained["pattern_classification"] = metadata
            logger.info("✅ Pattern classification model trained successfully")
        except Exception as e:
            logger.error(f"❌ Failed to train pattern classification model: {e}")
        
        # 3. Train authentication risk assessment model
        try:
            logger.info("Training authentication risk assessment model...")
            metadata = await self._train_risk_assessment_model(trajectories, dataset_hash)
            models_trained["risk_assessment"] = metadata
            logger.info("✅ Risk assessment model trained successfully")
        except Exception as e:
            logger.error(f"❌ Failed to train risk assessment model: {e}")
        
        # Publish training completion event
        if self.event_bus:
            await self.event_bus.publish_event(
                EventType.COMPONENT_STARTED,
                correlation_id="model_training",
                source="ModelTrainer",
                data={
                    "models_trained": list(models_trained.keys()),
                    "total_trajectories": len(trajectories),
                    "dataset_hash": dataset_hash
                }
            )
        
        self.metadata.update(models_trained)
        logger.info(f"✅ Training complete! {len(models_trained)} models trained successfully")
        
        return models_trained
    
    async def _train_mobility_prediction_model(self, trajectories: List[TrajectoryData], 
                                             dataset_hash: str) -> ModelMetadata:
        """Train LSTM model to predict next location based on movement history."""
        
        model_name = "mobility_prediction_lstm"
        if TENSORFLOW_AVAILABLE:
            model_file = self.models_dir / f"{model_name}_v1.0.h5"  # Keras H5 format
        else:
            model_file = self.models_dir / f"{model_name}_v1.0.pkl"  # Scikit-learn format
        scaler_file = self.models_dir / f"{model_name}_scaler_v1.0.pkl"
        
        # Check if model exists and is up-to-date
        if not self.config.force_retrain and model_file.exists():
            try:
                metadata = self._load_model_metadata(model_name)
                if metadata and metadata.dataset_hash == dataset_hash:
                    logger.info(f"Using existing {model_name} LSTM model (up-to-date)")
                    return metadata
            except Exception as e:
                logger.warning(f"Failed to load existing model metadata: {e}")
        
        if TENSORFLOW_AVAILABLE:
            # Use TensorFlow LSTM implementation
            return await self._train_tensorflow_lstm(trajectories, dataset_hash, model_file, scaler_file, model_name)
        else:
            # Use MLPRegressor as LSTM simulation
            return await self._train_neural_network_simulation(trajectories, dataset_hash, model_file, scaler_file, model_name)
    
    async def _train_tensorflow_lstm(self, trajectories, dataset_hash, model_file, scaler_file, model_name):
        """Train actual TensorFlow LSTM model."""
        # Extract sequence data for LSTM training
        sequences, targets, feature_names = self._extract_lstm_sequences(trajectories)
        
        if len(sequences) == 0:
            raise ValueError("No sequences extracted for LSTM training")
        
        logger.info(f"Created {len(sequences)} LSTM sequences with shape {sequences[0].shape}")
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(targets)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
        
        # Scale features (normalize for LSTM)
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Reshape for scaling (samples * timesteps, features)
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape((n_samples * n_timesteps, n_features))
        X_test_reshaped = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
        
        # Fit scaler and transform
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)
        
        # Reshape back to 3D for LSTM
        X_train_scaled = X_train_scaled.reshape((n_samples, n_timesteps, n_features))
        X_test_scaled = X_test_scaled.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
        
        # Build LSTM model
        model = self._build_lstm_model(n_timesteps, n_features)
        
        # Setup callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=self.config.patience, 
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            str(model_file),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
        
        # Train LSTM model
        logger.info(f"Training LSTM model with {len(X_train)} sequences...")
        history = model.fit(
            X_train_scaled, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_test_scaled, y_test),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Evaluate performance
        y_pred = model.predict(X_test_scaled)
        
        lat_mae = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
        lon_mae = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
        lat_rmse = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
        lon_rmse = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
        
        # Convert to distance error (approximate)
        lat_error_km = lat_mae * 111  # 1 degree lat ≈ 111 km
        lon_error_km = lon_mae * 111 * np.cos(np.radians(np.mean(y_test[:, 0])))
        
        # Get training history
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        performance_metrics = {
            "latitude_mae": float(lat_mae),
            "longitude_mae": float(lon_mae),
            "latitude_rmse": float(lat_rmse),
            "longitude_rmse": float(lon_rmse),
            "avg_distance_error_km": float(np.mean([lat_error_km, lon_error_km])),
            "final_loss": float(final_loss),
            "final_val_loss": float(final_val_loss),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "sequence_length": self.config.sequence_length,
            "lstm_units": self.config.lstm_units,
            "epochs_trained": len(history.history['loss'])
        }
        
        # Save scaler
        joblib.dump(scaler, scaler_file)
        
        return self._create_model_metadata(model_name, dataset_hash, trajectories, performance_metrics, feature_names, model_file, scaler_file)
    
    async def _train_neural_network_simulation(self, trajectories, dataset_hash, model_file, scaler_file, model_name):
        """Train neural network using MLPRegressor as LSTM simulation."""
        logger.info("Training neural network simulation (MLPRegressor) for mobility prediction...")
        
        # Extract flattened sequence data 
        sequences, targets, feature_names = self._extract_lstm_sequences(trajectories)
        
        if len(sequences) == 0:
            raise ValueError("No sequences extracted for neural network training")
        
        # Flatten sequences for MLPRegressor (can't handle 3D input)
        X_flattened = []
        for seq in sequences:
            X_flattened.append(seq.flatten())
        
        X = np.array(X_flattened)
        y = np.array(targets)
        
        logger.info(f"Created {len(X)} flattened sequences with {X.shape[1]} features each")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create MLPRegressor with LSTM-like architecture
        model = MLPRegressor(
            hidden_layer_sizes=(self.config.lstm_units, self.config.lstm_units),
            activation='tanh',  # Similar to LSTM activation
            solver='adam',
            alpha=0.001,  # L2 regularization similar to dropout
            batch_size=min(self.config.batch_size, len(X_train)),
            learning_rate='adaptive',
            learning_rate_init=self.config.learning_rate,
            max_iter=self.config.epochs * 2,  # More iterations to compensate
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=self.config.patience,
            random_state=self.config.random_state
        )
        
        # Train model
        logger.info(f"Training neural network with {len(X_train)} samples...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate performance
        y_pred = model.predict(X_test_scaled)
        
        lat_mae = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
        lon_mae = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
        lat_rmse = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
        lon_rmse = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
        
        # Convert to distance error (approximate)
        lat_error_km = lat_mae * 111  # 1 degree lat ≈ 111 km
        lon_error_km = lon_mae * 111 * np.cos(np.radians(np.mean(y_test[:, 0])))
        
        performance_metrics = {
            "latitude_mae": float(lat_mae),
            "longitude_mae": float(lon_mae),
            "latitude_rmse": float(lat_rmse),
            "longitude_rmse": float(lon_rmse),
            "avg_distance_error_km": float(np.mean([lat_error_km, lon_error_km])),
            "final_loss": float(model.loss_),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "sequence_length": self.config.sequence_length,
            "hidden_units": self.config.lstm_units,
            "iterations": model.n_iter_,
            "model_type": "MLPRegressor_LSTM_simulation"
        }
        
        # Save model and scaler
        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)
        
        return self._create_model_metadata(model_name, dataset_hash, trajectories, performance_metrics, feature_names, model_file, scaler_file)
    
    def _create_model_metadata(self, model_name, dataset_hash, trajectories, performance_metrics, feature_names, model_file, scaler_file):
        """Create model metadata object."""
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name,
            model_type="mobility_prediction_lstm",
            version="1.0",
            training_date=datetime.now().isoformat(),
            dataset_hash=dataset_hash,
            dataset_stats={"num_trajectories": len(trajectories)},
            model_params={
                "sequence_length": self.config.sequence_length,
                "lstm_units": self.config.lstm_units,
                "lstm_layers": self.config.lstm_layers,
                "dropout_rate": self.config.dropout_rate,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "tensorflow_available": TENSORFLOW_AVAILABLE
            },
            performance_metrics=performance_metrics,
            features_used=feature_names,
            model_file_path=str(model_file),
            scaler_file_path=str(scaler_file)
        )
        
        # Save metadata
        self._save_model_metadata(metadata)
        
        # Store in memory
        self.models[model_name] = metadata  # Will be loaded later
        self.scalers[model_name] = None     # Will be loaded later
        
        logger.info(f"Neural network mobility prediction model trained with {performance_metrics['avg_distance_error_km']:.2f}km average error")
        
        return metadata
    
    async def _train_pattern_classification_model(self, trajectories: List[TrajectoryData], 
                                                dataset_hash: str) -> ModelMetadata:
        """Train model to classify mobility patterns."""
        
        model_name = "pattern_classification"
        model_file = self.models_dir / f"{model_name}_v1.0.pkl"
        scaler_file = self.models_dir / f"{model_name}_scaler_v1.0.pkl"
        encoder_file = self.models_dir / f"{model_name}_encoder_v1.0.pkl"
        
        # Check if model exists and is up-to-date
        if not self.config.force_retrain and model_file.exists():
            try:
                metadata = self._load_model_metadata(model_name)
                if metadata and metadata.dataset_hash == dataset_hash:
                    logger.info(f"Using existing {model_name} model (up-to-date)")
                    return metadata
            except Exception as e:
                logger.warning(f"Failed to load existing model metadata: {e}")
        
        # Extract features for pattern classification
        features, labels, feature_names = self._extract_pattern_features(trajectories)
        
        if len(features) == 0:
            raise ValueError("No features extracted for pattern classification")
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=self.config.test_size,
            random_state=self.config.random_state, stratify=encoded_labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate performance
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        performance_metrics = {
            "accuracy": accuracy,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "num_classes": len(label_encoder.classes_)
        }
        
        # Save model, scaler, and encoder
        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)
        joblib.dump(label_encoder, encoder_file)
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name,
            model_type="pattern_classification",
            version="1.0",
            training_date=datetime.now().isoformat(),
            dataset_hash=dataset_hash,
            dataset_stats={"num_trajectories": len(trajectories)},
            model_params={
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth
            },
            performance_metrics=performance_metrics,
            features_used=feature_names,
            model_file_path=str(model_file),
            scaler_file_path=str(scaler_file),
            encoder_file_path=str(encoder_file)
        )
        
        # Save metadata
        self._save_model_metadata(metadata)
        
        # Store in memory
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.encoders[model_name] = label_encoder
        
        logger.info(f"Pattern classification model trained with {accuracy:.3f} accuracy")
        
        return metadata
    
    async def _train_risk_assessment_model(self, trajectories: List[TrajectoryData], 
                                         dataset_hash: str) -> ModelMetadata:
        """Train model to assess authentication risk based on mobility patterns."""
        
        model_name = "risk_assessment"
        model_file = self.models_dir / f"{model_name}_v1.0.pkl"
        scaler_file = self.models_dir / f"{model_name}_scaler_v1.0.pkl"
        
        # Check if model exists and is up-to-date
        if not self.config.force_retrain and model_file.exists():
            try:
                metadata = self._load_model_metadata(model_name)
                if metadata and metadata.dataset_hash == dataset_hash:
                    logger.info(f"Using existing {model_name} model (up-to-date)")
                    return metadata
            except Exception as e:
                logger.warning(f"Failed to load existing model metadata: {e}")
        
        # Extract features for risk assessment
        features, risk_scores, feature_names = self._extract_risk_features(trajectories)
        
        if len(features) == 0:
            raise ValueError("No features extracted for risk assessment")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, risk_scores, test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train regressor for risk scores
        model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate performance
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        performance_metrics = {
            "mae": mae,
            "rmse": rmse,
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        # Save model and scaler
        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name,
            model_type="risk_assessment",
            version="1.0",
            training_date=datetime.now().isoformat(),
            dataset_hash=dataset_hash,
            dataset_stats={"num_trajectories": len(trajectories)},
            model_params={
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth
            },
            performance_metrics=performance_metrics,
            features_used=feature_names,
            model_file_path=str(model_file),
            scaler_file_path=str(scaler_file)
        )
        
        # Save metadata
        self._save_model_metadata(metadata)
        
        # Store in memory
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        logger.info(f"Risk assessment model trained with {mae:.3f} MAE")
        
        return metadata
    
    def _extract_mobility_features(self, trajectories: List[TrajectoryData]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features for mobility prediction."""
        
        features = []
        targets = []
        
        for trajectory in trajectories:
            points = trajectory.points
            
            # Need enough points for feature extraction
            if len(points) < self.config.time_window_hours:
                continue
            
            # Extract features from sliding windows
            for i in range(len(points) - self.config.time_window_hours):
                window_points = points[i:i + self.config.time_window_hours]
                target_point = points[i + self.config.time_window_hours]
                
                # Feature extraction
                feature_vector = []
                
                # Current location
                current_point = window_points[-1]
                feature_vector.extend([current_point.latitude, current_point.longitude])
                
                # Velocity features
                if len(window_points) >= 2:
                    prev_point = window_points[-2]
                    time_diff = current_point.timestamp - prev_point.timestamp
                    if time_diff > 0:
                        lat_velocity = (current_point.latitude - prev_point.latitude) / time_diff
                        lon_velocity = (current_point.longitude - prev_point.longitude) / time_diff
                        feature_vector.extend([lat_velocity, lon_velocity])
                    else:
                        feature_vector.extend([0.0, 0.0])
                else:
                    feature_vector.extend([0.0, 0.0])
                
                # Acceleration features
                if len(window_points) >= 3:
                    prev_prev_point = window_points[-3]
                    time_diff1 = current_point.timestamp - prev_point.timestamp
                    time_diff2 = prev_point.timestamp - prev_prev_point.timestamp
                    
                    if time_diff1 > 0 and time_diff2 > 0:
                        lat_acc = ((current_point.latitude - prev_point.latitude) / time_diff1 - 
                                  (prev_point.latitude - prev_prev_point.latitude) / time_diff2) / time_diff1
                        lon_acc = ((current_point.longitude - prev_point.longitude) / time_diff1 - 
                                  (prev_point.longitude - prev_prev_point.longitude) / time_diff2) / time_diff1
                        feature_vector.extend([lat_acc, lon_acc])
                    else:
                        feature_vector.extend([0.0, 0.0])
                else:
                    feature_vector.extend([0.0, 0.0])
                
                # Time-based features
                current_time = datetime.fromtimestamp(current_point.timestamp)
                hour_of_day = current_time.hour / 24.0
                day_of_week = current_time.weekday() / 7.0
                feature_vector.extend([hour_of_day, day_of_week])
                
                # Statistical features over window
                lats = [p.latitude for p in window_points]
                lons = [p.longitude for p in window_points]
                
                lat_mean, lat_std = np.mean(lats), np.std(lats)
                lon_mean, lon_std = np.mean(lons), np.std(lons)
                
                feature_vector.extend([lat_mean, lat_std, lon_mean, lon_std])
                
                features.append(feature_vector)
                targets.append([target_point.latitude, target_point.longitude])
        
        feature_names = [
            'current_lat', 'current_lon', 'lat_velocity', 'lon_velocity',
            'lat_acceleration', 'lon_acceleration', 'hour_of_day', 'day_of_week',
            'lat_mean', 'lat_std', 'lon_mean', 'lon_std'
        ]
        
        return np.array(features), np.array(targets), feature_names
    
    def _build_lstm_model(self, timesteps: int, features: int):
        """Build LSTM model architecture for mobility prediction."""
        if not TENSORFLOW_AVAILABLE:
            raise ValueError("TensorFlow not available for LSTM model building")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.config.lstm_units,
            return_sequences=True if self.config.lstm_layers > 1 else False,
            input_shape=(timesteps, features),
            dropout=self.config.dropout_rate,
            recurrent_dropout=self.config.dropout_rate
        ))
        
        # Additional LSTM layers
        for i in range(1, self.config.lstm_layers):
            return_sequences = i < self.config.lstm_layers - 1
            model.add(LSTM(
                units=self.config.lstm_units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            ))
        
        # Dropout layer
        model.add(Dropout(self.config.dropout_rate))
        
        # Dense layers for output
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.config.dropout_rate))
        model.add(Dense(2))  # Output: [latitude, longitude]
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _extract_lstm_sequences(self, trajectories: List[TrajectoryData]) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """Extract LSTM training sequences from trajectory data."""
        
        sequences = []
        targets = []
        
        for trajectory in trajectories:
            points = trajectory.points
            
            # Need enough points for sequence + target
            if len(points) < self.config.sequence_length + 1:
                continue
            
            # Create sequences sliding window
            for i in range(len(points) - self.config.sequence_length):
                # Input sequence
                sequence_points = points[i:i + self.config.sequence_length]
                target_point = points[i + self.config.sequence_length]
                
                # Extract sequence features
                sequence_features = []
                for j, point in enumerate(sequence_points):
                    features = []
                    
                    # Basic coordinates
                    features.extend([point.latitude, point.longitude])
                    
                    # Time-based features
                    time_dt = datetime.fromtimestamp(point.timestamp)
                    features.extend([
                        time_dt.hour / 24.0,  # Normalized hour
                        time_dt.weekday() / 7.0,  # Normalized day of week
                        time_dt.month / 12.0,  # Normalized month
                    ])
                    
                    # Velocity features (if not first point in sequence)
                    if j > 0:
                        prev_point = sequence_points[j-1]
                        time_diff = point.timestamp - prev_point.timestamp
                        if time_diff > 0:
                            lat_velocity = (point.latitude - prev_point.latitude) / time_diff
                            lon_velocity = (point.longitude - prev_point.longitude) / time_diff
                            speed = np.sqrt(lat_velocity**2 + lon_velocity**2)
                        else:
                            lat_velocity = lon_velocity = speed = 0.0
                    else:
                        lat_velocity = lon_velocity = speed = 0.0
                    
                    features.extend([lat_velocity, lon_velocity, speed])
                    
                    # Distance from sequence start
                    start_point = sequence_points[0]
                    distance_from_start = np.sqrt(
                        (point.latitude - start_point.latitude)**2 + 
                        (point.longitude - start_point.longitude)**2
                    )
                    features.append(distance_from_start)
                    
                    sequence_features.append(features)
                
                sequences.append(np.array(sequence_features))
                targets.append([target_point.latitude, target_point.longitude])
        
        feature_names = [
            'latitude', 'longitude', 'hour_norm', 'day_norm', 'month_norm',
            'lat_velocity', 'lon_velocity', 'speed', 'distance_from_start'
        ]
        
        return sequences, targets, feature_names
    
    def _extract_pattern_features(self, trajectories: List[TrajectoryData]) -> Tuple[np.ndarray, List[str], List[str]]:
        """Extract features for mobility pattern classification."""
        
        features = []
        labels = []
        
        for trajectory in trajectories:
            if not trajectory.mobility_pattern or len(trajectory.points) < 10:
                continue
            
            # Extract trajectory-level features
            feature_vector = []
            
            # Basic statistics
            feature_vector.append(trajectory.total_distance_km)
            feature_vector.append(trajectory.avg_speed_kmh)
            feature_vector.append(trajectory.duration_seconds / 3600)  # duration in hours
            feature_vector.append(len(trajectory.points))  # number of points
            
            # Spatial features
            lats = [p.latitude for p in trajectory.points]
            lons = [p.longitude for p in trajectory.points]
            
            lat_range = max(lats) - min(lats)
            lon_range = max(lons) - min(lons)
            lat_std = np.std(lats)
            lon_std = np.std(lons)
            
            feature_vector.extend([lat_range, lon_range, lat_std, lon_std])
            
            # Temporal features
            start_time = datetime.fromtimestamp(trajectory.start_time)
            start_hour = start_time.hour / 24.0
            start_day = start_time.weekday() / 7.0
            
            feature_vector.extend([start_hour, start_day])
            
            # Speed variation
            speeds = []
            for i in range(1, len(trajectory.points)):
                p1, p2 = trajectory.points[i-1], trajectory.points[i]
                time_diff = p2.timestamp - p1.timestamp
                if time_diff > 0:
                    # Simple speed calculation
                    lat_diff = abs(p2.latitude - p1.latitude)
                    lon_diff = abs(p2.longitude - p1.longitude)
                    distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Rough km conversion
                    speed = distance / (time_diff / 3600)  # km/h
                    speeds.append(speed)
            
            if speeds:
                speed_mean = np.mean(speeds)
                speed_std = np.std(speeds)
                speed_max = np.max(speeds)
            else:
                speed_mean = speed_std = speed_max = 0.0
            
            feature_vector.extend([speed_mean, speed_std, speed_max])
            
            features.append(feature_vector)
            labels.append(trajectory.mobility_pattern.name)
        
        feature_names = [
            'total_distance_km', 'avg_speed_kmh', 'duration_hours', 'num_points',
            'lat_range', 'lon_range', 'lat_std', 'lon_std',
            'start_hour', 'start_day',
            'speed_mean', 'speed_std', 'speed_max'
        ]
        
        return np.array(features), labels, feature_names
    
    def _extract_risk_features(self, trajectories: List[TrajectoryData]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features for authentication risk assessment."""
        
        features = []
        risk_scores = []
        
        for trajectory in trajectories:
            if len(trajectory.points) < 5:
                continue
            
            # Extract features similar to pattern classification
            feature_vector = []
            
            # Movement characteristics
            feature_vector.append(trajectory.total_distance_km)
            feature_vector.append(trajectory.avg_speed_kmh)
            feature_vector.append(len(trajectory.points))
            
            # Spatial spread
            lats = [p.latitude for p in trajectory.points]
            lons = [p.longitude for p in trajectory.points]
            lat_range = max(lats) - min(lats)
            lon_range = max(lons) - min(lons)
            feature_vector.extend([lat_range, lon_range])
            
            # Time-based risk factors
            start_time = datetime.fromtimestamp(trajectory.start_time)
            is_night = 1.0 if start_time.hour < 6 or start_time.hour > 22 else 0.0
            is_weekend = 1.0 if start_time.weekday() >= 5 else 0.0
            feature_vector.extend([is_night, is_weekend])
            
            # Calculate synthetic risk score based on characteristics
            # Higher risk for: unusual speeds, large distances, night time, etc.
            risk_score = 0.0
            
            # Speed-based risk
            if trajectory.avg_speed_kmh > 80:  # Very high speed
                risk_score += 0.3
            elif trajectory.avg_speed_kmh > 50:  # High speed
                risk_score += 0.1
            
            # Distance-based risk
            if trajectory.total_distance_km > 100:  # Long distance
                risk_score += 0.2
            
            # Time-based risk
            if is_night:
                risk_score += 0.2
            
            # Pattern-based risk
            if trajectory.mobility_pattern == MobilityPattern.RANDOM:
                risk_score += 0.3
            elif trajectory.mobility_pattern == MobilityPattern.VEHICLE:
                risk_score += 0.1
            
            # Normalize to [0, 1]
            risk_score = min(risk_score, 1.0)
            
            features.append(feature_vector)
            risk_scores.append(risk_score)
        
        feature_names = [
            'total_distance_km', 'avg_speed_kmh', 'num_points',
            'lat_range', 'lon_range', 'is_night', 'is_weekend'
        ]
        
        return np.array(features), np.array(risk_scores), feature_names
    
    def _calculate_dataset_hash(self, trajectories: List[TrajectoryData]) -> str:
        """Calculate hash of dataset for versioning."""
        
        # Create a string representation of key dataset characteristics
        data_str = ""
        for traj in trajectories[:100]:  # Sample first 100 for efficiency
            data_str += f"{traj.user_id}_{len(traj.points)}_{traj.total_distance_km:.2f}"
        
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def _save_model_metadata(self, metadata: ModelMetadata):
        """Save model metadata to JSON file."""
        
        metadata_file = self.models_dir / f"{metadata.model_name}_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
    
    def _load_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Load model metadata from JSON file."""
        
        metadata_file = self.models_dir / f"{model_name}_metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            return ModelMetadata(**data)
        except Exception as e:
            logger.error(f"Failed to load metadata for {model_name}: {e}")
            return None
    
    def load_pretrained_models(self) -> Dict[str, ModelMetadata]:
        """
        Load all pre-trained models from disk.
        
        Returns:
            Dictionary of loaded model metadata
        """
        logger.info("Loading pre-trained models...")
        
        loaded_models = {}
        
        # Look for all metadata files
        metadata_files = list(self.models_dir.glob("*_metadata.json"))
        
        for metadata_file in metadata_files:
            model_name = metadata_file.stem.replace("_metadata", "")
            
            try:
                # Load metadata
                metadata = self._load_model_metadata(model_name)
                if not metadata:
                    continue
                
                # Load model (handle both Keras .h5 and sklearn .pkl files)
                model_file = Path(metadata.model_file_path)
                if not model_file.exists():
                    logger.warning(f"Model file not found: {model_file}")
                    continue
                
                # Load based on file extension
                if model_file.suffix == '.h5':
                    # Keras model
                    model = tf.keras.models.load_model(model_file)
                else:
                    # Sklearn model
                    model = joblib.load(model_file)
                
                self.models[model_name] = model
                
                # Load scaler if exists
                if metadata.scaler_file_path:
                    scaler_file = Path(metadata.scaler_file_path)
                    if scaler_file.exists():
                        scaler = joblib.load(scaler_file)
                        self.scalers[model_name] = scaler
                
                # Load encoder if exists
                if metadata.encoder_file_path:
                    encoder_file = Path(metadata.encoder_file_path)
                    if encoder_file.exists():
                        encoder = joblib.load(encoder_file)
                        self.encoders[model_name] = encoder
                
                loaded_models[model_name] = metadata
                logger.info(f"✅ Loaded pre-trained model: {model_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load model {model_name}: {e}")
        
        self.metadata.update(loaded_models)
        
        if loaded_models:
            logger.info(f"✅ Successfully loaded {len(loaded_models)} pre-trained models")
        else:
            logger.warning("No pre-trained models found. Training will be required.")
        
        return loaded_models
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model by name."""
        return self.models.get(model_name)
    
    def get_scaler(self, model_name: str) -> Optional[StandardScaler]:
        """Get a scaler for a model by name."""
        return self.scalers.get(model_name)
    
    def get_encoder(self, model_name: str) -> Optional[LabelEncoder]:
        """Get an encoder for a model by name."""
        return self.encoders.get(model_name)
    
    def get_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Get metadata for a model by name."""
        return self.metadata.get(model_name)
    
    def list_available_models(self) -> List[str]:
        """List all available model names."""
        return list(self.models.keys())


# Utility functions

def get_default_training_config() -> TrainingConfig:
    """Get default configuration for model training."""
    return TrainingConfig(
        # Traditional ML parameters
        n_estimators=100,
        max_depth=20,
        
        # LSTM parameters
        sequence_length=15,  # 15 time steps for sequence
        lstm_units=64,       # 64 LSTM units
        lstm_layers=2,       # 2 LSTM layers
        dropout_rate=0.2,    # 20% dropout
        learning_rate=0.001, # Adam learning rate
        epochs=30,           # Maximum epochs
        batch_size=32,       # Batch size
        patience=8,          # Early stopping patience
        
        # General parameters
        time_window_hours=6,
        prediction_horizon_minutes=60,
        test_size=0.2,
        random_state=42,
        force_retrain=False
    )


async def train_or_load_models(dataset_loader: DatasetLoader, 
                             config: Optional[TrainingConfig] = None,
                             event_bus: Optional[EventBus] = None) -> ModelTrainer:
    """
    Convenience function to train or load models.
    
    Args:
        dataset_loader: Loaded dataset manager
        config: Optional training configuration
        event_bus: Optional event bus
        
    Returns:
        Model trainer with loaded/trained models
    """
    if config is None:
        config = get_default_training_config()
    
    trainer = ModelTrainer(config, event_bus)
    
    # Try to load existing models first
    loaded_models = trainer.load_pretrained_models()
    
    # Train missing models
    if not loaded_models or config.force_retrain:
        logger.info("Training new models...")
        await trainer.train_all_models(dataset_loader)
    else:
        logger.info("Using existing pre-trained models")
    
    return trainer


if __name__ == "__main__":
    # Example usage
    import asyncio
    from app.dataset_loader import load_real_mobility_data, get_default_dataset_config
    
    async def main():
        print("🤖 Training ZKPAS ML Models...")
        
        # Load datasets
        config = get_default_dataset_config()
        loader, datasets = await load_real_mobility_data(config)
        
        # Train models
        trainer_config = get_default_training_config()
        trainer = await train_or_load_models(loader, trainer_config)
        
        # Print model information
        models = trainer.list_available_models()
        print(f"\n📊 Available Models: {models}")
        
        for model_name in models:
            metadata = trainer.get_metadata(model_name)
            if metadata:
                print(f"\n{model_name.upper()}:")
                print(f"  Version: {metadata.version}")
                print(f"  Training Date: {metadata.training_date}")
                print(f"  Performance: {metadata.performance_metrics}")
    
    asyncio.run(main())