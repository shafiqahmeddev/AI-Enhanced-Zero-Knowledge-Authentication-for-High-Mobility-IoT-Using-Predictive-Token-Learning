"""
Mobility Prediction Module for ZKPAS IoT Devices

This module implements machine learning-based mobility prediction for IoT devices
to enable proactive authentication and seamless handoffs between gateways.

Enhanced to use pre-trained models on real-world mobility datasets.
"""

import asyncio
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
from enum import Enum, auto
from datetime import datetime

from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.events import Event, EventBus, EventType


class MobilityPattern(Enum):
    """Types of mobility patterns for IoT devices."""
    
    STATIONARY = auto()     # Device rarely moves
    PERIODIC = auto()       # Predictable movement patterns
    RANDOM = auto()         # Unpredictable movement
    COMMUTER = auto()       # Regular home-work patterns
    VEHICLE = auto()        # High-speed movement patterns


@dataclass
class LocationPoint:
    """Represents a location point with timestamp."""
    
    latitude: float
    longitude: float
    timestamp: float
    altitude: Optional[float] = None
    accuracy: Optional[float] = None  # GPS accuracy in meters
    speed: Optional[float] = None     # Speed in m/s
    bearing: Optional[float] = None   # Direction in degrees


@dataclass
class MobilityFeatures:
    """Features extracted from mobility data for ML prediction."""
    
    # Temporal features
    hour_of_day: float
    day_of_week: float
    is_weekend: bool
    
    # Movement features
    distance_from_last: float
    speed: float
    acceleration: float
    direction_change: float
    
    # Historical features
    avg_speed_last_hour: float
    distance_traveled_today: float
    visits_to_current_area: int
    time_since_last_movement: float
    
    # Context features
    is_charging: bool = False
    signal_strength: float = 0.0
    battery_level: float = 1.0


@dataclass
class MobilityPrediction:
    """Represents a mobility prediction for a device."""
    
    device_id: str
    predicted_location: LocationPoint
    confidence: float
    time_horizon: float  # Prediction time horizon in seconds
    mobility_pattern: MobilityPattern
    next_gateway: Optional[str] = None
    handoff_probability: float = 0.0


class MobilityPredictor:
    """Enhanced ML-based mobility predictor using pre-trained models on real datasets."""
    
    def __init__(self, event_bus: EventBus, model_trainer=None):
        """
        Initialize mobility predictor.
        
        Args:
            event_bus: Event bus for communication
            model_trainer: Optional pre-trained model trainer instance
        """
        self.event_bus = event_bus
        self.model_trainer = model_trainer
        
        # Pre-trained models (loaded from model_trainer)
        self.mobility_model = None
        self.pattern_model = None
        self.risk_model = None
        self.mobility_scaler = None
        self.pattern_scaler = None
        self.risk_scaler = None
        self.pattern_encoder = None
        
        # Real-time tracking data
        self.mobility_history: Dict[str, List[LocationPoint]] = {}
        self.feature_history: Dict[str, List[MobilityFeatures]] = {}
        
        # Model status
        self.models_loaded = False
        self.model_performance = {}
        
        # Configuration
        self.max_history_points = 1000
        self.min_prediction_points = 5
        self.prediction_horizons = [60, 300, 900]  # 1min, 5min, 15min
        self.time_window_hours = 6  # Hours of history for prediction features
        
        # Load pre-trained models if available
        if model_trainer:
            self._load_pretrained_models()
        
        # Setup event handlers
        self._setup_event_handlers()
        
        logger.info(f"Enhanced mobility predictor initialized (models_loaded={self.models_loaded})")
    
    def _load_pretrained_models(self):
        """Load pre-trained models from model trainer."""
        try:
            # Load mobility prediction model (try LSTM first, then fallback to traditional)
            self.mobility_model = self.model_trainer.get_model("mobility_prediction_lstm")
            self.mobility_scaler = self.model_trainer.get_scaler("mobility_prediction_lstm")
            
            # Fallback to traditional model if LSTM not available
            if self.mobility_model is None:
                self.mobility_model = self.model_trainer.get_model("mobility_prediction")
                self.mobility_scaler = self.model_trainer.get_scaler("mobility_prediction")
            
            # Load pattern classification model
            self.pattern_model = self.model_trainer.get_model("pattern_classification")
            self.pattern_scaler = self.model_trainer.get_scaler("pattern_classification")
            self.pattern_encoder = self.model_trainer.get_encoder("pattern_classification")
            
            # Load risk assessment model
            self.risk_model = self.model_trainer.get_model("risk_assessment")
            self.risk_scaler = self.model_trainer.get_scaler("risk_assessment")
            
            # Check if all models loaded successfully
            models_available = [
                self.mobility_model is not None,
                self.pattern_model is not None,
                self.risk_model is not None
            ]
            
            # Log which mobility model is being used
            if self.mobility_model is not None:
                lstm_metadata = self.model_trainer.get_metadata("mobility_prediction_lstm")
                if lstm_metadata:
                    logger.info("✅ Using LSTM mobility prediction model")
                else:
                    logger.info("✅ Using traditional Random Forest mobility prediction model")
            
            self.models_loaded = all(models_available)
            
            if self.models_loaded:
                # Load performance metrics
                for model_name in ["mobility_prediction_lstm", "mobility_prediction", "pattern_classification", "risk_assessment"]:
                    metadata = self.model_trainer.get_metadata(model_name)
                    if metadata:
                        self.model_performance[model_name] = metadata.performance_metrics
                
                logger.info("✅ All pre-trained models loaded successfully")
            else:
                logger.warning(f"⚠️ Some models missing: mobility={self.mobility_model is not None}, "
                             f"pattern={self.pattern_model is not None}, risk={self.risk_model is not None}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load pre-trained models: {e}")
            self.models_loaded = False
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for mobility-related events."""
        # Subscribe to location updates
        self.event_bus.subscribe_sync(EventType.LOCATION_CHANGED, self._handle_location_update)
        
        # Subscribe to authentication events for context
        self.event_bus.subscribe_sync(EventType.DEVICE_AUTHENTICATED, self._handle_auth_event)
        self.event_bus.subscribe_sync(EventType.MOBILITY_PREDICTED, self._handle_prediction_feedback)
    
    async def _handle_location_update(self, event: Event) -> None:
        """Handle location update event."""
        device_id = event.data.get("device_id")
        location_data = event.data.get("location")
        
        if device_id and location_data:
            location = LocationPoint(
                latitude=location_data["latitude"],
                longitude=location_data["longitude"],
                timestamp=location_data["timestamp"],
                altitude=location_data.get("altitude")
            )
            await self.update_location(device_id, location)
    
    async def _handle_auth_event(self, event: Event) -> None:
        """Handle authentication event for contextual information."""
        # This could be used to correlate authentication patterns with mobility
        device_id = event.data.get("device_id")
        if device_id:
            logger.debug(f"Authentication event for device {device_id}")
    
    async def _handle_prediction_feedback(self, event: Event) -> None:
        """Handle prediction feedback for model improvement."""
        # This could be used for online learning in future versions
        pass
        device_id = event.data.get("device_id")
        location_data = event.data.get("location")
        
        if not device_id or not location_data:
            return
        
        location = LocationPoint(
            latitude=location_data["latitude"],
            longitude=location_data["longitude"],
            timestamp=location_data.get("timestamp", time.time()),
            altitude=location_data.get("altitude"),
            accuracy=location_data.get("accuracy"),
            speed=location_data.get("speed"),
            bearing=location_data.get("bearing")
        )
        
        # Update mobility history
        await self.update_location(device_id, location)
        
        # Generate predictions if enough data
        if len(self.mobility_history.get(device_id, [])) >= 10:
            predictions = await self.predict_mobility(device_id)
            
            # Publish prediction events
            for prediction in predictions:
                await self.event_bus.publish_event(
                    event_type=EventType.MOBILITY_PREDICTED,
                    correlation_id=event.correlation_id,
                    source="mobility_predictor",
                    target=device_id,
                    data={
                        "prediction": self._prediction_to_dict(prediction),
                        "device_id": device_id
                    }
                )
    
    async def _handle_prediction_feedback(self, event: Event) -> None:
        """Handle prediction feedback for model improvement."""
        prediction_data = event.data.get("prediction")
        actual_location = event.data.get("actual_location")
        
        if prediction_data and actual_location:
            # Calculate prediction error for model improvement
            predicted_lat = prediction_data["predicted_location"]["latitude"]
            predicted_lon = prediction_data["predicted_location"]["longitude"]
            actual_lat = actual_location["latitude"]
            actual_lon = actual_location["longitude"]
            
            error = self._calculate_distance(
                predicted_lat, predicted_lon,
                actual_lat, actual_lon
            )
            
            logger.debug(f"Prediction error: {error:.2f} meters")
    
    async def update_location(self, device_id: str, location: LocationPoint) -> None:
        """Update location history for a device."""
        if device_id not in self.mobility_history:
            self.mobility_history[device_id] = []
        
        self.mobility_history[device_id].append(location)
        
        # Maintain history size limit
        if len(self.mobility_history[device_id]) > self.max_history_points:
            self.mobility_history[device_id] = self.mobility_history[device_id][-self.max_history_points:]
        
        # Extract features for ML
        features = self._extract_features(device_id, location)
        if features:
            if device_id not in self.feature_history:
                self.feature_history[device_id] = []
            
            self.feature_history[device_id].append(features)
            
            # Maintain feature history size
            if len(self.feature_history[device_id]) > self.max_history_points:
                self.feature_history[device_id] = self.feature_history[device_id][-self.max_history_points:]
        
        # Check if we need to retrain models (simplified for pre-trained model system)
        # In the pre-trained model system, we don't retrain during runtime
        
        logger.debug(f"Updated location for device {device_id}")
    
    async def predict_mobility(self, device_id: str) -> List[MobilityPrediction]:
        """Enhanced mobility prediction using pre-trained models."""
        if device_id not in self.mobility_history:
            return []
        
        history = self.mobility_history[device_id]
        if len(history) < self.min_prediction_points:
            return []
        
        current_location = history[-1]
        predictions = []
        
        for horizon in self.prediction_horizons:
            try:
                # Use pre-trained models if available
                if self.models_loaded:
                    prediction = await self._predict_with_pretrained_models(
                        device_id, current_location, horizon
                    )
                    if prediction:
                        predictions.append(prediction)
                else:
                    # Fallback to simple heuristic prediction
                    prediction = self._predict_with_heuristics(
                        device_id, current_location, horizon
                    )
                    if prediction:
                        predictions.append(prediction)
                        
            except Exception as e:
                logger.error(f"Error predicting mobility for device {device_id}: {e}")
        
        # Publish prediction event
        if predictions and self.event_bus:
            await self.event_bus.publish_event(
                EventType.MOBILITY_PREDICTED,
                correlation_id=f"prediction_{device_id}",
                source="MobilityPredictor",
                target=device_id,
                data={
                    "device_id": device_id,
                    "predictions": [
                        {
                            "horizon_seconds": p.time_horizon,
                            "predicted_location": {
                                "latitude": p.predicted_location.latitude,
                                "longitude": p.predicted_location.longitude,
                                "timestamp": p.predicted_location.timestamp
                            },
                            "confidence": p.confidence,
                            "mobility_pattern": p.mobility_pattern.name,
                            "handoff_probability": p.handoff_probability
                        } for p in predictions
                    ]
                }
            )
        
        return predictions
    
    async def _predict_with_pretrained_models(self, device_id: str, current_location: LocationPoint, 
                                            horizon: float) -> Optional[MobilityPrediction]:
        """Make prediction using pre-trained models on real data."""
        try:
            history = self.mobility_history[device_id]
            
            # Extract features for prediction (same format as training)
            features = self._extract_prediction_features(history, current_location)
            if not features:
                return None
            
            # Scale features
            features_scaled = self.mobility_scaler.transform([features])
            
            # Predict location using pre-trained mobility model
            if self.mobility_model:
                model_metadata = self.model_trainer.get_metadata("mobility_prediction_lstm")
                
                if model_metadata and model_metadata.model_type == "mobility_prediction_lstm":
                    # Check if using TensorFlow LSTM or MLPRegressor simulation
                    if model_metadata.model_params.get("tensorflow_available", False):
                        # TensorFlow LSTM model prediction
                        lstm_sequence = self._prepare_lstm_sequence(history, current_location)
                        if lstm_sequence is not None:
                            prediction = self.mobility_model.predict(lstm_sequence, verbose=0)[0]
                            lat_pred, lon_pred = prediction[0], prediction[1]
                        else:
                            return None
                    else:
                        # MLPRegressor simulation prediction
                        lstm_features = self._prepare_neural_network_features(history, current_location)
                        if lstm_features is not None:
                            features_scaled = self.mobility_scaler.transform([lstm_features])
                            prediction = self.mobility_model.predict(features_scaled)[0]
                            lat_pred, lon_pred = prediction[0], prediction[1]
                        else:
                            return None
                else:
                    # Traditional Random Forest model prediction
                    if isinstance(self.mobility_model, dict):
                        lat_pred = self.mobility_model['latitude_model'].predict(features_scaled)[0]
                        lon_pred = self.mobility_model['longitude_model'].predict(features_scaled)[0]
                    else:
                        # Single model predicting both lat/lon
                        prediction = self.mobility_model.predict(features_scaled)[0]
                        lat_pred, lon_pred = prediction[0], prediction[1]
                
                predicted_location = LocationPoint(
                    latitude=lat_pred,
                    longitude=lon_pred,
                    timestamp=time.time() + horizon
                )
            else:
                return None
            
            # Classify mobility pattern using pre-trained pattern model
            mobility_pattern = MobilityPattern.RANDOM  # Default
            confidence = 0.5  # Default confidence
            
            if self.pattern_model and self.pattern_scaler:
                # Extract pattern features
                pattern_features = self._extract_pattern_features_single(history)
                if pattern_features:
                    pattern_features_scaled = self.pattern_scaler.transform([pattern_features])
                    pattern_pred = self.pattern_model.predict(pattern_features_scaled)[0]
                    
                    if self.pattern_encoder:
                        pattern_name = self.pattern_encoder.inverse_transform([int(pattern_pred)])[0]
                        try:
                            mobility_pattern = MobilityPattern[pattern_name]
                        except KeyError:
                            mobility_pattern = MobilityPattern.RANDOM
                    
                    # Get confidence from model performance
                    if "pattern_classification" in self.model_performance:
                        confidence = self.model_performance["pattern_classification"].get("accuracy", 0.5)
            
            # Calculate handoff probability
            handoff_prob = self._calculate_handoff_probability(current_location, predicted_location)
            
            # Create prediction
            prediction = MobilityPrediction(
                device_id=device_id,
                predicted_location=predicted_location,
                confidence=confidence,
                time_horizon=horizon,
                mobility_pattern=mobility_pattern,
                handoff_probability=handoff_prob
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in pre-trained model prediction: {e}")
            return None
    
    def _predict_with_heuristics(self, device_id: str, current_location: LocationPoint, 
                               horizon: float) -> Optional[MobilityPrediction]:
        """Fallback prediction using simple heuristics when models not available."""
        history = self.mobility_history[device_id]
        
        if len(history) < 2:
            return None
        
        # Simple linear extrapolation
        prev_location = history[-2]
        time_diff = current_location.timestamp - prev_location.timestamp
        
        if time_diff <= 0:
            return None
        
        # Calculate velocity
        lat_velocity = (current_location.latitude - prev_location.latitude) / time_diff
        lon_velocity = (current_location.longitude - prev_location.longitude) / time_diff
        
        # Extrapolate position
        predicted_lat = current_location.latitude + lat_velocity * horizon
        predicted_lon = current_location.longitude + lon_velocity * horizon
        
        predicted_location = LocationPoint(
            latitude=predicted_lat,
            longitude=predicted_lon,
            timestamp=time.time() + horizon
        )
        
        # Simple pattern classification based on speed
        distance = self._calculate_distance(
            prev_location.latitude, prev_location.longitude,
            current_location.latitude, current_location.longitude
        )
        speed_ms = distance / time_diff if time_diff > 0 else 0
        speed_kmh = speed_ms * 3.6
        
        if speed_kmh < 1:
            mobility_pattern = MobilityPattern.STATIONARY
        elif speed_kmh > 25:
            mobility_pattern = MobilityPattern.VEHICLE
        else:
            mobility_pattern = MobilityPattern.RANDOM
        
        prediction = MobilityPrediction(
            device_id=device_id,
            predicted_location=predicted_location,
            confidence=0.3,  # Low confidence for heuristic prediction
            time_horizon=horizon,
            mobility_pattern=mobility_pattern,
            handoff_probability=0.1  # Conservative estimate
        )
        
        return prediction
    
    def _extract_prediction_features(self, history: List[LocationPoint], 
                                   current_location: LocationPoint) -> Optional[List[float]]:
        """Extract features for prediction in same format as training data."""
        if len(history) < 2:
            return None
        
        try:
            features = []
            
            # Current location
            features.extend([current_location.latitude, current_location.longitude])
            
            # Velocity features
            if len(history) >= 2:
                prev_point = history[-2]
                time_diff = current_location.timestamp - prev_point.timestamp
                if time_diff > 0:
                    lat_velocity = (current_location.latitude - prev_point.latitude) / time_diff
                    lon_velocity = (current_location.longitude - prev_point.longitude) / time_diff
                    features.extend([lat_velocity, lon_velocity])
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
            
            # Acceleration features
            if len(history) >= 3:
                prev_prev_point = history[-3]
                prev_point = history[-2]
                time_diff1 = current_location.timestamp - prev_point.timestamp
                time_diff2 = prev_point.timestamp - prev_prev_point.timestamp
                
                if time_diff1 > 0 and time_diff2 > 0:
                    lat_acc = ((current_location.latitude - prev_point.latitude) / time_diff1 - 
                              (prev_point.latitude - prev_prev_point.latitude) / time_diff2) / time_diff1
                    lon_acc = ((current_location.longitude - prev_point.longitude) / time_diff1 - 
                              (prev_point.longitude - prev_prev_point.longitude) / time_diff2) / time_diff1
                    features.extend([lat_acc, lon_acc])
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
            
            # Time-based features
            current_time = datetime.fromtimestamp(current_location.timestamp)
            hour_of_day = current_time.hour / 24.0
            day_of_week = current_time.weekday() / 7.0
            features.extend([hour_of_day, day_of_week])
            
            # Statistical features over recent history (last 6 points)
            recent_points = history[-6:] if len(history) >= 6 else history
            lats = [p.latitude for p in recent_points]
            lons = [p.longitude for p in recent_points]
            
            lat_mean, lat_std = np.mean(lats), np.std(lats)
            lon_mean, lon_std = np.mean(lons), np.std(lons)
            
            features.extend([lat_mean, lat_std, lon_mean, lon_std])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting prediction features: {e}")
            return None
    
    def _extract_pattern_features_single(self, history: List[LocationPoint]) -> Optional[List[float]]:
        """Extract pattern classification features for a single trajectory."""
        if len(history) < 10:
            return None
        
        try:
            features = []
            
            # Calculate trajectory statistics
            start_time = history[0].timestamp
            end_time = history[-1].timestamp
            duration_hours = (end_time - start_time) / 3600
            
            # Calculate total distance
            total_distance = 0.0
            for i in range(1, len(history)):
                p1, p2 = history[i-1], history[i]
                distance = self._calculate_distance(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
                total_distance += distance / 1000  # Convert to km
            
            avg_speed = (total_distance / duration_hours) if duration_hours > 0 else 0
            
            features.extend([total_distance, avg_speed, duration_hours, len(history)])
            
            # Spatial features
            lats = [p.latitude for p in history]
            lons = [p.longitude for p in history]
            
            lat_range = max(lats) - min(lats)
            lon_range = max(lons) - min(lons)
            lat_std = np.std(lats)
            lon_std = np.std(lons)
            
            features.extend([lat_range, lon_range, lat_std, lon_std])
            
            # Temporal features
            start_datetime = datetime.fromtimestamp(start_time)
            start_hour = start_datetime.hour / 24.0
            start_day = start_datetime.weekday() / 7.0
            
            features.extend([start_hour, start_day])
            
            # Speed variation
            speeds = []
            for i in range(1, len(history)):
                p1, p2 = history[i-1], history[i]
                time_diff = p2.timestamp - p1.timestamp
                if time_diff > 0:
                    distance = self._calculate_distance(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
                    speed = (distance / 1000) / (time_diff / 3600)  # km/h
                    speeds.append(speed)
            
            if speeds:
                speed_mean = np.mean(speeds)
                speed_std = np.std(speeds)
                speed_max = np.max(speeds)
            else:
                speed_mean = speed_std = speed_max = 0.0
            
            features.extend([speed_mean, speed_std, speed_max])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting pattern features: {e}")
            return None
    
    def _prepare_lstm_sequence(self, history: List[LocationPoint], current_location: LocationPoint) -> Optional[np.ndarray]:
        """Prepare LSTM input sequence for prediction."""
        try:
            # Get sequence length from model metadata
            model_metadata = self.model_trainer.get_metadata("mobility_prediction_lstm")
            if not model_metadata:
                return None
            
            sequence_length = model_metadata.model_params.get("sequence_length", 20)
            
            # Need enough history for sequence
            if len(history) < sequence_length:
                return None
            
            # Take the last sequence_length points
            sequence_points = history[-sequence_length:]
            
            # Extract features for each point in sequence
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
            
            # Convert to numpy array and reshape for LSTM (1, timesteps, features)
            sequence_array = np.array(sequence_features).reshape(1, sequence_length, -1)
            
            # Apply the same scaling as during training
            if self.mobility_scaler:
                # Reshape for scaling
                n_samples, n_timesteps, n_features = sequence_array.shape
                sequence_reshaped = sequence_array.reshape((n_samples * n_timesteps, n_features))
                
                # Scale
                sequence_scaled = self.mobility_scaler.transform(sequence_reshaped)
                
                # Reshape back
                sequence_array = sequence_scaled.reshape((n_samples, n_timesteps, n_features))
            
            return sequence_array
            
        except Exception as e:
            logger.error(f"Error preparing LSTM sequence: {e}")
            return None
    
    def _prepare_neural_network_features(self, history: List[LocationPoint], current_location: LocationPoint) -> Optional[List[float]]:
        """Prepare flattened features for MLPRegressor neural network simulation."""
        try:
            # Get sequence length from model metadata
            model_metadata = self.model_trainer.get_metadata("mobility_prediction_lstm")
            if not model_metadata:
                return None
            
            sequence_length = model_metadata.model_params.get("sequence_length", 15)
            
            # Need enough history for sequence
            if len(history) < sequence_length:
                return None
            
            # Take the last sequence_length points
            sequence_points = history[-sequence_length:]
            
            # Extract features for each point in sequence and flatten
            all_features = []
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
                
                all_features.extend(features)  # Flatten into single array
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error preparing neural network features: {e}")
            return None
    
    async def retrain_models_if_needed(self) -> None:
        """Check if models need retraining and trigger if necessary."""
        if not self.model_trainer:
            logger.warning("No model trainer available for retraining")
            return
            
        # This would typically check if we have enough new data to warrant retraining
        # For now, we rely on the model trainer's existing models
        logger.debug("Model retraining check - using existing pre-trained models")
    
    def _extract_features(self, device_id: str, location: LocationPoint) -> Optional[MobilityFeatures]:
        """Extract features from location history for ML."""
        history = self.mobility_history.get(device_id, [])
        if len(history) < 2:
            return None
        
        try:
            current_time = location.timestamp
            dt = time.localtime(current_time)
            
            # Calculate movement features
            prev_location = history[-2]
            distance = self._calculate_distance(
                prev_location.latitude, prev_location.longitude,
                location.latitude, location.longitude
            )
            
            time_diff = location.timestamp - prev_location.timestamp
            speed = distance / time_diff if time_diff > 0 else 0.0
            
            # Calculate acceleration
            acceleration = 0.0
            if len(history) >= 3:
                prev_prev = history[-3]
                prev_speed = self._calculate_distance(
                    prev_prev.latitude, prev_prev.longitude,
                    prev_location.latitude, prev_location.longitude
                ) / (prev_location.timestamp - prev_prev.timestamp)
                acceleration = (speed - prev_speed) / time_diff if time_diff > 0 else 0.0
            
            # Calculate direction change
            direction_change = 0.0
            if len(history) >= 3 and prev_location.bearing and location.bearing:
                direction_change = abs(location.bearing - prev_location.bearing)
                if direction_change > 180:
                    direction_change = 360 - direction_change
            
            # Historical features
            recent_history = [h for h in history if current_time - h.timestamp <= 3600]  # Last hour
            avg_speed_last_hour = np.mean([
                self._calculate_distance(h1.latitude, h1.longitude, h2.latitude, h2.longitude) /
                max(h2.timestamp - h1.timestamp, 1.0)
                for h1, h2 in zip(recent_history[:-1], recent_history[1:])
            ]) if len(recent_history) > 1 else 0.0
            
            # Daily movement
            today_history = [h for h in history if 
                            time.gmtime(current_time).tm_yday == time.gmtime(h.timestamp).tm_yday]
            distance_today = sum([
                self._calculate_distance(h1.latitude, h1.longitude, h2.latitude, h2.longitude)
                for h1, h2 in zip(today_history[:-1], today_history[1:])
            ])
            
            # Area visits (simplified)
            current_area_visits = len([h for h in history if 
                                     self._calculate_distance(h.latitude, h.longitude,
                                                            location.latitude, location.longitude) < 100])
            
            # Time since last significant movement
            time_since_movement = 0.0
            for h in reversed(history[:-1]):
                if self._calculate_distance(h.latitude, h.longitude,
                                          location.latitude, location.longitude) > 50:
                    time_since_movement = current_time - h.timestamp
                    break
            
            return MobilityFeatures(
                hour_of_day=dt.tm_hour,
                day_of_week=dt.tm_wday,
                is_weekend=dt.tm_wday >= 5,
                distance_from_last=distance,
                speed=speed,
                acceleration=acceleration,
                direction_change=direction_change,
                avg_speed_last_hour=avg_speed_last_hour,
                distance_traveled_today=distance_today,
                visits_to_current_area=current_area_visits,
                time_since_last_movement=time_since_movement
            )
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _features_to_vector(self, features: MobilityFeatures, horizon: float) -> List[float]:
        """Convert mobility features to ML feature vector."""
        return [
            features.hour_of_day,
            features.day_of_week,
            float(features.is_weekend),
            features.distance_from_last,
            features.speed,
            features.acceleration,
            features.direction_change,
            features.avg_speed_last_hour,
            features.distance_traveled_today,
            features.visits_to_current_area,
            features.time_since_last_movement,
            float(features.is_charging),
            features.signal_strength,
            features.battery_level,
            horizon  # Time horizon as feature
        ]
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get performance metrics of loaded models."""
        if not self.models_loaded or not self.model_performance:
            return {"status": "no_models_loaded"}
        
        return self.model_performance.copy()
    
    def _classify_pattern(self, pattern_prob: float) -> MobilityPattern:
        """Classify mobility pattern from probability."""
        if pattern_prob < 0.2:
            return MobilityPattern.STATIONARY
        elif pattern_prob < 0.4:
            return MobilityPattern.PERIODIC
        elif pattern_prob < 0.6:
            return MobilityPattern.COMMUTER
        elif pattern_prob < 0.8:
            return MobilityPattern.VEHICLE
        else:
            return MobilityPattern.RANDOM
    
    def _calculate_handoff_probability(self, current: LocationPoint, predicted: LocationPoint) -> float:
        """Calculate probability of gateway handoff."""
        distance = self._calculate_distance(
            current.latitude, current.longitude,
            predicted.latitude, predicted.longitude
        )
        
        # Simple heuristic: higher distance = higher handoff probability
        # In real implementation, this would consider gateway coverage areas
        return min(distance / 1000.0, 1.0)  # Normalize by 1km
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        try:
            # Handle edge cases
            if lat1 == lat2 and lon1 == lon2:
                return 0.0
            
            # Validate coordinates
            if abs(lat1) > 90 or abs(lat2) > 90 or abs(lon1) > 180 or abs(lon2) > 180:
                logger.warning(f"Invalid coordinates: ({lat1}, {lon1}) to ({lat2}, {lon2})")
                return 0.0
            
            # Convert to radians
            lat1_rad = np.radians(lat1)
            lon1_rad = np.radians(lon1)
            lat2_rad = np.radians(lat2)
            lon2_rad = np.radians(lon2)
            
            # Haversine formula
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = (np.sin(dlat/2)**2 + 
                 np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
            
            # Ensure a is within valid range [0, 1]
            a = max(0.0, min(1.0, a))
            
            c = 2 * np.arcsin(np.sqrt(a))
            
            # Earth radius in meters
            r = 6371000
            
            distance = r * c
            
            # Sanity check - if distance is unreasonably large, something is wrong
            if distance > 20003931:  # Half of Earth's circumference
                logger.warning(f"Unreasonable distance calculated: {distance}m")
                return 0.0
            
            return distance
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0
    
    def is_ready_for_prediction(self) -> bool:
        """Check if predictor is ready to make predictions."""
        return self.models_loaded and self.model_trainer is not None
    
    def _prediction_to_dict(self, prediction: MobilityPrediction) -> Dict[str, Any]:
        """Convert prediction to dictionary for serialization."""
        return {
            "device_id": prediction.device_id,
            "predicted_location": {
                "latitude": prediction.predicted_location.latitude,
                "longitude": prediction.predicted_location.longitude,
                "timestamp": prediction.predicted_location.timestamp
            },
            "confidence": prediction.confidence,
            "time_horizon": prediction.time_horizon,
            "mobility_pattern": prediction.mobility_pattern.name,
            "next_gateway": prediction.next_gateway,
            "handoff_probability": prediction.handoff_probability
        }
    
    def get_device_stats(self, device_id: str) -> Dict[str, Any]:
        """Get mobility statistics for a device."""
        if device_id not in self.mobility_history:
            return {}
        
        history = self.mobility_history[device_id]
        features = self.feature_history.get(device_id, [])
        
        if not history:
            return {}
        
        # Calculate basic stats
        recent_locations = history[-100:]  # Last 100 points
        distances = [
            self._calculate_distance(l1.latitude, l1.longitude, l2.latitude, l2.longitude)
            for l1, l2 in zip(recent_locations[:-1], recent_locations[1:])
        ]
        
        speeds = [
            d / max(l2.timestamp - l1.timestamp, 1.0)
            for d, l1, l2 in zip(distances, recent_locations[:-1], recent_locations[1:])
        ]
        
        return {
            "device_id": device_id,
            "total_locations": len(history),
            "total_features": len(features),
            "avg_speed": np.mean(speeds) if speeds else 0.0,
            "max_speed": np.max(speeds) if speeds else 0.0,
            "total_distance": np.sum(distances) if distances else 0.0,
            "first_seen": history[0].timestamp if history else 0.0,
            "last_seen": history[-1].timestamp if history else 0.0,
            "models_loaded": self.models_loaded,
            "model_performance": self.model_performance
        }

    # Validation-compatible method aliases
    def predict_next_location(self, device_id: str, current_time: Optional[float] = None) -> Optional[LocationPoint]:
        """Alias for predict_mobility for validation compatibility."""
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we're in an event loop, we need to handle this differently
            # For now, return None as we can't easily await in a sync method
            logger.warning("predict_next_location called from within event loop - use predict_mobility directly for async contexts")
            return None
        except RuntimeError:
            # No event loop is running, safe to use asyncio.run()
            predictions = asyncio.run(self.predict_mobility(device_id))
            if predictions:
                # Return the shortest time horizon prediction
                return min(predictions, key=lambda p: p.time_horizon).predicted_location
            return None
    
    async def update_model(self, device_id: str, new_location: LocationPoint) -> None:
        """Async alias for update_location for validation compatibility."""
        await self.update_location(device_id, new_location)
