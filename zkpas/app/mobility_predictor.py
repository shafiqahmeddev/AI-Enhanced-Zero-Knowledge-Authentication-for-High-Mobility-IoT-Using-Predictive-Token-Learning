"""
Mobility Prediction Module for ZKPAS IoT Devices

This module implements machine learning-based mobility prediction for IoT devices
to enable proactive authentication and seamless handoffs between gateways.
"""

import asyncio
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
from enum import Enum, auto

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
    """ML-based mobility predictor for IoT devices."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        
        # ML models
        self.location_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.pattern_classifier = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        
        # Training data storage
        self.mobility_history: Dict[str, List[LocationPoint]] = {}
        self.feature_history: Dict[str, List[MobilityFeatures]] = {}
        
        # Model metadata
        self.is_trained = False
        self.last_training_time = 0.0
        self.model_accuracy = 0.0
        
        # Configuration
        self.max_history_points = 10000
        self.min_training_samples = 100
        self.retrain_interval = 3600  # 1 hour
        self.prediction_horizons = [60, 300, 900]  # 1min, 5min, 15min
        
        # Setup event handlers
        self._setup_event_handlers()
        
        logger.info("Mobility predictor initialized")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for mobility-related events."""
        self.event_bus.subscribe(EventType.LOCATION_CHANGED, self._handle_location_update)
        self.event_bus.subscribe(EventType.MOBILITY_PREDICTED, self._handle_prediction_feedback)
    
    async def _handle_location_update(self, event: Event) -> None:
        """Handle location update event."""
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
        
        # Check if we need to retrain models
        if self._should_retrain():
            await self.train_models()
        
        logger.debug(f"Updated location for device {device_id}")
    
    async def predict_mobility(self, device_id: str) -> List[MobilityPrediction]:
        """Predict future mobility for a device."""
        if not self.is_trained or device_id not in self.mobility_history:
            return []
        
        current_location = self.mobility_history[device_id][-1]
        predictions = []
        
        for horizon in self.prediction_horizons:
            try:
                # Extract current features
                features = self._extract_features(device_id, current_location)
                if not features:
                    continue
                
                # Convert features to ML input
                feature_vector = self._features_to_vector(features, horizon)
                feature_vector_scaled = self.scaler.transform([feature_vector])
                
                # Predict location
                predicted_coords = self.location_model.predict(feature_vector_scaled)[0]
                predicted_lat, predicted_lon = predicted_coords[0], predicted_coords[1]
                
                # Predict mobility pattern
                pattern_prob = self.pattern_classifier.predict(feature_vector_scaled)[0]
                mobility_pattern = self._classify_pattern(pattern_prob)
                
                # Create prediction
                predicted_location = LocationPoint(
                    latitude=predicted_lat,
                    longitude=predicted_lon,
                    timestamp=time.time() + horizon
                )
                
                prediction = MobilityPrediction(
                    device_id=device_id,
                    predicted_location=predicted_location,
                    confidence=min(self.model_accuracy, 0.95),
                    time_horizon=horizon,
                    mobility_pattern=mobility_pattern,
                    handoff_probability=self._calculate_handoff_probability(
                        current_location, predicted_location
                    )
                )
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error predicting mobility for device {device_id}: {e}")
        
        return predictions
    
    async def train_models(self) -> None:
        """Train mobility prediction models."""
        logger.info("Training mobility prediction models")
        
        try:
            # Collect training data from all devices
            X, y_location, y_pattern = self._prepare_training_data()
            
            if len(X) < self.min_training_samples:
                logger.warning(f"Not enough training samples: {len(X)} < {self.min_training_samples}")
                return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train location prediction model
            self.location_model.fit(X_scaled, y_location)
            
            # Train pattern classification model
            self.pattern_classifier.fit(X_scaled, y_pattern)
            
            # Evaluate model performance
            self.model_accuracy = self._evaluate_models(X_scaled, y_location)
            
            self.is_trained = True
            self.last_training_time = time.time()
            
            logger.info(f"Models trained successfully. Accuracy: {self.model_accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
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
    
    def _prepare_training_data(self) -> Tuple[List[List[float]], List[List[float]], List[float]]:
        """Prepare training data from feature history."""
        X = []
        y_location = []
        y_pattern = []
        
        for device_id, features_list in self.feature_history.items():
            if device_id not in self.mobility_history:
                continue
            
            locations = self.mobility_history[device_id]
            
            for i, features in enumerate(features_list[:-1]):
                # Find corresponding future location
                if i + 1 < len(locations):
                    future_location = locations[i + 1]
                    
                    for horizon in self.prediction_horizons:
                        feature_vector = self._features_to_vector(features, horizon)
                        X.append(feature_vector)
                        y_location.append([future_location.latitude, future_location.longitude])
                        
                        # Pattern classification (simplified)
                        pattern_value = float(features.speed > 5.0)  # High mobility pattern
                        y_pattern.append(pattern_value)
        
        return X, y_location, y_pattern
    
    def _evaluate_models(self, X: np.ndarray, y_location: List[List[float]]) -> float:
        """Evaluate model performance."""
        try:
            # Simple train-test split evaluation
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y_location[:split_idx], y_location[split_idx:]
            
            # Predict on test set
            predictions = self.location_model.predict(X_test)
            
            # Calculate mean absolute error in meters
            errors = []
            for pred, actual in zip(predictions, y_test):
                error = self._calculate_distance(pred[0], pred[1], actual[0], actual[1])
                errors.append(error)
            
            mae = np.mean(errors)
            accuracy = max(0.0, 1.0 - mae / 1000.0)  # Normalize by 1km
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            return 0.0
    
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
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            # Earth radius in meters
            r = 6371000
            
            return r * c
            
        except Exception:
            return 0.0
    
    def _should_retrain(self) -> bool:
        """Check if models should be retrained."""
        return (
            not self.is_trained or
            time.time() - self.last_training_time > self.retrain_interval or
            sum(len(features) for features in self.feature_history.values()) > self.max_history_points
        )
    
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
            "is_model_trained": self.is_trained,
            "model_accuracy": self.model_accuracy
        }

    # Validation-compatible method aliases
    def predict_next_location(self, device_id: str, current_time: Optional[float] = None) -> LocationPoint:
        """Alias for predict_location for validation compatibility."""
        return self.predict_location(device_id, current_time)
    
    def update_model(self, device_id: str, new_location: LocationPoint) -> None:
        """Alias for add_location for validation compatibility."""
        self.add_location(device_id, new_location)
