"""
Lightweight Mobility Predictor for ZKPAS

This module provides a simplified, reliable mobility prediction system that works
without complex dependencies and provides accurate results.
"""

import asyncio
import numpy as np
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class SimpleLocationPoint:
    """Simple location point representation."""
    latitude: float
    longitude: float
    timestamp: float
    
@dataclass
class SimpleMobilityPrediction:
    """Simple mobility prediction result."""
    device_id: str
    predicted_lat: float
    predicted_lon: float
    confidence: float
    time_horizon: float
    error_estimate: float  # Expected error in meters

class LightweightMobilityPredictor:
    """Lightweight mobility predictor with reliable results."""
    
    def __init__(self):
        """Initialize the lightweight predictor."""
        self.device_history: Dict[str, List[SimpleLocationPoint]] = {}
        self.max_history = 100
        self.min_points_for_prediction = 3
        
        # Performance tracking
        self.prediction_errors = []
        self.prediction_count = 0
        
        logger.info("Lightweight mobility predictor initialized")
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula (meters)."""
        try:
            # Handle edge cases
            if lat1 == lat2 and lon1 == lon2:
                return 0.0
            
            # Validate coordinates
            if abs(lat1) > 90 or abs(lat2) > 90 or abs(lon1) > 180 or abs(lon2) > 180:
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
            
            # Ensure a is within valid range
            a = max(0.0, min(1.0, a))
            c = 2 * np.arcsin(np.sqrt(a))
            
            # Earth radius in meters
            distance = 6371000 * c
            
            return distance
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0
    
    def add_location(self, device_id: str, lat: float, lon: float, timestamp: Optional[float] = None):
        """Add a location point for a device."""
        if timestamp is None:
            timestamp = time.time()
        
        location = SimpleLocationPoint(lat, lon, timestamp)
        
        if device_id not in self.device_history:
            self.device_history[device_id] = []
        
        self.device_history[device_id].append(location)
        
        # Maintain history limit
        if len(self.device_history[device_id]) > self.max_history:
            self.device_history[device_id] = self.device_history[device_id][-self.max_history:]
    
    def predict_next_location(self, device_id: str, time_horizon: float = 60.0) -> Optional[SimpleMobilityPrediction]:
        """Predict next location for a device."""
        if device_id not in self.device_history:
            return None
        
        history = self.device_history[device_id]
        if len(history) < self.min_points_for_prediction:
            return None
        
        try:
            # Get recent points for prediction
            recent_points = history[-10:]  # Use last 10 points
            
            # Calculate velocity (lat/lon per second)
            velocities = []
            for i in range(1, len(recent_points)):
                p1, p2 = recent_points[i-1], recent_points[i]
                time_diff = p2.timestamp - p1.timestamp
                
                if time_diff > 0:
                    lat_velocity = (p2.latitude - p1.latitude) / time_diff
                    lon_velocity = (p2.longitude - p1.longitude) / time_diff
                    velocities.append((lat_velocity, lon_velocity))
            
            if not velocities:
                return None
            
            # Calculate average velocity
            avg_lat_vel = np.mean([v[0] for v in velocities])
            avg_lon_vel = np.mean([v[1] for v in velocities])
            
            # Add some momentum-based prediction
            recent_velocities = velocities[-3:]  # Last 3 velocities
            if len(recent_velocities) >= 2:
                # Apply exponential smoothing
                alpha = 0.3
                smooth_lat_vel = recent_velocities[-1][0]
                smooth_lon_vel = recent_velocities[-1][1]
                
                for i in range(len(recent_velocities) - 2, -1, -1):
                    smooth_lat_vel = alpha * recent_velocities[i][0] + (1 - alpha) * smooth_lat_vel
                    smooth_lon_vel = alpha * recent_velocities[i][1] + (1 - alpha) * smooth_lon_vel
                
                avg_lat_vel = smooth_lat_vel
                avg_lon_vel = smooth_lon_vel
            
            # Predict location after time_horizon seconds
            current = history[-1]
            predicted_lat = current.latitude + (avg_lat_vel * time_horizon)
            predicted_lon = current.longitude + (avg_lon_vel * time_horizon)
            
            # Calculate confidence based on velocity consistency
            if len(velocities) >= 3:
                lat_velocities = [v[0] for v in velocities]
                lon_velocities = [v[1] for v in velocities]
                lat_std = np.std(lat_velocities)
                lon_std = np.std(lon_velocities)
                
                # Lower std deviation = higher confidence
                velocity_consistency = 1.0 / (1.0 + lat_std + lon_std)
                confidence = min(max(velocity_consistency, 0.1), 0.95)
            else:
                confidence = 0.5
            
            # Estimate prediction error based on historical accuracy and time horizon
            base_error = 50.0  # Base error in meters
            time_factor = np.sqrt(time_horizon / 60.0)  # Error increases with time
            velocity_factor = np.sqrt(avg_lat_vel**2 + avg_lon_vel**2) * 100000  # Convert to roughly meters/s
            
            error_estimate = base_error * time_factor * (1 + velocity_factor)
            error_estimate = min(error_estimate, 1000.0)  # Cap at 1km
            
            prediction = SimpleMobilityPrediction(
                device_id=device_id,
                predicted_lat=predicted_lat,
                predicted_lon=predicted_lon,
                confidence=confidence,
                time_horizon=time_horizon,
                error_estimate=error_estimate
            )
            
            self.prediction_count += 1
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting location for device {device_id}: {e}")
            return None
    
    def validate_prediction(self, prediction: SimpleMobilityPrediction, actual_lat: float, actual_lon: float):
        """Validate a prediction against actual location."""
        error = self.calculate_distance(
            prediction.predicted_lat, prediction.predicted_lon,
            actual_lat, actual_lon
        )
        
        self.prediction_errors.append(error)
        
        # Keep only recent errors
        if len(self.prediction_errors) > 100:
            self.prediction_errors = self.prediction_errors[-100:]
        
        return error
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.prediction_errors:
            return {
                "predictions_made": self.prediction_count,
                "avg_error_m": 0.0,
                "avg_error_km": 0.0,
                "accuracy_100m": 0.0,
                "accuracy_500m": 0.0,
                "min_error_m": 0.0,
                "max_error_m": 0.0
            }
        
        errors = np.array(self.prediction_errors)
        
        # Calculate accuracy as percentage within thresholds
        accuracy_100m = np.sum(errors <= 100.0) / len(errors)
        accuracy_500m = np.sum(errors <= 500.0) / len(errors)
        
        metrics = {
            "predictions_made": self.prediction_count,
            "avg_error_m": float(np.mean(errors)),
            "avg_error_km": float(np.mean(errors) / 1000.0),
            "accuracy_100m": float(accuracy_100m),
            "accuracy_500m": float(accuracy_500m),
            "min_error_m": float(np.min(errors)),
            "max_error_m": float(np.max(errors))
        }
        
        return metrics
    
    def generate_synthetic_trajectory(self, device_id: str, duration_minutes: int = 30, 
                                    pattern: str = "random_walk") -> List[SimpleLocationPoint]:
        """Generate a synthetic trajectory for testing."""
        trajectory = []
        
        # Starting location (somewhere in NYC for example)
        start_lat = 40.7589 + random.uniform(-0.1, 0.1)
        start_lon = -73.9851 + random.uniform(-0.1, 0.1)
        start_time = time.time()
        
        current_lat = start_lat
        current_lon = start_lon
        current_time = start_time
        
        # Generate points every 30 seconds
        interval = 30.0
        num_points = int(duration_minutes * 60 / interval)
        
        for i in range(num_points):
            trajectory.append(SimpleLocationPoint(current_lat, current_lon, current_time))
            
            if pattern == "random_walk":
                # Random walk with slight persistence
                lat_change = random.uniform(-0.001, 0.001)
                lon_change = random.uniform(-0.001, 0.001)
                
                current_lat += lat_change
                current_lon += lon_change
                
            elif pattern == "linear":
                # Linear movement
                current_lat += 0.0005  # Moving north
                current_lon += 0.0002  # Moving east
                
            elif pattern == "circular":
                # Circular movement
                angle = (i / num_points) * 2 * np.pi
                radius = 0.01
                current_lat = start_lat + radius * np.cos(angle)
                current_lon = start_lon + radius * np.sin(angle)
            
            current_time += interval
        
        # Add trajectory to device history
        self.device_history[device_id] = trajectory
        
        return trajectory

# Simulation function for testing
async def run_lightweight_demo():
    """Run a demonstration of the lightweight predictor."""
    print("🧠 Lightweight LSTM-Alternative Mobility Prediction Demo")
    print("=" * 60)
    
    predictor = LightweightMobilityPredictor()
    
    # Generate synthetic trajectories for testing
    print("📊 Generating synthetic mobility data...")
    
    devices = ["device_001", "device_002", "device_003"]
    patterns = ["random_walk", "linear", "circular"]
    
    for device, pattern in zip(devices, patterns):
        trajectory = predictor.generate_synthetic_trajectory(device, duration_minutes=20, pattern=pattern)
        print(f"   Generated {len(trajectory)} points for {device} ({pattern} pattern)")
    
    # Test predictions and validation
    print("\n🔮 Testing mobility predictions...")
    
    all_errors = []
    for device in devices:
        print(f"\n   Testing {device}:")
        
        # Use first 80% of trajectory for history, predict the rest
        history = predictor.device_history[device]
        split_point = int(len(history) * 0.8)
        
        # Reset device history to training portion
        predictor.device_history[device] = history[:split_point]
        
        # Make predictions and validate
        test_points = history[split_point:]
        device_errors = []
        
        for i, actual_point in enumerate(test_points[:5]):  # Test first 5 points
            # Predict location
            prediction = predictor.predict_next_location(device, time_horizon=60.0)
            
            if prediction:
                # Validate against actual location
                error = predictor.validate_prediction(
                    prediction, actual_point.latitude, actual_point.longitude
                )
                device_errors.append(error)
                all_errors.append(error)
                
                print(f"     Prediction {i+1}: {error:.1f}m error (confidence: {prediction.confidence:.3f})")
                
                # Add actual point to history for next prediction
                predictor.add_location(device, actual_point.latitude, actual_point.longitude, actual_point.timestamp)
            else:
                print(f"     Prediction {i+1}: Failed to generate prediction")
        
        if device_errors:
            avg_error = np.mean(device_errors)
            print(f"     Average error: {avg_error:.1f}m")
    
    # Overall performance
    print(f"\n📈 Overall Performance:")
    if all_errors:
        metrics = predictor.get_performance_metrics()
        print(f"   Predictions made: {metrics['predictions_made']}")
        print(f"   Average error: {metrics['avg_error_m']:.1f}m ({metrics['avg_error_km']:.3f}km)")
        print(f"   Accuracy (±100m): {metrics['accuracy_100m']:.3f} ({metrics['accuracy_100m']*100:.1f}%)")
        print(f"   Accuracy (±500m): {metrics['accuracy_500m']:.3f} ({metrics['accuracy_500m']*100:.1f}%)")
        print(f"   Error range: {metrics['min_error_m']:.1f}m - {metrics['max_error_m']:.1f}m")
    else:
        print("   No predictions made")
    
    print("\n✅ Lightweight prediction demo completed!")
    
    return metrics if all_errors else {}

if __name__ == "__main__":
    asyncio.run(run_lightweight_demo())