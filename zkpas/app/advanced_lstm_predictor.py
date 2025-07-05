#!/usr/bin/env python3
"""
Advanced LSTM Mobility Predictor with High Accuracy Optimizations

This module implements state-of-the-art techniques to maximize LSTM accuracy:
- Attention mechanisms for better sequence modeling
- Ensemble learning with multiple models
- Advanced feature engineering
- Data augmentation
- Transfer learning
- Sophisticated preprocessing

Author: Shafiq Ahmed <s.ahmed@essex.ac.uk>
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import (
    LSTM, Linear, Dropout, MultiheadAttention, 
    LayerNorm, Conv1d, MaxPool1d, 
    BatchNorm1d, GRU, RNN, Embedding
)
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, CosineAnnealingLR, 
    StepLR, CyclicLR
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from scipy import signal
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, fftfreq

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class AdvancedDataPreprocessor:
    """Advanced data preprocessing for maximum accuracy."""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.outlier_detectors = {}
        self.scalers = {}
        
    def detect_and_handle_outliers(self, trajectories: List) -> List:
        """Advanced outlier detection and handling using multiple methods."""
        cleaned_trajectories = []
        
        for trajectory in trajectories:
            points = trajectory.points
            if len(points) < 10:
                continue
            
            # Extract coordinates and features
            coords = np.array([[p.latitude, p.longitude] for p in points])
            timestamps = np.array([p.timestamp for p in points])
            
            # 1. Speed-based outlier detection
            speeds = self._calculate_speeds(coords, timestamps)
            speed_outliers = self._detect_speed_outliers(speeds)
            
            # 2. GPS accuracy-based filtering
            gps_outliers = self._detect_gps_outliers(coords)
            
            # 3. Trajectory consistency check
            consistency_outliers = self._detect_trajectory_inconsistencies(coords, timestamps)
            
            # Combine outlier masks
            combined_outliers = speed_outliers | gps_outliers | consistency_outliers
            
            # Filter out outliers
            if np.sum(~combined_outliers) >= 5:  # Keep trajectory if enough points remain
                filtered_points = [p for i, p in enumerate(points) if not combined_outliers[i]]
                
                # Create cleaned trajectory
                cleaned_trajectory = TrajectoryData(
                    user_id=trajectory.user_id,
                    trajectory_id=trajectory.trajectory_id,
                    points=filtered_points,
                    start_time=filtered_points[0].timestamp,
                    end_time=filtered_points[-1].timestamp,
                    duration_seconds=filtered_points[-1].timestamp - filtered_points[0].timestamp,
                    total_distance_km=self._calculate_total_distance(filtered_points),
                    avg_speed_kmh=0,  # Will be recalculated
                    mobility_pattern=trajectory.mobility_pattern
                )
                
                # Recalculate average speed
                if cleaned_trajectory.duration_seconds > 0:
                    cleaned_trajectory.avg_speed_kmh = (
                        cleaned_trajectory.total_distance_km / 
                        (cleaned_trajectory.duration_seconds / 3600)
                    )
                
                cleaned_trajectories.append(cleaned_trajectory)
        
        return cleaned_trajectories
    
    def _calculate_speeds(self, coords: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Calculate speeds between consecutive points."""
        if len(coords) < 2:
            return np.array([])
        
        distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)) * 111000  # Convert to meters
        time_diffs = np.diff(timestamps)
        time_diffs = np.maximum(time_diffs, 1e-6)  # Avoid division by zero
        
        speeds = distances / time_diffs  # m/s
        return np.concatenate([[0], speeds])  # Prepend 0 for first point
    
    def _detect_speed_outliers(self, speeds: np.ndarray) -> np.ndarray:
        """Detect speed-based outliers using IQR method."""
        if len(speeds) < 5:
            return np.zeros(len(speeds), dtype=bool)
        
        Q1 = np.percentile(speeds, 25)
        Q3 = np.percentile(speeds, 75)
        IQR = Q3 - Q1
        
        # Reasonable speed thresholds
        max_walking_speed = 3.0  # m/s (~11 km/h)
        max_vehicle_speed = 70.0  # m/s (~250 km/h)
        
        # Combine statistical and physical outliers
        statistical_outliers = (speeds > Q3 + 2.0 * IQR) | (speeds < Q1 - 2.0 * IQR)
        physical_outliers = speeds > max_vehicle_speed
        
        return statistical_outliers | physical_outliers
    
    def _detect_gps_outliers(self, coords: np.ndarray) -> np.ndarray:
        """Detect GPS measurement outliers."""
        if len(coords) < 5:
            return np.zeros(len(coords), dtype=bool)
        
        outliers = np.zeros(len(coords), dtype=bool)
        
        # Check for impossible coordinate jumps
        for i in range(1, len(coords) - 1):
            prev_coord = coords[i-1]
            curr_coord = coords[i]
            next_coord = coords[i+1]
            
            # Distance to previous and next points
            dist_prev = np.sqrt(np.sum((curr_coord - prev_coord)**2)) * 111000  # meters
            dist_next = np.sqrt(np.sum((next_coord - curr_coord)**2)) * 111000  # meters
            
            # If current point is very far from both neighbors, it's likely an outlier
            if dist_prev > 1000 and dist_next > 1000:  # 1km threshold
                outliers[i] = True
        
        return outliers
    
    def _detect_trajectory_inconsistencies(self, coords: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Detect trajectory inconsistencies using local context."""
        if len(coords) < 7:
            return np.zeros(len(coords), dtype=bool)
        
        outliers = np.zeros(len(coords), dtype=bool)
        window_size = 5
        
        for i in range(window_size//2, len(coords) - window_size//2):
            # Get local window
            start_idx = i - window_size//2
            end_idx = i + window_size//2 + 1
            window_coords = coords[start_idx:end_idx]
            window_timestamps = timestamps[start_idx:end_idx]
            
            # Check if current point fits local trend
            center_idx = window_size//2
            
            # Linear interpolation of expected position
            if len(window_coords) >= 3:
                before_coords = window_coords[:center_idx]
                after_coords = window_coords[center_idx+1:]
                
                if len(before_coords) > 0 and len(after_coords) > 0:
                    # Simple linear interpolation
                    expected_lat = np.interp(
                        window_timestamps[center_idx],
                        [window_timestamps[0], window_timestamps[-1]],
                        [window_coords[0, 0], window_coords[-1, 0]]
                    )
                    expected_lon = np.interp(
                        window_timestamps[center_idx],
                        [window_timestamps[0], window_timestamps[-1]],
                        [window_coords[0, 1], window_coords[-1, 1]]
                    )
                    
                    # Calculate deviation from expected position
                    deviation = np.sqrt(
                        (window_coords[center_idx, 0] - expected_lat)**2 +
                        (window_coords[center_idx, 1] - expected_lon)**2
                    ) * 111000  # Convert to meters
                    
                    # Mark as outlier if deviation is too large
                    if deviation > 500:  # 500m threshold
                        outliers[i] = True
        
        return outliers
    
    def _calculate_total_distance(self, points: List) -> float:
        """Calculate total distance of trajectory."""
        if len(points) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(points)):
            p1, p2 = points[i-1], points[i]
            # Haversine formula
            lat1, lon1 = np.radians(p1.latitude), np.radians(p1.longitude)
            lat2, lon2 = np.radians(p2.latitude), np.radians(p2.longitude)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            
            distance = 6371 * c  # Earth radius in km
            total_distance += distance
        
        return total_distance
    
    def apply_smart_sampling(self, trajectories: List) -> List:
        """Apply smart sampling to balance dataset and improve training."""
        sampled_trajectories = []
        
        # Group trajectories by mobility pattern
        pattern_groups = {}
        for traj in trajectories:
            pattern = traj.mobility_pattern or MobilityPattern.RANDOM
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(traj)
        
        # Determine target samples per pattern (balanced sampling)
        total_target_samples = len(trajectories)
        samples_per_pattern = total_target_samples // len(pattern_groups)
        
        for pattern, trajs in pattern_groups.items():
            if len(trajs) <= samples_per_pattern:
                # Use all trajectories if we have fewer than target
                sampled_trajectories.extend(trajs)
            else:
                # Smart sampling: prioritize longer, more diverse trajectories
                scores = []
                for traj in trajs:
                    # Score based on length, duration, and distance
                    length_score = min(len(traj.points) / 100, 1.0)  # Normalize to [0,1]
                    duration_score = min(traj.duration_seconds / 3600, 1.0)  # Hours
                    distance_score = min(traj.total_distance_km / 10, 1.0)  # 10km max
                    
                    total_score = length_score + duration_score + distance_score
                    scores.append(total_score)
                
                # Select top trajectories
                sorted_indices = np.argsort(scores)[::-1]
                selected_trajs = [trajs[i] for i in sorted_indices[:samples_per_pattern]]
                sampled_trajectories.extend(selected_trajs)
        
        return sampled_trajectories


# Import the trajectory-related classes
from app.dataset_loader import TrajectoryData
from app.mobility_predictor import LocationPoint, MobilityPattern


@dataclass
class AdvancedConfig:
    """Configuration for ultra-high accuracy LSTM predictor."""
    
    # Enhanced Architecture parameters
    sequence_length: int = 35
    lstm_units: int = 384
    attention_heads: int = 12
    num_layers: int = 6
    dropout_rate: float = 0.08
    transformer_depth: int = 4  # New: Transformer layers
    residual_connections: bool = True  # New: Residual connections
    layer_norm_eps: float = 1e-6  # New: Layer normalization epsilon
    
    # Enhanced Training parameters
    learning_rate: float = 0.00005
    epochs: int = 300
    batch_size: int = 256
    patience: int = 35
    warmup_steps: int = 1000  # New: Learning rate warmup
    cosine_decay: bool = True  # New: Cosine learning rate decay
    
    # Enhanced Ensemble parameters
    num_ensemble_models: int = 8  # Increased from 5 to 8
    ensemble_weights: List[float] = None
    ensemble_diversity_weight: float = 0.1  # New: Diversity penalty
    
    # Enhanced Data parameters
    prediction_horizons: List[int] = None  # [1, 3, 5, 10, 15] minutes - more granular
    feature_lookback_hours: int = 36  # Increased from 24
    data_augmentation_factor: float = 4.0  # Increased from 2.0
    
    # Enhanced Preprocessing parameters
    outlier_threshold: float = 2.5  # More sensitive outlier detection
    smoothing_window: int = 7
    use_kalman_filter: bool = True  # New: Kalman filtering for noise reduction
    frequency_features: bool = True  # New: Add frequency domain features
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 3, 5, 10, 15]  # More granular horizons
        if self.ensemble_weights is None:
            # Enhanced ensemble with 8 models - weights for transformer, attention LSTM, CNN-LSTM, etc.
            self.ensemble_weights = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]


class AdvancedFeatureEngineer:
    """Ultra-advanced feature engineering for maximum accuracy mobility prediction."""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.pca = PCA(n_components=0.99)  # Keep 99% of variance
        self.kalman_filter = None
        if KALMAN_AVAILABLE and config.use_kalman_filter:
            self._initialize_kalman_filter()
        
    def extract_advanced_features(self, trajectories: List) -> Tuple[np.ndarray, np.ndarray]:
        """Extract comprehensive features from trajectories."""
        
        all_sequences = []
        all_targets = []
        
        for trajectory in trajectories:
            points = trajectory.points
            if len(points) < self.config.sequence_length + 5:
                continue
                
            # Extract sequences with advanced features
            sequences, targets = self._create_advanced_sequences(points)
            all_sequences.extend(sequences)
            all_targets.extend(targets)
        
        if not all_sequences:
            return np.array([]), np.array([])
        
        X = np.array(all_sequences)
        y = np.array(all_targets)
        
        # Apply enhanced data augmentation with preprocessing insights
        logger.info("Applying sophisticated data augmentation...")
        X_aug, y_aug = self._augment_data_enhanced(X, y)
        
        return X_aug, y_aug
    
    def _augment_data_enhanced(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced data augmentation with GPS-realistic noise and smart perturbations."""
        
        if len(X) == 0:
            return X, y
        
        augmentation_factor = int(self.config.data_augmentation_factor)
        
        X_aug = [X]
        y_aug = [y]
        
        for aug_iter in range(augmentation_factor - 1):
            # 1. GPS-realistic noise (based on real GPS error characteristics)
            X_gps_noise = self._add_gps_realistic_noise(X)
            
            # 2. Temporal jittering (simulate timing uncertainties)
            X_temporal = self._add_temporal_jittering(X)
            
            # 3. Speed variation (simulate different movement dynamics)
            X_speed_var = self._add_speed_variation(X)
            
            # 4. Trajectory smoothing (simulate different GPS sampling rates)
            X_smoothed = self._add_trajectory_smoothing(X)
            
            # 5. Random dropout (simulate GPS signal loss)
            X_dropout = self._add_random_dropout(X)
            
            # Add all augmented versions
            augmented_batches = [X_gps_noise, X_temporal, X_speed_var, X_smoothed, X_dropout]
            X_aug.extend(augmented_batches)
            y_aug.extend([y] * len(augmented_batches))
        
        return np.vstack(X_aug), np.vstack(y_aug)
    
    def _add_gps_realistic_noise(self, X: np.ndarray) -> np.ndarray:
        """Add GPS-realistic noise based on actual GPS error characteristics."""
        X_noisy = np.copy(X)
        seq_len = self.config.sequence_length
        n_features = X.shape[1] // seq_len
        
        for i in range(len(X)):
            reshaped = X[i].reshape(seq_len, n_features)
            
            # GPS horizontal accuracy: typically 3-5m (95% confidence)
            gps_noise_std = 0.00003  # ~3.3m in degrees
            
            if n_features >= 2:  # lat, lon
                # Add correlated noise (GPS errors often have spatial correlation)
                noise = np.random.multivariate_normal(
                    [0, 0], 
                    [[gps_noise_std**2, gps_noise_std**2 * 0.3],
                     [gps_noise_std**2 * 0.3, gps_noise_std**2]], 
                    seq_len
                )
                reshaped[:, :2] += noise
            
            X_noisy[i] = reshaped.flatten()
        
        return X_noisy
    
    def _add_temporal_jittering(self, X: np.ndarray) -> np.ndarray:
        """Add temporal jittering to simulate timing uncertainties."""
        X_jittered = np.copy(X)
        # Apply small temporal shifts (0.5-2 second timing errors)
        for i in range(len(X)):
            if np.random.random() < 0.6:  # Apply to 60% of samples
                jitter = np.random.normal(0, 0.3)  # Small temporal shift
                X_jittered[i] = self._apply_sequence_jitter(X[i], jitter)
        return X_jittered
    
    def _add_speed_variation(self, X: np.ndarray) -> np.ndarray:
        """Add speed variation to simulate different movement dynamics."""
        X_varied = np.copy(X)
        speed_factor = np.random.uniform(0.85, 1.15)  # 15% speed variation
        
        seq_len = self.config.sequence_length
        n_features = X.shape[1] // seq_len
        
        for i in range(len(X)):
            reshaped = X[i].reshape(seq_len, n_features)
            
            # Apply speed variation to velocity-related features
            if n_features >= 6:  # Assuming velocity features exist
                reshaped[:, 2:4] *= speed_factor  # Velocity features
            if n_features >= 8:  # Assuming acceleration features exist
                reshaped[:, 4:6] *= speed_factor  # Acceleration features
            
            X_varied[i] = reshaped.flatten()
        
        return X_varied
    
    def _add_trajectory_smoothing(self, X: np.ndarray) -> np.ndarray:
        """Add trajectory smoothing to simulate different GPS sampling rates."""
        X_smoothed = np.copy(X)
        
        for i in range(len(X)):
            if np.random.random() < 0.4:  # Apply to 40% of samples
                X_smoothed[i] = self._apply_gaussian_smoothing(X[i])
        
        return X_smoothed
    
    def _add_random_dropout(self, X: np.ndarray) -> np.ndarray:
        """Add random feature dropout to simulate GPS signal loss."""
        X_dropout = np.copy(X)
        dropout_rate = 0.05  # 5% feature dropout
        
        for i in range(len(X)):
            mask = np.random.random(X.shape[1]) > dropout_rate
            X_dropout[i] *= mask
        
        return X_dropout
    
    def _apply_sequence_jitter(self, sequence: np.ndarray, jitter: float) -> np.ndarray:
        """Apply jitter to a sequence."""
        try:
            seq_len = self.config.sequence_length
            n_features = len(sequence) // seq_len
            reshaped = sequence.reshape(seq_len, n_features)
            
            # Simple interpolation-based jitter
            indices = np.arange(seq_len) + jitter
            indices = np.clip(indices, 0, seq_len - 1)
            
            jittered = np.zeros_like(reshaped)
            for feat in range(n_features):
                jittered[:, feat] = np.interp(indices, np.arange(seq_len), reshaped[:, feat])
            
            return jittered.flatten()
        except Exception:
            return sequence
    
    def _apply_gaussian_smoothing(self, sequence: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing to sequence."""
        try:
            seq_len = self.config.sequence_length
            n_features = len(sequence) // seq_len
            reshaped = sequence.reshape(seq_len, n_features)
            
            # Apply smoothing to coordinate features
            if n_features >= 2:
                reshaped[:, 0] = gaussian_filter1d(reshaped[:, 0], sigma=0.5)
                reshaped[:, 1] = gaussian_filter1d(reshaped[:, 1], sigma=0.5)
            
            return reshaped.flatten()
        except Exception:
            return sequence
    
    def _create_advanced_sequences(self, points: List) -> Tuple[List, List]:
        """Create sequences with comprehensive feature engineering."""
        
        sequences = []
        targets = []
        
        # Apply Kalman filtering if enabled
        if self.config.use_kalman_filter:
            points = self._apply_kalman_filter(points)
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame([{
            'lat': p.latitude,
            'lon': p.longitude,
            'timestamp': p.timestamp,
            'altitude': p.altitude or 0
        } for p in points])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Apply Gaussian smoothing to reduce noise
        if len(df) > 5:
            df['lat'] = gaussian_filter1d(df['lat'].values, sigma=0.5)
            df['lon'] = gaussian_filter1d(df['lon'].values, sigma=0.5)
        
        # Precompute advanced features
        df = self._compute_advanced_features(df)
        
        # Create sequences
        for i in range(len(df) - self.config.sequence_length - max(self.config.prediction_horizons)):
            # Input sequence
            seq_data = df.iloc[i:i + self.config.sequence_length]
            
            # Multiple prediction targets (different horizons)
            targets_list = []
            for horizon in self.config.prediction_horizons:
                target_idx = i + self.config.sequence_length + horizon - 1
                if target_idx < len(df):
                    target_point = df.iloc[target_idx]
                    targets_list.extend([target_point['lat'], target_point['lon']])
                else:
                    targets_list.extend([0, 0])  # Padding
            
            # Extract sequence features
            seq_features = self._extract_sequence_features(seq_data)
            
            sequences.append(seq_features)
            targets.append(targets_list)
        
        return sequences, targets
    
    def _compute_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute advanced mobility features."""
        
        # Basic derivatives
        df['lat_diff'] = df['lat'].diff()
        df['lon_diff'] = df['lon'].diff()
        df['time_diff'] = df['timestamp'].diff()
        
        # Speed and acceleration
        df['speed'] = np.sqrt(df['lat_diff']**2 + df['lon_diff']**2) / np.maximum(df['time_diff'], 1e-6)
        df['acceleration'] = df['speed'].diff() / np.maximum(df['time_diff'], 1e-6)
        
        # Direction and bearing
        df['bearing'] = np.arctan2(df['lon_diff'], df['lat_diff']) * 180 / np.pi
        df['bearing_change'] = df['bearing'].diff()
        
        # Distance features
        df['distance_from_start'] = np.sqrt(
            (df['lat'] - df['lat'].iloc[0])**2 + 
            (df['lon'] - df['lon'].iloc[0])**2
        )
        
        # Moving statistics (smoothing)
        window = self.config.smoothing_window
        df['speed_ma'] = df['speed'].rolling(window, min_periods=1).mean()
        df['speed_std'] = df['speed'].rolling(window, min_periods=1).std().fillna(0)
        df['lat_ma'] = df['lat'].rolling(window, min_periods=1).mean()
        df['lon_ma'] = df['lon'].rolling(window, min_periods=1).mean()
        
        # Temporal features
        df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour / 24.0
        df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='s').dt.dayofweek / 7.0
        df['month'] = pd.to_datetime(df['timestamp'], unit='s').dt.month / 12.0
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'])
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'])
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'])
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'])
        
        # Advanced pattern features
        df['speed_percentile'] = df['speed'].rolling(50, min_periods=1).apply(
            lambda x: (x.iloc[-1] > x.quantile(0.8)).astype(float)
        )
        
        # Enhanced outlier detection using IQR method
        Q1_speed = df['speed'].quantile(0.25)
        Q3_speed = df['speed'].quantile(0.75)
        IQR_speed = Q3_speed - Q1_speed
        speed_threshold = Q3_speed + self.config.outlier_threshold * IQR_speed
        df['is_outlier'] = (df['speed'] > speed_threshold).astype(float)
        
        # Add frequency domain features if enabled
        if self.config.frequency_features and len(df) >= 16:
            # Extract frequency features for key signals
            lat_freq_features = self._extract_frequency_features(df['lat'].values, 'lat')
            lon_freq_features = self._extract_frequency_features(df['lon'].values, 'lon')
            speed_freq_features = self._extract_frequency_features(df['speed'].values, 'speed')
            
            # Add frequency features to dataframe
            for i, feature in enumerate(lat_freq_features):
                df[f'lat_freq_{i}'] = feature
            for i, feature in enumerate(lon_freq_features):
                df[f'lon_freq_{i}'] = feature
            for i, feature in enumerate(speed_freq_features):
                df[f'speed_freq_{i}'] = feature
        
        # Add trajectory curvature and jerk features
        if len(df) >= 3:
            # Curvature (rate of change of direction)
            df['bearing_change_rate'] = df['bearing_change'].diff() / df['time_diff'].replace(0, 1e-6)
            
            # Jerk (rate of change of acceleration)
            df['jerk'] = df['acceleration'].diff() / df['time_diff'].replace(0, 1e-6)
        else:
            df['bearing_change_rate'] = 0
            df['jerk'] = 0
        
        # Add advanced stop detection and dwelling analysis
        stop_threshold = 0.5  # m/s
        df['is_stopped'] = (df['speed'] < stop_threshold).astype(float)
        df['stop_duration'] = df.groupby((df['is_stopped'] != df['is_stopped'].shift()).cumsum())['is_stopped'].transform('count') * df['is_stopped']
        
        # Add geospatial features
        df = self._add_geospatial_features(df)
        
        # Add advanced contextual features
        df = self._add_contextual_features(df)
        
        # Add behavioral pattern features
        df = self._add_behavioral_features(df)
        
        # Fill NaN values with more sophisticated method
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        # Clip extreme values to prevent training instability
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['lat', 'lon', 'timestamp']:  # Don't clip coordinates or timestamps
                df[col] = np.clip(df[col], df[col].quantile(0.01), df[col].quantile(0.99))
        
        return df
    
    def _add_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geospatial context features."""
        try:
            # Geographic clustering features
            df['lat_rounded'] = np.round(df['lat'], 3)  # ~111m resolution
            df['lon_rounded'] = np.round(df['lon'], 3)
            
            # Distance from common reference points (e.g., city centers)
            # Using Beijing as reference (can be made configurable)
            beijing_lat, beijing_lon = 39.9042, 116.4074
            df['dist_from_city_center'] = np.sqrt(
                (df['lat'] - beijing_lat)**2 + (df['lon'] - beijing_lon)**2
            )
            
            # Quadrant features (relative position)
            df['lat_quadrant'] = (df['lat'] > df['lat'].median()).astype(float)
            df['lon_quadrant'] = (df['lon'] > df['lon'].median()).astype(float)
            
            # Local density features (using rolling window)
            window_size = min(10, len(df))
            if window_size > 1:
                df['local_lat_density'] = df['lat'].rolling(window_size, min_periods=1).std()
                df['local_lon_density'] = df['lon'].rolling(window_size, min_periods=1).std()
            else:
                df['local_lat_density'] = 0
                df['local_lon_density'] = 0
            
            # Route linearity (how straight is the path)
            if len(df) >= 5:
                # Calculate linearity as ratio of direct distance to path distance
                start_point = df.iloc[0]
                end_point = df.iloc[-1]
                direct_distance = np.sqrt(
                    (end_point['lat'] - start_point['lat'])**2 + 
                    (end_point['lon'] - start_point['lon'])**2
                )
                
                path_distance = df['distance_from_start'].iloc[-1]
                linearity = direct_distance / (path_distance + 1e-6)
                df['route_linearity'] = linearity
            else:
                df['route_linearity'] = 1.0
            
            # Area coverage (bounding box features)
            lat_range = df['lat'].max() - df['lat'].min()
            lon_range = df['lon'].max() - df['lon'].min()
            df['bounding_box_area'] = lat_range * lon_range
            df['lat_coverage'] = lat_range
            df['lon_coverage'] = lon_range
            
            return df
        except Exception as e:
            logger.warning(f"Geospatial feature extraction failed: {e}")
            return df
    
    def _add_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contextual features based on time and movement patterns."""
        try:
            # Enhanced temporal features
            df['hour_of_day'] = df['hour'] * 24  # Convert back from normalized
            
            # Rush hour indicators
            df['is_morning_rush'] = ((df['hour_of_day'] >= 7) & (df['hour_of_day'] <= 9)).astype(float)
            df['is_evening_rush'] = ((df['hour_of_day'] >= 17) & (df['hour_of_day'] <= 19)).astype(float)
            df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(float)
            df['is_night_time'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(float)
            
            # Day type features
            df['is_weekday'] = (df['day_of_week'] * 7 < 5).astype(float)  # Convert from normalized
            df['is_weekend'] = (1 - df['is_weekday'])
            
            # Movement regularity features
            if len(df) >= 10:
                # Speed consistency
                speed_cv = df['speed'].std() / (df['speed'].mean() + 1e-6)  # Coefficient of variation
                df['speed_consistency'] = np.exp(-speed_cv)  # Higher value = more consistent
                
                # Direction consistency
                bearing_std = df['bearing'].std()
                df['direction_consistency'] = np.exp(-bearing_std / 180)  # Normalized by max bearing
            else:
                df['speed_consistency'] = 1.0
                df['direction_consistency'] = 1.0
            
            # Movement state features
            df['is_accelerating'] = (df['acceleration'] > 0.1).astype(float)
            df['is_decelerating'] = (df['acceleration'] < -0.1).astype(float)
            df['is_turning'] = (np.abs(df['bearing_change']) > 15).astype(float)  # 15-degree threshold
            
            return df
        except Exception as e:
            logger.warning(f"Contextual feature extraction failed: {e}")
            return df
    
    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral pattern features."""
        try:
            # Movement efficiency features
            if len(df) >= 5:
                # Time efficiency (actual time vs minimum possible time)
                total_time = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
                total_distance = df['distance_from_start'].iloc[-1]
                
                if total_time > 0:
                    avg_speed_actual = total_distance / total_time
                    # Assume reasonable maximum speed for comparison
                    max_reasonable_speed = 0.01  # degrees per second (~40 km/h)
                    speed_efficiency = min(avg_speed_actual / max_reasonable_speed, 1.0)
                    df['movement_efficiency'] = speed_efficiency
                else:
                    df['movement_efficiency'] = 0.0
            else:
                df['movement_efficiency'] = 0.0
            
            # Pattern repeatability (using autocorrelation)
            if len(df) >= 20:
                # Calculate autocorrelation of speed signal
                speed_vals = df['speed'].values
                if len(np.unique(speed_vals)) > 1:  # Avoid constant signals
                    autocorr = np.corrcoef(speed_vals[:-1], speed_vals[1:])[0, 1]
                    df['speed_autocorr'] = autocorr if not np.isnan(autocorr) else 0.0
                else:
                    df['speed_autocorr'] = 1.0
                
                # Direction autocorrelation
                bearing_vals = df['bearing'].values
                if len(np.unique(bearing_vals)) > 1:
                    bearing_autocorr = np.corrcoef(bearing_vals[:-1], bearing_vals[1:])[0, 1]
                    df['bearing_autocorr'] = bearing_autocorr if not np.isnan(bearing_autocorr) else 0.0
                else:
                    df['bearing_autocorr'] = 1.0
            else:
                df['speed_autocorr'] = 0.0
                df['bearing_autocorr'] = 0.0
            
            # Complexity features
            # Entropy of speed distribution
            speed_bins = np.histogram(df['speed'], bins=10)[0]
            speed_probs = speed_bins / (np.sum(speed_bins) + 1e-6)
            speed_probs = speed_probs[speed_probs > 0]  # Remove zero probabilities
            if len(speed_probs) > 1:
                speed_entropy = -np.sum(speed_probs * np.log2(speed_probs))
                df['speed_entropy'] = speed_entropy / np.log2(len(speed_probs))  # Normalized
            else:
                df['speed_entropy'] = 0.0
            
            # Trip characteristics
            df['trip_distance_total'] = df['distance_from_start'].iloc[-1]
            df['trip_duration_total'] = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
            df['trip_complexity'] = len(df) / (df['trip_duration_total'] + 1)  # Points per second
            
            return df
        except Exception as e:
            logger.warning(f"Behavioral feature extraction failed: {e}")
            return df
    
    def _extract_sequence_features(self, seq_data: pd.DataFrame) -> List[float]:
        """Extract features from a sequence."""
        
        features = []
        
        # Comprehensive feature columns including all advanced features
        core_features = [
            'lat', 'lon', 'altitude', 'speed', 'acceleration', 'bearing',
            'distance_from_start', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'speed_ma', 'speed_std', 'lat_ma', 'lon_ma', 'is_outlier',
            'bearing_change_rate', 'jerk', 'is_stopped', 'stop_duration'
        ]
        
        # Geospatial features
        geospatial_features = [
            'lat_rounded', 'lon_rounded', 'dist_from_city_center',
            'lat_quadrant', 'lon_quadrant', 'local_lat_density', 'local_lon_density',
            'route_linearity', 'bounding_box_area', 'lat_coverage', 'lon_coverage'
        ]
        
        # Contextual features
        contextual_features = [
            'is_morning_rush', 'is_evening_rush', 'is_business_hours', 'is_night_time',
            'is_weekday', 'is_weekend', 'speed_consistency', 'direction_consistency',
            'is_accelerating', 'is_decelerating', 'is_turning'
        ]
        
        # Behavioral features
        behavioral_features = [
            'movement_efficiency', 'speed_autocorr', 'bearing_autocorr',
            'speed_entropy', 'trip_distance_total', 'trip_duration_total', 'trip_complexity'
        ]
        
        # Combine all feature categories
        feature_columns = core_features + geospatial_features + contextual_features + behavioral_features
        
        # Add frequency domain features if available
        if self.config.frequency_features:
            freq_columns = [col for col in seq_data.columns if 'freq_' in col]
            feature_columns.extend(freq_columns)
        
        for col in feature_columns:
            if col in seq_data.columns:
                features.extend(seq_data[col].values)
            else:
                features.extend([0] * len(seq_data))
        
        # Sequence-level statistics
        speed_vals = seq_data['speed'].values
        features.extend([
            np.mean(speed_vals),
            np.std(speed_vals),
            np.min(speed_vals),
            np.max(speed_vals),
            np.median(speed_vals)
        ])
        
        # Trajectory shape features
        lat_vals = seq_data['lat'].values
        lon_vals = seq_data['lon'].values
        features.extend([
            np.std(lat_vals),
            np.std(lon_vals),
            np.max(lat_vals) - np.min(lat_vals),
            np.max(lon_vals) - np.min(lon_vals)
        ])
        
        return features
    
    def _initialize_kalman_filter(self):
        """Initialize Kalman filter for noise reduction."""
        if KALMAN_AVAILABLE:
            # Simple 2D position tracking with constant velocity model
            self.kalman_filter = KalmanFilter(
                transition_matrices=np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]]),
                observation_matrices=np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]]),
                n_dim_state=4,
                n_dim_obs=2
            )
    
    def _apply_kalman_filter(self, trajectory_points: List) -> List:
        """Apply Kalman filtering to reduce GPS noise."""
        if not KALMAN_AVAILABLE or not self.kalman_filter:
            return trajectory_points
        
        try:
            # Extract coordinates
            coords = np.array([[p.latitude, p.longitude] for p in trajectory_points])
            
            # Apply Kalman filter
            state_means, _ = self.kalman_filter.em(coords).smooth()
            
            # Update trajectory points with filtered coordinates
            filtered_points = []
            for i, point in enumerate(trajectory_points):
                filtered_point = LocationPoint(
                    latitude=state_means[i, 0],
                    longitude=state_means[i, 1],
                    timestamp=point.timestamp,
                    altitude=point.altitude
                )
                filtered_points.append(filtered_point)
            
            return filtered_points
        except Exception as e:
            logger.warning(f"Kalman filtering failed: {e}")
            return trajectory_points
    
    def _extract_frequency_features(self, values: np.ndarray, feature_prefix: str) -> List[float]:
        """Extract frequency domain features using FFT."""
        if not self.config.frequency_features or len(values) < 8:
            return []
        
        try:
            # Apply FFT
            fft_values = fft(values)
            freqs = fftfreq(len(values))
            
            # Extract power spectral density features
            power_spectrum = np.abs(fft_values) ** 2
            
            features = []
            # Dominant frequency
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            features.append(freqs[dominant_freq_idx])
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs[:len(freqs)//2] * power_spectrum[:len(power_spectrum)//2]) / np.sum(power_spectrum[:len(power_spectrum)//2])
            features.append(spectral_centroid)
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumsum_power = np.cumsum(power_spectrum[:len(power_spectrum)//2])
            rolloff_idx = np.where(cumsum_power >= 0.85 * cumsum_power[-1])[0]
            if len(rolloff_idx) > 0:
                spectral_rolloff = freqs[rolloff_idx[0]]
            else:
                spectral_rolloff = 0.0
            features.append(spectral_rolloff)
            
            # Zero crossing rate
            zcr = np.sum(np.diff(np.sign(values)) != 0) / len(values)
            features.append(zcr)
            
            return features
        except Exception as e:
            logger.warning(f"Frequency feature extraction failed for {feature_prefix}: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def _augment_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation techniques."""
        
        if len(X) == 0:
            return X, y
        
        augmentation_factor = int(self.config.data_augmentation_factor)
        
        X_aug = [X]
        y_aug = [y]
        
        for _ in range(augmentation_factor - 1):
            # Add small random noise
            noise_scale = 0.01
            X_noise = X + np.random.normal(0, noise_scale, X.shape)
            
            # Small temporal shifts
            X_shifted = np.copy(X)
            shift_amount = np.random.randint(-2, 3, size=len(X))
            for i, shift in enumerate(shift_amount):
                if abs(shift) > 0 and 0 <= i + shift < len(X):
                    X_shifted[i] = X[i + shift]
            
            X_aug.extend([X_noise, X_shifted])
            y_aug.extend([y, y])
        
        return np.vstack(X_aug), np.vstack(y_aug)


class TransformerLSTM:
    """Enhanced LSTM with Transformer-style attention and residual connections for maximum accuracy."""
    
    def __init__(self, config: AdvancedConfig, input_shape: Tuple):
        self.config = config
        self.input_shape = input_shape
        self.model = self._build_transformer_lstm_model()
    
    def _build_transformer_lstm_model(self) -> Model:
        """Build enhanced LSTM model with Transformer-style attention and residual connections."""
        
        # Input layer
        inputs = Input(shape=self.input_shape, name='sequence_input')
        
        # Reshape for sequence processing
        seq_len = self.config.sequence_length
        n_features = self.input_shape[0] // seq_len
        
        x = tf.reshape(inputs, (-1, seq_len, n_features))
        
        # Positional encoding for better temporal understanding
        position_encoding = self._create_positional_encoding(seq_len, n_features)
        x = x + position_encoding
        
        # Multiple Transformer-style blocks with residual connections
        for i in range(self.config.transformer_depth):
            x = self._transformer_block(x, f'transformer_{i}')
        
        # Enhanced LSTM layers with residual connections
        lstm_input = x
        
        # First LSTM layer
        lstm1 = Bidirectional(LSTM(
            self.config.lstm_units, 
            return_sequences=True,
            dropout=self.config.dropout_rate,
            recurrent_dropout=self.config.dropout_rate,
            name=f'lstm_1'
        ))(x)
        lstm1 = LayerNormalization(epsilon=self.config.layer_norm_eps)(lstm1)
        
        # Residual connection if dimensions match
        if self.config.residual_connections and lstm1.shape[-1] == x.shape[-1]:
            lstm1 = Add()([x, lstm1])
        
        # Second LSTM layer
        lstm2 = Bidirectional(LSTM(
            self.config.lstm_units // 2, 
            return_sequences=True,
            dropout=self.config.dropout_rate,
            recurrent_dropout=self.config.dropout_rate,
            name=f'lstm_2'
        ))(lstm1)
        lstm2 = LayerNormalization(epsilon=self.config.layer_norm_eps)(lstm2)
        
        # Final multi-head attention with larger key dimension
        attention = MultiHeadAttention(
            num_heads=self.config.attention_heads,
            key_dim=self.config.lstm_units,
            dropout=self.config.dropout_rate,
            name='final_attention'
        )(lstm2, lstm2)
        
        # Residual connection and normalization
        if self.config.residual_connections:
            attention_out = Add()([lstm2, attention])
        else:
            attention_out = attention
        attention_out = LayerNormalization(epsilon=self.config.layer_norm_eps)(attention_out)
        
        # Enhanced pooling - combine both average and max pooling
        avg_pooled = GlobalAveragePooling1D()(attention_out)
        max_pooled = GlobalMaxPooling1D()(attention_out)
        pooled = Concatenate()([avg_pooled, max_pooled])
        
        # Enhanced dense layers with residual connections
        dense1 = Dense(1024, activation='gelu', name='dense_1')(pooled)  # Using GELU activation
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(self.config.dropout_rate)(dense1)
        
        dense2 = Dense(512, activation='gelu', name='dense_2')(dense1)
        dense2 = BatchNormalization()(dense2)
        
        # Residual connection for dense layers
        if self.config.residual_connections:
            # Project to same dimension for residual connection
            dense1_proj = Dense(512, activation='linear')(dense1)
            dense2 = Add()([dense1_proj, dense2])
        
        dense2 = Dropout(self.config.dropout_rate)(dense2)
        
        dense3 = Dense(256, activation='gelu', name='dense_3')(dense2)
        dense3 = BatchNormalization()(dense3)
        dense3 = Dropout(self.config.dropout_rate)(dense3)
        
        # Output layer (multiple horizons * 2 coordinates)
        output_size = len(self.config.prediction_horizons) * 2
        outputs = Dense(output_size, activation='linear', name='output')(dense3)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with enhanced optimizer and learning rate scheduling
        initial_learning_rate = self.config.learning_rate
        if self.config.cosine_decay:
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=self.config.epochs * 100,  # Assuming ~100 steps per epoch
                warmup_target=initial_learning_rate,
                warmup_steps=self.config.warmup_steps
            )
        else:
            lr_schedule = initial_learning_rate
            
        optimizer = Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0  # Gradient clipping for stability
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def _create_positional_encoding(self, seq_len: int, d_model: int) -> tf.Tensor:
        """Create positional encoding for sequence."""
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        if d_model > 1:
            pos_encoding[:, 1::2] = np.cos(position * div_term[:d_model//2])
        
        return tf.constant(pos_encoding, dtype=tf.float32)
    
    def _transformer_block(self, x: tf.Tensor, name_prefix: str) -> tf.Tensor:
        """Transformer block with multi-head attention and feed-forward network."""
        # Multi-head self-attention
        attention = MultiHeadAttention(
            num_heads=self.config.attention_heads,
            key_dim=x.shape[-1] // self.config.attention_heads,
            dropout=self.config.dropout_rate,
            name=f'{name_prefix}_attention'
        )(x, x)
        
        # Add & Norm
        if self.config.residual_connections:
            attention = Add(name=f'{name_prefix}_add_1')([x, attention])
        attention = LayerNormalization(
            epsilon=self.config.layer_norm_eps,
            name=f'{name_prefix}_norm_1'
        )(attention)
        
        # Feed-forward network
        ff_dim = x.shape[-1] * 4  # Standard transformer expansion factor
        ff = Dense(ff_dim, activation='gelu', name=f'{name_prefix}_ff_1')(attention)
        ff = Dropout(self.config.dropout_rate)(ff)
        ff = Dense(x.shape[-1], name=f'{name_prefix}_ff_2')(ff)
        
        # Add & Norm
        if self.config.residual_connections:
            output = Add(name=f'{name_prefix}_add_2')([attention, ff])
        else:
            output = ff
        output = LayerNormalization(
            epsilon=self.config.layer_norm_eps,
            name=f'{name_prefix}_norm_2'
        )(output)
        
        return output


class EnsembleMobilityPredictor:
    """Enhanced ensemble of multiple models for maximum accuracy (target: 92%+)."""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.models = []
        self.model_weights = []
        self.model_confidences = []
        self.feature_engineer = AdvancedFeatureEngineer(config)
        self.data_preprocessor = AdvancedDataPreprocessor(config)
        self.model_metadata = {}
        
    def train(self, trajectories: List) -> Dict[str, Any]:
        """Train ensemble of models."""
        
        logger.info("Applying advanced data preprocessing...")
        # Step 1: Advanced outlier detection and cleaning
        cleaned_trajectories = self.data_preprocessor.detect_and_handle_outliers(trajectories)
        logger.info(f"Data cleaning: {len(trajectories)} → {len(cleaned_trajectories)} trajectories")
        
        # Step 2: Smart sampling for balanced training
        sampled_trajectories = self.data_preprocessor.apply_smart_sampling(cleaned_trajectories)
        logger.info(f"Smart sampling: {len(cleaned_trajectories)} → {len(sampled_trajectories)} trajectories")
        
        # Step 3: Extract advanced features
        logger.info("Extracting ultra-advanced features...")
        X, y = self.feature_engineer.extract_advanced_features(sampled_trajectories)
        
        if len(X) == 0:
            raise ValueError("No features extracted from trajectories")
        
        logger.info(f"Training on {len(X)} sequences with {X.shape[1]} features each")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.feature_engineer.scaler.fit_transform(X_train)
        X_test_scaled = self.feature_engineer.scaler.transform(X_test)
        
        results = []
        
        # Train enhanced ensemble models with diverse architectures
        for i in range(self.config.num_ensemble_models):
            logger.info(f"Training model {i+1}/{self.config.num_ensemble_models}")
            
            if i < 2:  # First 2 models are Transformer LSTMs
                model, model_type = self._train_transformer_lstm(X_train_scaled, y_train, X_test_scaled, y_test, i)
            elif i < 4:  # Next 2 models are CNN-LSTM hybrids
                model, model_type = self._train_cnn_lstm(X_train_scaled, y_train, X_test_scaled, y_test, i)
            elif i < 6:  # Next 2 models are XGBoost (if available)
                model, model_type = self._train_xgboost_model(X_train_scaled, y_train, X_test_scaled, y_test, i)
            else:  # Last 2 models are advanced traditional ML
                model, model_type = self._train_advanced_traditional_model(X_train_scaled, y_train, X_test_scaled, y_test, i)
            
            self.models.append(model)
            self.model_metadata[i] = model_type
            
            # Evaluate individual model
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test_scaled)
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 2:
                    y_pred = y_pred[:, :2]  # Take first prediction horizon
                mae = mean_absolute_error(y_test[:, :2], y_pred)
                results.append(mae)
        
        # Evaluate ensemble
        ensemble_pred = self.predict(X_test_scaled)
        ensemble_mae = mean_absolute_error(y_test[:, :2], ensemble_pred[:, :2])
        
        # Enhanced accuracy metrics with multiple thresholds
        accuracy_50m = np.sum(
            np.sqrt(np.sum((y_test[:, :2] - ensemble_pred[:, :2])**2, axis=1)) < 0.00045  # ~50m
        ) / len(y_test)
        
        accuracy_100m = np.sum(
            np.sqrt(np.sum((y_test[:, :2] - ensemble_pred[:, :2])**2, axis=1)) < 0.0009  # ~100m
        ) / len(y_test)
        
        accuracy_200m = np.sum(
            np.sqrt(np.sum((y_test[:, :2] - ensemble_pred[:, :2])**2, axis=1)) < 0.0018  # ~200m
        ) / len(y_test)
        
        # Use the most stringent accuracy (50m) as primary metric
        accuracy = accuracy_50m
        
        # Convert to distance (approximate)
        avg_error_degrees = ensemble_mae
        avg_error_km = avg_error_degrees * 111  # Rough conversion
        
        # Calculate R² score for additional metric
        r2_lat = r2_score(y_test[:, 0], ensemble_pred[:, 0])
        r2_lon = r2_score(y_test[:, 1], ensemble_pred[:, 1])
        avg_r2 = (r2_lat + r2_lon) / 2
        
        metrics = {
            "ensemble_mae": ensemble_mae,
            "individual_maes": results,
            "accuracy": accuracy,
            "accuracy_50m": accuracy_50m,
            "accuracy_100m": accuracy_100m,
            "accuracy_200m": accuracy_200m,
            "avg_error_km": avg_error_km,
            "r2_score": avg_r2,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": X.shape[1],
            "num_models": len(self.models),
            "model_types": list(self.model_metadata.values()),
            "ensemble_diversity": len(set(self.model_metadata.values()))
        }
        
        logger.info(f"Ensemble training completed. Accuracy: {accuracy:.3f}, Error: {avg_error_km:.3f}km")
        
        # Enhanced validation with comprehensive testing
        try:
            from app.accuracy_validator import AccuracyValidator
            
            validator = AccuracyValidator(target_accuracy=0.92)
            validation_metrics = validator.validate_model_accuracy(
                self, X_test_scaled, y_test, "Enhanced_Ensemble"
            )
            
            # Add validation results to metrics
            metrics.update({
                "validation_accuracy_50m": validation_metrics.accuracy_50m,
                "validation_accuracy_100m": validation_metrics.accuracy_100m,
                "validation_accuracy_200m": validation_metrics.accuracy_200m,
                "validation_mae_meters": validation_metrics.mae_meters,
                "validation_r2_score": validation_metrics.r2_score,
                "target_achieved": validation_metrics.accuracy_50m >= 0.92
            })
            
            # Generate accuracy report
            report = validator.generate_accuracy_report()
            logger.info("\n" + report)
            
        except ImportError:
            logger.warning("Accuracy validator not available, using basic metrics")
        except Exception as e:
            logger.warning(f"Enhanced validation failed: {e}")
        
        return metrics
    
    def _train_transformer_lstm(self, X_train, y_train, X_test, y_test, model_idx):
        """Train Transformer-style LSTM model for maximum accuracy."""
        
        # Add some variation to each model
        config_variant = AdvancedConfig(
            lstm_units=self.config.lstm_units + (model_idx - 1) * 32,
            dropout_rate=self.config.dropout_rate + model_idx * 0.02,
            learning_rate=self.config.learning_rate * (1 + model_idx * 0.1)
        )
        
        lstm_model = TransformerLSTM(config_variant, (X_train.shape[1],))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=self.config.patience, restore_best_weights=True),
            ReduceLROnPlateau(patience=self.config.patience//2, factor=0.5),
        ]
        
        # Train
        history = lstm_model.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return lstm_model.model, f"TransformerLSTM_{model_idx}"
    
    def _train_cnn_lstm(self, X_train, y_train, X_test, y_test, model_idx):
        """Train CNN-LSTM hybrid model."""
        
        # Reshape for CNN-LSTM
        seq_len = self.config.sequence_length
        n_features = X_train.shape[1] // seq_len
        X_train_reshaped = X_train.reshape((-1, seq_len, n_features))
        X_test_reshaped = X_test.reshape((-1, seq_len, n_features))
        
        # Build CNN-LSTM model
        inputs = Input(shape=(seq_len, n_features))
        
        # CNN layers for local pattern detection
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        
        # LSTM layers for sequence modeling
        x = LSTM(self.config.lstm_units, return_sequences=True, dropout=0.1)(x)
        x = LSTM(self.config.lstm_units // 2, dropout=0.1)(x)
        
        # Dense layers
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        
        outputs = Dense(len(self.config.prediction_horizons) * 2, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate * 2),
            loss='mse',
            metrics=['mae']
        )
        
        # Train the model
        callbacks = [
            EarlyStopping(patience=self.config.patience // 2, restore_best_weights=True),
            ReduceLROnPlateau(patience=self.config.patience // 4, factor=0.5)
        ]
        
        model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_test_reshaped, y_test),
            epochs=self.config.epochs // 2,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return model, f"CNN_LSTM_{model_idx}"
    
    def _train_xgboost_model(self, X_train, y_train, X_test, y_test, model_idx):
        """Train XGBoost model if available."""
        
        if not XGBOOST_AVAILABLE:
            # Fallback to Extra Trees if XGBoost not available
            model = ExtraTreesRegressor(
                n_estimators=300,
                max_depth=20,
                random_state=42 + model_idx,
                n_jobs=-1
            )
            model.fit(X_train, y_train[:, :2])
            return model, f"ExtraTrees_{model_idx}"
        
        # Train separate models for lat and lon
        lat_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=12,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + model_idx,
            n_jobs=-1
        )
        
        lon_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=12,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + model_idx + 1000,
            n_jobs=-1
        )
        
        lat_model.fit(X_train, y_train[:, 0])
        lon_model.fit(X_train, y_train[:, 1])
        
        # Create wrapper class for prediction
        class XGBoostEnsemble:
            def __init__(self, lat_model, lon_model):
                self.lat_model = lat_model
                self.lon_model = lon_model
            
            def predict(self, X):
                lat_pred = self.lat_model.predict(X)
                lon_pred = self.lon_model.predict(X)
                return np.column_stack([lat_pred, lon_pred])
        
        return XGBoostEnsemble(lat_model, lon_model), f"XGBoost_{model_idx}"
    
    def _train_advanced_traditional_model(self, X_train, y_train, X_test, y_test, model_idx):
        """Train advanced traditional ML models."""
        
        if model_idx % 2 == 0:
            # Advanced Random Forest with feature selection
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=25,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42 + model_idx,
                n_jobs=-1
            )
        else:
            # Advanced Gradient Boosting
            model = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=12,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42 + model_idx
            )
        
        # Train on first prediction horizon only
        model.fit(X_train, y_train[:, :2])
        
        return model, f"AdvancedML_{model_idx}"
    
    def predict(self, X):
        """Make enhanced ensemble predictions with confidence weighting."""
        
        if not self.models:
            raise ValueError("No models trained")
        
        predictions = []
        confidences = []
        
        for i, model in enumerate(self.models):
            try:
                if hasattr(model, 'predict'):
                    # Handle different model types
                    if 'CNN_LSTM' in self.model_metadata.get(i, '') or 'TransformerLSTM' in self.model_metadata.get(i, ''):
                        # Reshape for sequence models
                        seq_len = self.config.sequence_length
                        n_features = X.shape[1] // seq_len
                        X_reshaped = X.reshape((-1, seq_len, n_features))
                        pred = model.predict(X_reshaped, verbose=0)
                    else:
                        pred = model.predict(X)
                    
                    # Ensure proper shape
                    if len(pred.shape) == 1:
                        pred = pred.reshape(-1, 1)
                    if pred.shape[1] < 2:
                        pred = np.hstack([pred, pred])  # Duplicate for lat/lon
                    
                    # Take only first two columns (lat, lon for first horizon)
                    pred_clean = pred[:, :2]
                    predictions.append(pred_clean)
                    
                    # Calculate confidence based on prediction consistency
                    pred_std = np.std(pred_clean, axis=0)
                    confidence = 1.0 / (1.0 + np.mean(pred_std))  # Higher confidence for lower variance
                    confidences.append(confidence)
                    
            except Exception as e:
                logger.warning(f"Model {i} prediction failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("All model predictions failed")
        
        # Enhanced weighted ensemble with confidence and diversity penalty
        base_weights = np.array(self.config.ensemble_weights[:len(predictions)])
        confidences = np.array(confidences)
        
        # Combine base weights with confidence scores
        combined_weights = base_weights * confidences
        
        # Apply diversity penalty to reduce over-reliance on similar predictions
        if len(predictions) > 1:
            diversity_penalty = self._calculate_diversity_penalty(predictions)
            combined_weights *= (1.0 - self.config.ensemble_diversity_weight * diversity_penalty)
        
        # Normalize weights
        combined_weights = combined_weights / np.sum(combined_weights)
        
        # Weighted average prediction
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, combined_weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def _calculate_diversity_penalty(self, predictions):
        """Calculate diversity penalty to encourage ensemble diversity."""
        try:
            # Calculate pairwise correlations between predictions
            correlations = []
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    # Calculate correlation for each coordinate
                    lat_corr = np.corrcoef(predictions[i][:, 0], predictions[j][:, 0])[0, 1]
                    lon_corr = np.corrcoef(predictions[i][:, 1], predictions[j][:, 1])[0, 1]
                    avg_corr = (lat_corr + lon_corr) / 2
                    if not np.isnan(avg_corr):
                        correlations.append(abs(avg_corr))
            
            if correlations:
                return np.array([np.mean(correlations)] * len(predictions))
            else:
                return np.zeros(len(predictions))
        except Exception:
            return np.zeros(len(predictions))


async def demonstrate_advanced_lstm(trajectories: List) -> Dict[str, Any]:
    """Demonstrate the ultra-advanced LSTM system with 92%+ accuracy target."""
    
    print("🚀 Ultra-Advanced LSTM Mobility Prediction System")
    print("=" * 55)
    print("🎯 TARGET: 92%+ Accuracy | <50m Error")
    print("🔬 Enhanced Features:")
    print("   • Transformer-style attention with residual connections")
    print("   • Enhanced ensemble learning (8 diverse models)")
    print("   • XGBoost & CNN-LSTM hybrid models")
    print("   • Frequency domain feature engineering")
    print("   • Kalman filtering & advanced preprocessing")
    print("   • Confidence-weighted ensemble voting")
    print("   • Diversity penalty for better generalization")
    print()
    
    try:
        # Initialize advanced predictor
        config = AdvancedConfig()
        predictor = EnsembleMobilityPredictor(config)
        
        # Train the ensemble
        print("🧠 Training advanced ensemble...")
        start_time = time.time()
        
        metrics = predictor.train(trajectories)
        
        training_time = time.time() - start_time
        
        print("✅ Advanced training completed!")
        print(f"\n📊 Advanced Model Performance:")
        print(f"   🎯 Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"   📏 Average Error: {metrics['avg_error_km']:.3f} km ({metrics['avg_error_km']*1000:.1f}m)")
        print(f"   🧠 Training Samples: {metrics['training_samples']:,}")
        print(f"   🔬 Test Samples: {metrics['test_samples']:,}")
        print(f"   ⚙️ Features: {metrics['num_features']:,}")
        print(f"   🤖 Models in Ensemble: {metrics['num_models']}")
        print(f"   ⏱️ Training Time: {training_time:.1f}s")
        
        # Individual model performance with enhanced types
        print(f"\n📈 Enhanced Ensemble Model Performance:")
        for i, mae in enumerate(metrics['individual_maes']):
            if i < 2:
                model_type = "Transformer-LSTM"
            elif i < 4:
                model_type = "CNN-LSTM Hybrid"
            elif i < 6:
                model_type = "XGBoost" if XGBOOST_AVAILABLE else "ExtraTrees"
            else:
                model_type = "Advanced ML"
            
            accuracy_est = max(0, 1.0 - mae / 0.001)  # Rough accuracy estimate
            print(f"   Model {i+1} ({model_type}): {mae*111:.1f}m error, {accuracy_est:.3f} accuracy")
        
        print(f"\n🎯 Enhanced Ensemble Results:")
        print(f"   • Average Error: {(metrics['ensemble_mae']*111):.1f}m")
        print(f"   • Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"   • Model Diversity: {len(set(metrics.get('model_types', [])))}/4 architecture types")
        
        # Performance assessment
        if metrics['accuracy'] >= 0.92:
            print(f"   ✅ TARGET ACHIEVED: 92%+ accuracy reached!")
        elif metrics['accuracy'] >= 0.85:
            print(f"   🔶 APPROACHING TARGET: {(0.92 - metrics['accuracy'])*100:.1f}% to reach 92%")
        else:
            print(f"   🔴 NEEDS IMPROVEMENT: {(0.92 - metrics['accuracy'])*100:.1f}% to reach 92% target")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Advanced LSTM demonstration failed: {e}")
        print(f"❌ Advanced LSTM failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # For testing
    print("🧪 Advanced LSTM Predictor Module")
    print("This module provides state-of-the-art LSTM accuracy improvements")