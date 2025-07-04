"""
Real-World Dataset Integration for ZKPAS System

This module loads and processes real mobility datasets (Geolife and Beijing Taxi)
to replace synthetic data with authentic IoT device movement patterns.

Features:
- Load Geolife Trajectories 1.3 (Microsoft Research)
- Load Beijing Taxi Logs 2008
- Convert to standardized LocationPoint format
- Apply privacy-preserving transformations
- Cache processed data for performance
- Support for different sampling strategies
"""

import os
import pandas as pd
import numpy as np
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

from app.mobility_predictor import LocationPoint, MobilityPattern
from app.events import EventBus, EventType

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    
    # Dataset paths (relative to project root)
    geolife_path: str = "../Datasets/Geolife Trajectories 1.3/Data"
    taxi_path: str = "../Datasets/release/taxi_log_2008_by_id"
    
    # Processing parameters
    max_users: Optional[int] = None  # None = load all users
    max_trajectories_per_user: int = 100
    min_trajectory_points: int = 10
    temporal_resolution_seconds: int = 30  # Resample to 30-second intervals
    
    # Privacy settings
    apply_noise: bool = True
    noise_level: float = 0.001  # GPS coordinate noise (degrees)
    k_anonymity: int = 5
    
    # Cache settings
    cache_processed_data: bool = True
    cache_dir: str = "data/processed_datasets"


@dataclass
class TrajectoryData:
    """Container for processed trajectory data."""
    
    user_id: str
    trajectory_id: str
    points: List[LocationPoint]
    start_time: float
    end_time: float
    duration_seconds: float
    total_distance_km: float
    avg_speed_kmh: float
    mobility_pattern: Optional[MobilityPattern] = None


class DatasetLoader:
    """Loads and processes real-world mobility datasets for ZKPAS system."""
    
    def __init__(self, config: DatasetConfig, event_bus: Optional[EventBus] = None):
        """
        Initialize dataset loader.
        
        Args:
            config: Dataset loading configuration
            event_bus: Optional event bus for notifications
        """
        self.config = config
        self.event_bus = event_bus
        
        # Setup paths
        self.project_root = Path(__file__).parent.parent
        self.geolife_path = self.project_root / config.geolife_path
        self.taxi_path = self.project_root / config.taxi_path
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.trajectories: Dict[str, List[TrajectoryData]] = {}
        self.dataset_stats = {}
        
        logger.info(f"Dataset loader initialized with cache dir: {self.cache_dir}")
    
    async def load_all_datasets(self) -> Dict[str, List[TrajectoryData]]:
        """
        Load all available datasets.
        
        Returns:
            Dictionary mapping dataset names to trajectory lists
        """
        logger.info("Loading all real-world mobility datasets...")
        
        datasets = {}
        
        # Load Geolife dataset
        if self.geolife_path.exists():
            logger.info("Loading Geolife Trajectories dataset...")
            datasets["geolife"] = await self.load_geolife_dataset()
            logger.info(f"Loaded {len(datasets['geolife'])} Geolife trajectories")
        else:
            logger.warning(f"Geolife dataset not found at: {self.geolife_path}")
        
        # Load Taxi dataset
        if self.taxi_path.exists():
            logger.info("Loading Beijing Taxi dataset...")
            datasets["taxi"] = await self.load_taxi_dataset()
            logger.info(f"Loaded {len(datasets['taxi'])} taxi trajectories")
        else:
            logger.warning(f"Taxi dataset not found at: {self.taxi_path}")
        
        # Publish event if event bus available
        if self.event_bus:
            await self.event_bus.publish_event(
                EventType.COMPONENT_STARTED,
                correlation_id="dataset_loader",
                source="DatasetLoader",
                data={
                    "datasets_loaded": list(datasets.keys()),
                    "total_trajectories": sum(len(trajs) for trajs in datasets.values())
                }
            )
        
        self.trajectories = datasets
        return datasets
    
    async def load_geolife_dataset(self) -> List[TrajectoryData]:
        """
        Load Microsoft Geolife Trajectories 1.3 dataset.
        
        Returns:
            List of processed trajectory data
        """
        cache_file = self.cache_dir / "geolife_processed.pkl"
        
        # Try to load from cache first
        if self.config.cache_processed_data and cache_file.exists():
            logger.info("Loading Geolife dataset from cache...")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}. Reprocessing...")
        
        trajectories = []
        
        # Get all user directories
        user_dirs = [d for d in self.geolife_path.iterdir() if d.is_dir()]
        if self.config.max_users:
            user_dirs = user_dirs[:self.config.max_users]
        
        logger.info(f"Processing {len(user_dirs)} Geolife users...")
        
        for user_dir in user_dirs:
            user_id = user_dir.name
            trajectory_dir = user_dir / "Trajectory"
            
            if not trajectory_dir.exists():
                continue
            
            # Load trajectory files for this user
            plt_files = list(trajectory_dir.glob("*.plt"))
            if self.config.max_trajectories_per_user:
                plt_files = plt_files[:self.config.max_trajectories_per_user]
            
            for plt_file in plt_files:
                try:
                    trajectory = await self._load_geolife_trajectory(user_id, plt_file)
                    if trajectory and len(trajectory.points) >= self.config.min_trajectory_points:
                        trajectories.append(trajectory)
                except Exception as e:
                    logger.warning(f"Failed to load trajectory {plt_file}: {e}")
        
        # Apply privacy transformations
        if self.config.apply_noise:
            trajectories = self._apply_privacy_transformations(trajectories)
        
        # Cache processed data
        if self.config.cache_processed_data:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(trajectories, f)
                logger.info(f"Cached processed Geolife data to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
        
        return trajectories
    
    async def _load_geolife_trajectory(self, user_id: str, plt_file: Path) -> Optional[TrajectoryData]:
        """
        Load a single Geolife .plt trajectory file.
        
        Args:
            user_id: User identifier
            plt_file: Path to .plt file
            
        Returns:
            Processed trajectory data or None if invalid
        """
        try:
            # Read PLT file (skip first 6 header lines)
            df = pd.read_csv(plt_file, skiprows=6, header=None,
                           names=['latitude', 'longitude', 'reserved', 'altitude', 
                                 'date_days', 'date_str', 'time_str'])
            
            if len(df) < self.config.min_trajectory_points:
                return None
            
            # Parse timestamps
            df['datetime'] = pd.to_datetime(df['date_str'] + ' ' + df['time_str'])
            df['timestamp'] = df['datetime'].astype(np.int64) // 10**9
            
            # Convert altitude from feet to meters
            df['altitude_m'] = df['altitude'] * 0.3048
            
            # Create LocationPoint objects
            points = []
            for _, row in df.iterrows():
                point = LocationPoint(
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude']),
                    timestamp=float(row['timestamp']),
                    altitude=float(row['altitude_m']) if pd.notna(row['altitude_m']) else None
                )
                points.append(point)
            
            # Resample to uniform time intervals if requested
            if self.config.temporal_resolution_seconds > 0:
                points = self._resample_trajectory(points, self.config.temporal_resolution_seconds)
            
            # Calculate trajectory statistics
            start_time = points[0].timestamp
            end_time = points[-1].timestamp
            duration = end_time - start_time
            
            # Calculate total distance
            total_distance = self._calculate_total_distance(points)
            avg_speed = (total_distance / duration * 3600) if duration > 0 else 0  # km/h
            
            # Classify mobility pattern
            mobility_pattern = self._classify_mobility_pattern(points, avg_speed, total_distance)
            
            trajectory = TrajectoryData(
                user_id=user_id,
                trajectory_id=plt_file.stem,
                points=points,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                total_distance_km=total_distance,
                avg_speed_kmh=avg_speed,
                mobility_pattern=mobility_pattern
            )
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Error loading trajectory {plt_file}: {e}")
            return None
    
    async def load_taxi_dataset(self) -> List[TrajectoryData]:
        """
        Load Beijing Taxi dataset.
        
        Returns:
            List of processed trajectory data
        """
        cache_file = self.cache_dir / "taxi_processed.pkl"
        
        # Try to load from cache first
        if self.config.cache_processed_data and cache_file.exists():
            logger.info("Loading taxi dataset from cache...")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}. Reprocessing...")
        
        trajectories = []
        
        # Get all taxi log files
        txt_files = list(self.taxi_path.glob("*.txt"))
        if self.config.max_users:
            txt_files = txt_files[:self.config.max_users]
        
        logger.info(f"Processing {len(txt_files)} taxi trajectories...")
        
        for txt_file in txt_files:
            try:
                trajectory = await self._load_taxi_trajectory(txt_file)
                if trajectory and len(trajectory.points) >= self.config.min_trajectory_points:
                    trajectories.append(trajectory)
            except Exception as e:
                logger.warning(f"Failed to load taxi trajectory {txt_file}: {e}")
        
        # Apply privacy transformations
        if self.config.apply_noise:
            trajectories = self._apply_privacy_transformations(trajectories)
        
        # Cache processed data
        if self.config.cache_processed_data:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(trajectories, f)
                logger.info(f"Cached processed taxi data to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
        
        return trajectories
    
    async def _load_taxi_trajectory(self, txt_file: Path) -> Optional[TrajectoryData]:
        """
        Load a single taxi trajectory file.
        
        Args:
            txt_file: Path to taxi log file
            
        Returns:
            Processed trajectory data or None if invalid
        """
        try:
            # Read taxi log file
            df = pd.read_csv(txt_file, header=None,
                           names=['taxi_id', 'timestamp_str', 'longitude', 'latitude'])
            
            if len(df) < self.config.min_trajectory_points:
                return None
            
            # Parse timestamps
            df['datetime'] = pd.to_datetime(df['timestamp_str'])
            df['timestamp'] = df['datetime'].astype(np.int64) // 10**9
            
            # Create LocationPoint objects
            points = []
            for _, row in df.iterrows():
                point = LocationPoint(
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude']),
                    timestamp=float(row['timestamp'])
                )
                points.append(point)
            
            # Sort by timestamp
            points.sort(key=lambda p: p.timestamp)
            
            # Resample if requested
            if self.config.temporal_resolution_seconds > 0:
                points = self._resample_trajectory(points, self.config.temporal_resolution_seconds)
            
            # Calculate statistics
            taxi_id = str(df['taxi_id'].iloc[0])
            start_time = points[0].timestamp
            end_time = points[-1].timestamp
            duration = end_time - start_time
            
            total_distance = self._calculate_total_distance(points)
            avg_speed = (total_distance / duration * 3600) if duration > 0 else 0
            
            # Classify mobility pattern (taxis are typically VEHICLE pattern)
            mobility_pattern = self._classify_mobility_pattern(points, avg_speed, total_distance)
            
            trajectory = TrajectoryData(
                user_id=taxi_id,
                trajectory_id=f"{taxi_id}_{txt_file.stem}",
                points=points,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                total_distance_km=total_distance,
                avg_speed_kmh=avg_speed,
                mobility_pattern=mobility_pattern
            )
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Error loading taxi trajectory {txt_file}: {e}")
            return None
    
    def _resample_trajectory(self, points: List[LocationPoint], interval_seconds: int) -> List[LocationPoint]:
        """
        Resample trajectory to uniform time intervals.
        
        Args:
            points: Original trajectory points
            interval_seconds: Target time interval
            
        Returns:
            Resampled trajectory points
        """
        if len(points) < 2:
            return points
        
        # Create DataFrame for easier resampling
        df = pd.DataFrame([{
            'timestamp': p.timestamp,
            'latitude': p.latitude,
            'longitude': p.longitude,
            'altitude': p.altitude
        } for p in points])
        
        # Set timestamp as index
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        
        # Resample to target interval
        resampled = df.resample(f'{interval_seconds}S').mean().interpolate()
        
        # Convert back to LocationPoint objects
        resampled_points = []
        for idx, row in resampled.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                point = LocationPoint(
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude']),
                    timestamp=float(idx.timestamp()),
                    altitude=float(row['altitude']) if pd.notna(row['altitude']) else None
                )
                resampled_points.append(point)
        
        return resampled_points
    
    def _calculate_total_distance(self, points: List[LocationPoint]) -> float:
        """
        Calculate total distance of trajectory using Haversine formula.
        
        Args:
            points: Trajectory points
            
        Returns:
            Total distance in kilometers
        """
        if len(points) < 2:
            return 0.0
        
        total_distance = 0.0
        
        for i in range(1, len(points)):
            p1 = points[i-1]
            p2 = points[i]
            
            # Haversine formula
            R = 6371  # Earth radius in km
            
            lat1, lon1 = np.radians(p1.latitude), np.radians(p1.longitude)
            lat2, lon2 = np.radians(p2.latitude), np.radians(p2.longitude)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            
            distance = R * c
            total_distance += distance
        
        return total_distance
    
    def _classify_mobility_pattern(self, points: List[LocationPoint], avg_speed: float, 
                                 total_distance: float) -> MobilityPattern:
        """
        Classify mobility pattern based on trajectory characteristics.
        
        Args:
            points: Trajectory points
            avg_speed: Average speed in km/h
            total_distance: Total distance in km
            
        Returns:
            Classified mobility pattern
        """
        if len(points) < 10:
            return MobilityPattern.RANDOM
        
        # Calculate movement statistics
        distance_threshold_stationary = 0.5  # km
        speed_threshold_vehicle = 25  # km/h
        
        if total_distance < distance_threshold_stationary:
            return MobilityPattern.STATIONARY
        elif avg_speed > speed_threshold_vehicle:
            return MobilityPattern.VEHICLE
        else:
            # Analyze periodicity (simplified)
            if self._has_periodic_pattern(points):
                return MobilityPattern.PERIODIC
            else:
                return MobilityPattern.RANDOM
    
    def _has_periodic_pattern(self, points: List[LocationPoint]) -> bool:
        """
        Simple heuristic to detect periodic patterns.
        
        Args:
            points: Trajectory points
            
        Returns:
            True if trajectory shows periodic behavior
        """
        # This is a simplified implementation
        # In practice, you'd use more sophisticated time series analysis
        
        if len(points) < 50:
            return False
        
        # Check if device returns to similar locations
        start_point = points[0]
        end_point = points[-1]
        
        # Calculate distance between start and end
        distance = self._calculate_total_distance([start_point, end_point])
        
        # If start and end are close, might be periodic
        return distance < 1.0  # Within 1km
    
    def _apply_privacy_transformations(self, trajectories: List[TrajectoryData]) -> List[TrajectoryData]:
        """
        Apply privacy-preserving transformations to trajectories.
        
        Args:
            trajectories: Original trajectories
            
        Returns:
            Privacy-protected trajectories
        """
        logger.info(f"Applying privacy transformations to {len(trajectories)} trajectories...")
        
        for trajectory in trajectories:
            # Add Gaussian noise to coordinates
            for point in trajectory.points:
                noise_lat = np.random.normal(0, self.config.noise_level)
                noise_lon = np.random.normal(0, self.config.noise_level)
                
                point.latitude += noise_lat
                point.longitude += noise_lon
        
        return trajectories
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about loaded datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        for dataset_name, trajectories in self.trajectories.items():
            if not trajectories:
                continue
            
            total_points = sum(len(traj.points) for traj in trajectories)
            total_distance = sum(traj.total_distance_km for traj in trajectories)
            total_duration = sum(traj.duration_seconds for traj in trajectories)
            
            avg_speed = np.mean([traj.avg_speed_kmh for traj in trajectories])
            avg_duration = np.mean([traj.duration_seconds for traj in trajectories])
            
            # Mobility pattern distribution
            pattern_counts = {}
            for traj in trajectories:
                pattern = traj.mobility_pattern
                if pattern:
                    pattern_counts[pattern.name] = pattern_counts.get(pattern.name, 0) + 1
            
            stats[dataset_name] = {
                "num_trajectories": len(trajectories),
                "num_users": len(set(traj.user_id for traj in trajectories)),
                "total_points": total_points,
                "total_distance_km": total_distance,
                "total_duration_hours": total_duration / 3600,
                "avg_speed_kmh": avg_speed,
                "avg_duration_minutes": avg_duration / 60,
                "mobility_patterns": pattern_counts
            }
        
        return stats
    
    def get_trajectories_for_training(self, dataset_name: Optional[str] = None, 
                                    max_trajectories: Optional[int] = None) -> List[TrajectoryData]:
        """
        Get trajectories suitable for training ML models.
        
        Args:
            dataset_name: Specific dataset to use, or None for all
            max_trajectories: Maximum number of trajectories to return
            
        Returns:
            List of trajectories for training
        """
        if dataset_name and dataset_name in self.trajectories:
            trajectories = self.trajectories[dataset_name]
        else:
            # Combine all datasets
            trajectories = []
            for trajs in self.trajectories.values():
                trajectories.extend(trajs)
        
        # Filter trajectories suitable for training
        suitable_trajectories = [
            traj for traj in trajectories
            if len(traj.points) >= 20 and  # Minimum points for training
               traj.duration_seconds >= 300 and  # At least 5 minutes
               traj.total_distance_km > 0.1  # Some movement
        ]
        
        # Limit number if requested
        if max_trajectories:
            suitable_trajectories = suitable_trajectories[:max_trajectories]
        
        logger.info(f"Selected {len(suitable_trajectories)} trajectories for training")
        return suitable_trajectories
    
    def export_to_pandas(self, dataset_name: Optional[str] = None) -> pd.DataFrame:
        """
        Export trajectory data to pandas DataFrame for analysis.
        
        Args:
            dataset_name: Specific dataset to export, or None for all
            
        Returns:
            DataFrame with all trajectory points
        """
        if dataset_name and dataset_name in self.trajectories:
            trajectories = self.trajectories[dataset_name]
        else:
            trajectories = []
            for trajs in self.trajectories.values():
                trajectories.extend(trajs)
        
        rows = []
        for traj in trajectories:
            for point in traj.points:
                row = {
                    'user_id': traj.user_id,
                    'trajectory_id': traj.trajectory_id,
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'timestamp': point.timestamp,
                    'altitude': point.altitude,
                    'mobility_pattern': traj.mobility_pattern.name if traj.mobility_pattern else None,
                    'avg_speed_kmh': traj.avg_speed_kmh
                }
                rows.append(row)
        
        return pd.DataFrame(rows)


# Utility functions for integration with existing system

def get_default_dataset_config() -> DatasetConfig:
    """Get default configuration for dataset loading."""
    return DatasetConfig(
        max_users=50,  # Start with subset for faster loading
        max_trajectories_per_user=10,
        min_trajectory_points=20,
        temporal_resolution_seconds=60,  # 1-minute intervals
        apply_noise=True,
        noise_level=0.0001,  # Small noise for privacy
        cache_processed_data=True
    )


async def load_real_mobility_data(config: Optional[DatasetConfig] = None,
                                event_bus: Optional[EventBus] = None) -> Tuple[DatasetLoader, Dict[str, List[TrajectoryData]]]:
    """
    Convenience function to load real mobility datasets.
    
    Args:
        config: Optional configuration, uses default if None
        event_bus: Optional event bus for notifications
        
    Returns:
        Tuple of (loader instance, loaded datasets)
    """
    if config is None:
        config = get_default_dataset_config()
    
    loader = DatasetLoader(config, event_bus)
    datasets = await loader.load_all_datasets()
    
    return loader, datasets


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        print("🗂️  Loading real-world mobility datasets...")
        
        config = get_default_dataset_config()
        loader, datasets = await load_real_mobility_data(config)
        
        # Print statistics
        stats = loader.get_dataset_statistics()
        print("\n📊 Dataset Statistics:")
        for dataset_name, stat in stats.items():
            print(f"\n{dataset_name.upper()} Dataset:")
            print(f"  Trajectories: {stat['num_trajectories']}")
            print(f"  Users: {stat['num_users']}")
            print(f"  Total Points: {stat['total_points']:,}")
            print(f"  Total Distance: {stat['total_distance_km']:.1f} km")
            print(f"  Average Speed: {stat['avg_speed_kmh']:.1f} km/h")
            print(f"  Mobility Patterns: {stat['mobility_patterns']}")
    
    asyncio.run(main())