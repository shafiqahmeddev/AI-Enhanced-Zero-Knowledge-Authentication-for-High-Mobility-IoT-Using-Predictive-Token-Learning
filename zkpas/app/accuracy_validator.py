#!/usr/bin/env python3
"""
Ultra-High Accuracy Validation Framework for ZKPAS LSTM Models

This module provides comprehensive testing and validation to ensure models
achieve the target 92%+ accuracy with robust evaluation metrics.

Features:
- Multiple accuracy thresholds (50m, 100m, 200m)
- Cross-validation with time-series aware splits
- Real-world scenario testing
- Performance benchmarking
- Statistical significance testing
- Model comparison framework

Author: Shafiq Ahmed <s.ahmed@essex.ac.uk>
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics for model evaluation."""
    
    # Primary accuracy metrics (different distance thresholds)
    accuracy_50m: float = 0.0   # Target: 92%+
    accuracy_100m: float = 0.0  # Secondary metric
    accuracy_200m: float = 0.0  # Fallback metric
    
    # Error metrics
    mae_degrees: float = 0.0
    rmse_degrees: float = 0.0
    mae_meters: float = 0.0
    rmse_meters: float = 0.0
    
    # Statistical metrics
    r2_score: float = 0.0
    mean_error: float = 0.0
    std_error: float = 0.0
    
    # Performance metrics
    prediction_time_ms: float = 0.0
    training_time_seconds: float = 0.0
    
    # Validation metrics
    cv_accuracy_mean: float = 0.0
    cv_accuracy_std: float = 0.0
    
    # Test scenarios
    scenario_accuracies: Dict[str, float] = None
    
    def __post_init__(self):
        if self.scenario_accuracies is None:
            self.scenario_accuracies = {}


class AccuracyValidator:
    """Comprehensive accuracy validation framework."""
    
    def __init__(self, target_accuracy: float = 0.92):
        """
        Initialize accuracy validator.
        
        Args:
            target_accuracy: Target accuracy threshold (default: 92%)
        """
        self.target_accuracy = target_accuracy
        self.validation_results = {}
        self.benchmarks = {}
        
    def validate_model_accuracy(self, model, X_test: np.ndarray, y_test: np.ndarray,
                               model_name: str = "model") -> AccuracyMetrics:
        """
        Comprehensive model accuracy validation.
        
        Args:
            model: Trained model to validate
            X_test: Test features
            y_test: Test targets
            model_name: Name for logging and results
            
        Returns:
            AccuracyMetrics object with comprehensive results
        """
        logger.info(f"🔍 Validating accuracy for {model_name}...")
        
        # Make predictions with timing
        start_time = time.time()
        try:
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                raise ValueError(f"Model {model_name} does not have predict method")
            
            prediction_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Handle different prediction formats
            if len(y_pred.shape) == 1:
                y_pred = y_pred.reshape(-1, 1)
            if y_pred.shape[1] < 2:
                y_pred = np.hstack([y_pred, y_pred])  # Duplicate for lat/lon
            
            # Ensure we're working with first 2 dimensions (lat, lon)
            y_pred_clean = y_pred[:, :2]
            y_test_clean = y_test[:, :2]
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for {model_name}: {e}")
            return AccuracyMetrics()
        
        # Calculate distance errors
        distance_errors = self._calculate_distance_errors(y_test_clean, y_pred_clean)
        
        # Calculate accuracy at different thresholds
        accuracy_50m = np.mean(distance_errors < 50)    # 50m threshold
        accuracy_100m = np.mean(distance_errors < 100)  # 100m threshold  
        accuracy_200m = np.mean(distance_errors < 200)  # 200m threshold
        
        # Calculate error metrics
        mae_degrees = mean_absolute_error(y_test_clean, y_pred_clean)
        rmse_degrees = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
        mae_meters = np.mean(distance_errors)
        rmse_meters = np.sqrt(np.mean(distance_errors**2))
        
        # Calculate R² score
        r2_lat = r2_score(y_test_clean[:, 0], y_pred_clean[:, 0])
        r2_lon = r2_score(y_test_clean[:, 1], y_pred_clean[:, 1])
        r2_avg = (r2_lat + r2_lon) / 2
        
        # Statistical metrics
        mean_error = np.mean(distance_errors)
        std_error = np.std(distance_errors)
        
        # Create metrics object
        metrics = AccuracyMetrics(
            accuracy_50m=accuracy_50m,
            accuracy_100m=accuracy_100m,
            accuracy_200m=accuracy_200m,
            mae_degrees=mae_degrees,
            rmse_degrees=rmse_degrees,
            mae_meters=mae_meters,
            rmse_meters=rmse_meters,
            r2_score=r2_avg,
            mean_error=mean_error,
            std_error=std_error,
            prediction_time_ms=prediction_time / len(X_test)  # Per sample
        )
        
        # Store results
        self.validation_results[model_name] = metrics
        
        # Log results
        self._log_validation_results(model_name, metrics)
        
        return metrics
    
    def cross_validate_accuracy(self, model_class, X: np.ndarray, y: np.ndarray,
                               config=None, n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform time-series aware cross-validation.
        
        Args:
            model_class: Model class to instantiate
            X: Features
            y: Targets
            config: Model configuration
            n_splits: Number of CV splits
            
        Returns:
            Cross-validation results
        """
        logger.info(f"🔄 Performing {n_splits}-fold time-series cross-validation...")
        
        # Time series split (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_accuracies = []
        cv_maes = []
        cv_times = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"   Fold {fold + 1}/{n_splits}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            try:
                # Train model for this fold
                start_time = time.time()
                
                if config:
                    model = model_class(config)
                else:
                    model = model_class()
                
                # Train the model (implementation depends on model type)
                if hasattr(model, 'fit'):
                    model.fit(X_train_fold, y_train_fold)
                elif hasattr(model, 'train'):
                    model.train(X_train_fold, y_train_fold)
                
                training_time = time.time() - start_time
                
                # Validate on fold
                metrics = self.validate_model_accuracy(model, X_val_fold, y_val_fold, f"CV_fold_{fold}")
                
                cv_accuracies.append(metrics.accuracy_50m)
                cv_maes.append(metrics.mae_meters)
                cv_times.append(training_time)
                
            except Exception as e:
                logger.warning(f"⚠️ Fold {fold + 1} failed: {e}")
                continue
        
        if cv_accuracies:
            cv_results = {
                "mean_accuracy": np.mean(cv_accuracies),
                "std_accuracy": np.std(cv_accuracies),
                "mean_mae": np.mean(cv_maes),
                "std_mae": np.std(cv_maes),
                "mean_training_time": np.mean(cv_times),
                "accuracies_per_fold": cv_accuracies,
                "successful_folds": len(cv_accuracies)
            }
            
            logger.info(f"✅ CV Results: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f} accuracy")
            return cv_results
        else:
            logger.error("❌ All CV folds failed")
            return {"mean_accuracy": 0.0, "std_accuracy": 0.0}
    
    def test_real_world_scenarios(self, model, test_scenarios: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """
        Test model on real-world scenarios.
        
        Args:
            model: Trained model
            test_scenarios: Dictionary of scenario_name -> (X_test, y_test)
            
        Returns:
            Dictionary of scenario accuracies
        """
        logger.info("🌍 Testing real-world scenarios...")
        
        scenario_results = {}
        
        for scenario_name, (X_test, y_test) in test_scenarios.items():
            try:
                metrics = self.validate_model_accuracy(model, X_test, y_test, f"scenario_{scenario_name}")
                scenario_results[scenario_name] = metrics.accuracy_50m
                
                logger.info(f"   📍 {scenario_name}: {metrics.accuracy_50m:.3f} accuracy")
                
            except Exception as e:
                logger.warning(f"⚠️ Scenario {scenario_name} failed: {e}")
                scenario_results[scenario_name] = 0.0
        
        return scenario_results
    
    def benchmark_against_baseline(self, model, baseline_model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Benchmark model against baseline.
        
        Args:
            model: Model to benchmark
            baseline_model: Baseline model for comparison
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Benchmark comparison results
        """
        logger.info("📊 Benchmarking against baseline...")
        
        # Validate both models
        model_metrics = self.validate_model_accuracy(model, X_test, y_test, "enhanced_model")
        baseline_metrics = self.validate_model_accuracy(baseline_model, X_test, y_test, "baseline_model")
        
        # Calculate improvements
        accuracy_improvement = model_metrics.accuracy_50m - baseline_metrics.accuracy_50m
        error_reduction = (baseline_metrics.mae_meters - model_metrics.mae_meters) / baseline_metrics.mae_meters
        
        # Statistical significance test
        model_errors = self._calculate_distance_errors(
            y_test[:, :2], 
            model.predict(X_test)[:, :2] if hasattr(model, 'predict') else np.zeros_like(y_test[:, :2])
        )
        baseline_errors = self._calculate_distance_errors(
            y_test[:, :2], 
            baseline_model.predict(X_test)[:, :2] if hasattr(baseline_model, 'predict') else np.zeros_like(y_test[:, :2])
        )
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(baseline_errors, model_errors)
        
        benchmark_results = {
            "enhanced_accuracy": model_metrics.accuracy_50m,
            "baseline_accuracy": baseline_metrics.accuracy_50m,
            "accuracy_improvement": accuracy_improvement,
            "enhanced_error": model_metrics.mae_meters,
            "baseline_error": baseline_metrics.mae_meters,
            "error_reduction_pct": error_reduction * 100,
            "statistical_significance": p_value < 0.05,
            "p_value": p_value,
            "improvement_factor": model_metrics.accuracy_50m / baseline_metrics.accuracy_50m if baseline_metrics.accuracy_50m > 0 else float('inf')
        }
        
        self._log_benchmark_results(benchmark_results)
        
        return benchmark_results
    
    def generate_accuracy_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive accuracy report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        logger.info("📋 Generating comprehensive accuracy report...")
        
        report_lines = [
            "🎯 ZKPAS LSTM Ultra-High Accuracy Validation Report",
            "=" * 60,
            f"📅 Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"🎯 Target Accuracy: {self.target_accuracy * 100:.1f}%",
            "",
            "📊 MODEL PERFORMANCE SUMMARY",
            "-" * 40
        ]
        
        # Results for each model
        for model_name, metrics in self.validation_results.items():
            status = "✅ TARGET ACHIEVED" if metrics.accuracy_50m >= self.target_accuracy else "❌ NEEDS IMPROVEMENT"
            
            report_lines.extend([
                f"\n🤖 {model_name.upper()}:",
                f"   📈 Accuracy (50m):  {metrics.accuracy_50m:.3f} ({metrics.accuracy_50m*100:.1f}%) {status}",
                f"   📈 Accuracy (100m): {metrics.accuracy_100m:.3f} ({metrics.accuracy_100m*100:.1f}%)",
                f"   📈 Accuracy (200m): {metrics.accuracy_200m:.3f} ({metrics.accuracy_200m*100:.1f}%)",
                f"   📏 Mean Error:      {metrics.mae_meters:.1f}m",
                f"   📏 RMSE Error:      {metrics.rmse_meters:.1f}m",
                f"   📊 R² Score:        {metrics.r2_score:.3f}",
                f"   ⚡ Prediction Time: {metrics.prediction_time_ms:.2f}ms/sample"
            ])
        
        # Overall assessment
        best_model = max(self.validation_results.items(), key=lambda x: x[1].accuracy_50m)
        report_lines.extend([
            "",
            "🏆 BEST MODEL ASSESSMENT",
            "-" * 30,
            f"🥇 Best Model: {best_model[0]}",
            f"🎯 Best Accuracy: {best_model[1].accuracy_50m:.3f} ({best_model[1].accuracy_50m*100:.1f}%)"
        ])
        
        if best_model[1].accuracy_50m >= self.target_accuracy:
            report_lines.extend([
                "",
                "🎉 SUCCESS: Target accuracy achieved!",
                f"✅ Exceeded target by {(best_model[1].accuracy_50m - self.target_accuracy)*100:.1f} percentage points",
                "🚀 Model ready for production deployment"
            ])
        else:
            gap = (self.target_accuracy - best_model[1].accuracy_50m) * 100
            report_lines.extend([
                "",
                "⚠️ IMPROVEMENT NEEDED",
                f"📊 Gap to target: {gap:.1f} percentage points",
                "🔧 Consider: More training data, hyperparameter tuning, ensemble methods"
            ])
        
        # Recommendations
        report_lines.extend([
            "",
            "💡 RECOMMENDATIONS",
            "-" * 20
        ])
        
        if best_model[1].accuracy_50m < 0.85:
            report_lines.append("🔴 Critical: Model needs significant improvement")
        elif best_model[1].accuracy_50m < 0.92:
            report_lines.append("🟡 Good: Close to target, minor improvements needed")
        else:
            report_lines.append("🟢 Excellent: Target achieved, ready for deployment")
        
        report = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"📄 Report saved to {output_path}")
        
        return report
    
    def _calculate_distance_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate distance errors in meters using Haversine formula."""
        try:
            # Ensure proper shape
            if y_true.shape != y_pred.shape:
                logger.warning(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
            
            # Extract lat/lon
            lat_true, lon_true = y_true[:, 0], y_true[:, 1]
            lat_pred, lon_pred = y_pred[:, 0], y_pred[:, 1]
            
            # Haversine formula
            dlat = np.radians(lat_pred - lat_true)
            dlon = np.radians(lon_pred - lon_true)
            
            a = (np.sin(dlat/2)**2 + 
                 np.cos(np.radians(lat_true)) * np.cos(np.radians(lat_pred)) * 
                 np.sin(dlon/2)**2)
            
            # Avoid numerical issues
            a = np.clip(a, 0, 1)
            c = 2 * np.arcsin(np.sqrt(a))
            
            # Earth radius in meters
            earth_radius = 6371000
            distances = earth_radius * c
            
            # Sanity check
            distances = np.clip(distances, 0, 20000000)  # Max 20,000km
            
            return distances
            
        except Exception as e:
            logger.error(f"Error calculating distances: {e}")
            # Fallback: simple Euclidean distance in degrees converted to meters
            euclidean_degrees = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
            return euclidean_degrees * 111000  # Rough conversion
    
    def _log_validation_results(self, model_name: str, metrics: AccuracyMetrics):
        """Log validation results."""
        status = "✅" if metrics.accuracy_50m >= self.target_accuracy else "❌"
        
        logger.info(f"📊 {model_name} Results:")
        logger.info(f"   🎯 Accuracy (50m): {metrics.accuracy_50m:.3f} ({metrics.accuracy_50m*100:.1f}%) {status}")
        logger.info(f"   📏 Mean Error: {metrics.mae_meters:.1f}m")
        logger.info(f"   📊 R² Score: {metrics.r2_score:.3f}")
    
    def _log_benchmark_results(self, results: Dict[str, Any]):
        """Log benchmark results."""
        logger.info("📊 Benchmark Results:")
        logger.info(f"   🚀 Enhanced: {results['enhanced_accuracy']:.3f} ({results['enhanced_accuracy']*100:.1f}%)")
        logger.info(f"   📊 Baseline: {results['baseline_accuracy']:.3f} ({results['baseline_accuracy']*100:.1f}%)")
        logger.info(f"   ⬆️ Improvement: +{results['accuracy_improvement']:.3f} ({results['accuracy_improvement']*100:.1f} pp)")
        logger.info(f"   📉 Error Reduction: {results['error_reduction_pct']:.1f}%")
        
        if results['statistical_significance']:
            logger.info(f"   ✅ Statistically significant (p={results['p_value']:.4f})")
        else:
            logger.info(f"   ⚠️ Not statistically significant (p={results['p_value']:.4f})")


def create_test_scenarios(trajectories: List) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Create test scenarios for real-world validation."""
    try:
        scenarios = {}
        
        if not trajectories:
            return scenarios
        
        # Group trajectories by characteristics
        short_trajectories = [t for t in trajectories if len(t.points) < 30]
        long_trajectories = [t for t in trajectories if len(t.points) >= 50]
        high_speed_trajectories = [t for t in trajectories if t.avg_speed_kmh > 25]
        
        # Note: This is a simplified version
        # In practice, you'd extract features and targets for each scenario
        logger.info(f"Created scenarios: {len(short_trajectories)} short, {len(long_trajectories)} long, {len(high_speed_trajectories)} high-speed trajectories")
        
        return scenarios
        
    except Exception as e:
        logger.error(f"Failed to create test scenarios: {e}")
        return {}


if __name__ == "__main__":
    # Example usage
    print("🧪 ZKPAS Accuracy Validation Framework")
    print("Target: 92%+ accuracy with <50m error")
    
    # This would be used with actual models and data
    validator = AccuracyValidator(target_accuracy=0.92)
    print("✅ Validator initialized")