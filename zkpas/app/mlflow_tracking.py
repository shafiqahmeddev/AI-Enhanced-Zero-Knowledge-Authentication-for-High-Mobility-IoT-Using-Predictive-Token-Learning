"""
Task 4.3: Experiment Tracking with MLflow
==========================================

This module integrates MLflow for comprehensive experiment tracking in the
ZKPAS MLOps pipeline. It provides reproducible experiment management,
model versioning, and performance monitoring for federated learning scenarios.

Key Features:
- Automated experiment tracking for federated learning rounds
- Model versioning and registry integration
- Hyperparameter optimization tracking
- Privacy-preserving metrics logging
- Integration with existing ZKPAS event system
- Experiment comparison and analysis tools
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import pandas as pd
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import hashlib
import uuid

# Import from our existing modules
from app.events import EventBus, Event, EventType
from app.federated_learning import FederatedRound, ModelUpdate
from app.model_interpretability import ExplanationResult, GlobalExplanation

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an ML experiment."""
    experiment_name: str
    experiment_id: Optional[str] = None
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    description: Optional[str] = None


@dataclass
class RunMetrics:
    """Metrics for a single experiment run."""
    run_id: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    tags: Dict[str, str]
    artifacts: List[str]
    start_time: float
    end_time: Optional[float] = None
    status: str = "RUNNING"


class ZKPASMLflowTracker:
    """
    MLflow integration for ZKPAS federated learning experiments.
    
    Provides comprehensive experiment tracking, model versioning,
    and performance monitoring for the ZKPAS MLOps pipeline.
    """
    
    def __init__(self,
                 tracking_uri: str = "./mlruns",
                 experiment_name: str = "zkpas_federated_learning",
                 auto_log: bool = True):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
            auto_log: Whether to automatically log metrics and parameters
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.auto_log = auto_log
        
        # Setup MLflow
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=f"{tracking_uri}/artifacts/{experiment_name}"
            )
        except Exception:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
        
        # Set active experiment
        mlflow.set_experiment(experiment_name)
        
        # State management
        self.active_runs = {}
        self.completed_runs = []
        self.current_federated_experiment = None
        
        # Privacy tracking
        self.privacy_budgets = {}
        self.privacy_spent = {}
        
        logger.info(f"MLflow tracker initialized: {experiment_name} (ID: {self.experiment_id})")
    
    def start_federated_experiment(self,
                                 experiment_config: Dict[str, Any],
                                 privacy_budget: float = 10.0) -> str:
        """
        Start a new federated learning experiment.
        
        Args:
            experiment_config: Configuration for the experiment
            privacy_budget: Total privacy budget for the experiment
            
        Returns:
            Run ID for the federated experiment
        """
        # Generate unique experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"federated_experiment_{timestamp}"
        
        # Start MLflow run
        run = mlflow.start_run(run_name=run_name)
        run_id = run.info.run_id
        
        # Log experiment configuration
        mlflow.log_params({
            "experiment_type": "federated_learning",
            "privacy_budget": privacy_budget,
            "max_rounds": experiment_config.get("max_rounds", 100),
            "min_clients": experiment_config.get("min_clients", 2),
            "client_fraction": experiment_config.get("client_fraction", 1.0),
            "aggregation_strategy": experiment_config.get("aggregation_strategy", "fedavg"),
            "differential_privacy": experiment_config.get("differential_privacy", True)
        })
        
        # Log additional parameters
        for key, value in experiment_config.items():
            if key not in ["max_rounds", "min_clients", "client_fraction", "aggregation_strategy"]:
                mlflow.log_param(key, value)
        
        # Set tags
        mlflow.set_tags({
            "experiment_type": "federated_learning",
            "privacy_preserving": "true",
            "zkpas_version": "1.0",
            "start_time": datetime.now().isoformat()
        })
        
        # Initialize privacy tracking
        self.privacy_budgets[run_id] = privacy_budget
        self.privacy_spent[run_id] = 0.0
        
        self.current_federated_experiment = run_id
        self.active_runs[run_id] = RunMetrics(
            run_id=run_id,
            metrics={},
            parameters=experiment_config,
            tags={"experiment_type": "federated_learning"},
            artifacts=[],
            start_time=datetime.now().timestamp()
        )
        
        logger.info(f"Started federated experiment: {run_name} ({run_id})")
        return run_id
    
    def log_federated_round(self,
                           round_info: FederatedRound,
                           global_metrics: Dict[str, float],
                           model_weights: Optional[Dict[str, Any]] = None):
        """
        Log metrics and artifacts for a federated learning round.
        
        Args:
            round_info: Information about the federated round
            global_metrics: Global model performance metrics
            model_weights: Model weights to save as artifacts
        """
        if not self.current_federated_experiment:
            logger.warning("No active federated experiment")
            return
        
        round_num = round_info.round_number
        
        # Log round-specific metrics
        round_metrics = {
            f"round_{round_num}_participants": len(round_info.participating_clients),
            f"round_{round_num}_privacy_spent": round_info.privacy_budget_used,
            f"round_{round_num}_duration": (round_info.end_time or datetime.now().timestamp()) - round_info.start_time
        }
        
        # Log global performance metrics
        for metric_name, value in global_metrics.items():
            round_metrics[f"round_{round_num}_{metric_name}"] = value
            # Also log as current metric for trend tracking
            round_metrics[f"global_{metric_name}"] = value
        
        # Log convergence metrics if available
        if round_info.convergence_metrics:
            for metric_name, value in round_info.convergence_metrics.items():
                round_metrics[f"round_{round_num}_convergence_{metric_name}"] = value
        
        # Log all metrics
        for metric_name, value in round_metrics.items():
            mlflow.log_metric(metric_name, value, step=round_num)
        
        # Update privacy tracking
        if self.current_federated_experiment in self.privacy_spent:
            self.privacy_spent[self.current_federated_experiment] += round_info.privacy_budget_used
            
            # Log cumulative privacy spent
            mlflow.log_metric(
                "cumulative_privacy_spent", 
                self.privacy_spent[self.current_federated_experiment], 
                step=round_num
            )
            
            # Log privacy budget remaining
            budget_remaining = (self.privacy_budgets[self.current_federated_experiment] - 
                              self.privacy_spent[self.current_federated_experiment])
            mlflow.log_metric("privacy_budget_remaining", budget_remaining, step=round_num)
        
        # Save model artifacts if provided
        if model_weights:
            self._save_model_artifacts(model_weights, round_num)
        
        # Update run metrics
        if self.current_federated_experiment in self.active_runs:
            self.active_runs[self.current_federated_experiment].metrics.update(round_metrics)
        
        logger.debug(f"Logged federated round {round_num} metrics")
    
    def log_client_update(self,
                         update: ModelUpdate,
                         privacy_metrics: Optional[Dict[str, float]] = None):
        """
        Log individual client update information.
        
        Args:
            update: Model update from a federated client
            privacy_metrics: Privacy-related metrics for the update
        """
        if not self.current_federated_experiment:
            return
        
        # Log client-specific metrics
        client_metrics = {
            f"client_{update.client_id}_round_{update.round_number}_samples": update.sample_count,
            f"client_{update.client_id}_round_{update.round_number}_gradient_norm": update.gradient_norm,
            f"client_{update.client_id}_round_{update.round_number}_privacy_spent": update.privacy_spent
        }
        
        # Log validation metrics if available
        if update.validation_metrics:
            for metric_name, value in update.validation_metrics.items():
                client_metrics[f"client_{update.client_id}_round_{update.round_number}_{metric_name}"] = value
        
        # Log privacy metrics if provided
        if privacy_metrics:
            for metric_name, value in privacy_metrics.items():
                client_metrics[f"client_{update.client_id}_round_{update.round_number}_privacy_{metric_name}"] = value
        
        # Log metrics with round as step
        for metric_name, value in client_metrics.items():
            mlflow.log_metric(metric_name, value, step=update.round_number)
        
        logger.debug(f"Logged client update from {update.client_id} for round {update.round_number}")
    
    def log_interpretability_results(self,
                                   explanations: List[ExplanationResult],
                                   global_explanation: Optional[GlobalExplanation] = None):
        """
        Log model interpretability results.
        
        Args:
            explanations: List of local explanations
            global_explanation: Global model explanation
        """
        if not self.current_federated_experiment:
            return
        
        # Aggregate explanation metrics
        interp_metrics = {
            "total_explanations": len(explanations),
            "lime_explanations": sum(1 for exp in explanations if exp.explanation_method.startswith('lime')),
            "shap_explanations": sum(1 for exp in explanations if exp.explanation_method.startswith('shap')),
            "privacy_preserving_explanations": sum(1 for exp in explanations if exp.privacy_preserving),
            "avg_explanation_confidence": np.mean([exp.confidence_score for exp in explanations]) if explanations else 0.0
        }
        
        # Log explanation metrics
        for metric_name, value in interp_metrics.items():
            mlflow.log_metric(f"interpretability_{metric_name}", value)
        
        # Save explanation artifacts
        if explanations:
            self._save_interpretability_artifacts(explanations, global_explanation)
        
        logger.info(f"Logged interpretability results: {len(explanations)} explanations")
    
    def log_hyperparameter_search(self,
                                hyperparams: Dict[str, Any],
                                performance_metrics: Dict[str, float],
                                search_iteration: int):
        """
        Log hyperparameter search results.
        
        Args:
            hyperparams: Hyperparameter configuration
            performance_metrics: Performance metrics for this configuration
            search_iteration: Iteration number in the search
        """
        if not self.current_federated_experiment:
            return
        
        # Log hyperparameters with iteration prefix
        for param_name, value in hyperparams.items():
            mlflow.log_param(f"search_iter_{search_iteration}_{param_name}", value)
        
        # Log performance metrics with iteration step
        for metric_name, value in performance_metrics.items():
            mlflow.log_metric(f"hp_search_{metric_name}", value, step=search_iteration)
        
        # Log hyperparameter combination hash for uniqueness
        hp_hash = hashlib.md5(json.dumps(hyperparams, sort_keys=True).encode()).hexdigest()[:8]
        mlflow.log_param(f"search_iter_{search_iteration}_config_hash", hp_hash)
        
        logger.debug(f"Logged hyperparameter search iteration {search_iteration}")
    
    def compare_experiments(self, 
                          run_ids: Optional[List[str]] = None,
                          metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple experiments or runs.
        
        Args:
            run_ids: List of run IDs to compare (if None, compares recent runs)
            metrics: List of metrics to include in comparison
            
        Returns:
            DataFrame with experiment comparison
        """
        if run_ids is None:
            # Get recent runs from current experiment
            experiment = mlflow.get_experiment(self.experiment_id)
            runs = mlflow.search_runs(experiment_ids=[self.experiment_id], max_results=10)
            run_ids = runs['run_id'].tolist() if not runs.empty else []
        
        if not run_ids:
            logger.warning("No runs found for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for run_id in run_ids:
            try:
                run = mlflow.get_run(run_id)
                
                run_data = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'status': run.info.status,
                    'duration_minutes': (run.info.end_time - run.info.start_time) / (1000 * 60) if run.info.end_time else None
                }
                
                # Add parameters
                for param_name, param_value in run.data.params.items():
                    run_data[f"param_{param_name}"] = param_value
                
                # Add metrics (latest values)
                if metrics:
                    for metric_name in metrics:
                        if metric_name in run.data.metrics:
                            run_data[f"metric_{metric_name}"] = run.data.metrics[metric_name]
                else:
                    # Add all metrics
                    for metric_name, metric_value in run.data.metrics.items():
                        run_data[f"metric_{metric_name}"] = metric_value
                
                comparison_data.append(run_data)
                
            except Exception as e:
                logger.warning(f"Failed to get data for run {run_id}: {e}")
        
        df = pd.DataFrame(comparison_data)
        logger.info(f"Generated comparison for {len(comparison_data)} experiments")
        return df
    
    def end_federated_experiment(self,
                                final_metrics: Dict[str, float],
                                model_artifact: Optional[Any] = None):
        """
        End the current federated learning experiment.
        
        Args:
            final_metrics: Final performance metrics
            model_artifact: Final model to save
        """
        if not self.current_federated_experiment:
            logger.warning("No active federated experiment to end")
            return
        
        # Log final metrics
        for metric_name, value in final_metrics.items():
            mlflow.log_metric(f"final_{metric_name}", value)
        
        # Log experiment summary
        run_metrics = self.active_runs.get(self.current_federated_experiment)
        if run_metrics:
            mlflow.log_metric("total_duration", 
                            datetime.now().timestamp() - run_metrics.start_time)
            
            # Count total rounds
            round_metrics = [k for k in run_metrics.metrics.keys() if k.startswith("round_")]
            unique_rounds = set()
            for metric in round_metrics:
                parts = metric.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    unique_rounds.add(int(parts[1]))
            
            if unique_rounds:
                mlflow.log_metric("total_rounds_completed", max(unique_rounds))
        
        # Log final privacy summary
        run_id = self.current_federated_experiment
        if run_id in self.privacy_spent and run_id in self.privacy_budgets:
            mlflow.log_metric("final_privacy_spent", self.privacy_spent[run_id])
            mlflow.log_metric("final_privacy_efficiency", 
                            self.privacy_spent[run_id] / self.privacy_budgets[run_id])
        
        # Save final model if provided
        if model_artifact is not None:
            try:
                mlflow.sklearn.log_model(model_artifact, "final_model")
                logger.info("Saved final model artifact")
            except Exception as e:
                logger.warning(f"Failed to save final model: {e}")
        
        # End MLflow run
        mlflow.end_run()
        
        # Move to completed runs
        if run_id in self.active_runs:
            run_metrics = self.active_runs.pop(run_id)
            run_metrics.end_time = datetime.now().timestamp()
            run_metrics.status = "COMPLETED"
            self.completed_runs.append(run_metrics)
        
        self.current_federated_experiment = None
        logger.info(f"Ended federated experiment: {run_id}")
    
    def _save_model_artifacts(self, model_weights: Dict[str, Any], round_num: int):
        """Save model weights as MLflow artifacts."""
        try:
            # Create temporary file for model weights
            temp_path = f"temp_model_round_{round_num}.pkl"
            
            with open(temp_path, 'wb') as f:
                pickle.dump(model_weights, f)
            
            # Log as artifact
            mlflow.log_artifact(temp_path, f"models/round_{round_num}")
            
            # Clean up temporary file
            os.remove(temp_path)
            
        except Exception as e:
            logger.warning(f"Failed to save model artifacts for round {round_num}: {e}")
    
    def _save_interpretability_artifacts(self,
                                       explanations: List[ExplanationResult],
                                       global_explanation: Optional[GlobalExplanation]):
        """Save interpretability results as artifacts."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save local explanations
            explanations_data = [asdict(exp) for exp in explanations]
            explanations_path = f"temp_explanations_{timestamp}.json"
            
            with open(explanations_path, 'w') as f:
                json.dump(explanations_data, f, indent=2)
            
            mlflow.log_artifact(explanations_path, "interpretability")
            os.remove(explanations_path)
            
            # Save global explanation if available
            if global_explanation:
                global_exp_path = f"temp_global_explanation_{timestamp}.json"
                
                with open(global_exp_path, 'w') as f:
                    json.dump(asdict(global_explanation), f, indent=2)
                
                mlflow.log_artifact(global_exp_path, "interpretability")
                os.remove(global_exp_path)
                
                # Save visualizations if available
                for viz_path in global_explanation.visualization_paths:
                    if os.path.exists(viz_path):
                        mlflow.log_artifact(viz_path, "interpretability/visualizations")
            
        except Exception as e:
            logger.warning(f"Failed to save interpretability artifacts: {e}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        # Get experiment info
        experiment = mlflow.get_experiment(self.experiment_id)
        
        # Get all runs
        all_runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        
        summary = {
            'experiment_name': self.experiment_name,
            'experiment_id': self.experiment_id,
            'tracking_uri': self.tracking_uri,
            'total_runs': len(all_runs) if not all_runs.empty else 0,
            'active_runs': len(self.active_runs),
            'completed_runs': len(self.completed_runs),
            'current_federated_experiment': self.current_federated_experiment,
            'experiment_created': experiment.creation_time if experiment else None
        }
        
        if not all_runs.empty:
            summary.update({
                'avg_run_duration': all_runs['end_time'].subtract(all_runs['start_time']).mean() / (1000 * 60),  # minutes
                'successful_runs': len(all_runs[all_runs['status'] == 'FINISHED']),
                'failed_runs': len(all_runs[all_runs['status'] == 'FAILED'])
            })
        
        return summary


# Event handlers for MLflow integration
class MLflowEventHandler:
    """Event handler for integrating MLflow with ZKPAS events."""
    
    def __init__(self, mlflow_tracker: ZKPASMLflowTracker, event_bus: EventBus):
        self.mlflow_tracker = mlflow_tracker
        self.event_bus = event_bus
        
        # Register event handlers
        self.event_bus.subscribe_sync(EventType.FL_ROUND_COMPLETE, self._handle_round_complete)
        self.event_bus.subscribe_sync(EventType.FL_TRAINING_COMPLETE, self._handle_training_complete)
        
        logger.info("MLflow event handlers registered")
    
    def _handle_round_complete(self, event: Event):
        """Handle federated learning round completion."""
        data = event.data
        
        # Create mock FederatedRound object
        round_info = type('FederatedRound', (), {
            'round_number': data.get('round_number', 0),
            'participating_clients': data.get('participating_clients', []),
            'privacy_budget_used': data.get('privacy_spent', 0.0),
            'start_time': datetime.now().timestamp() - 60,  # Mock start time
            'end_time': datetime.now().timestamp()
        })()
        
        global_metrics = data.get('global_metrics', {})
        
        self.mlflow_tracker.log_federated_round(round_info, global_metrics)
    
    def _handle_training_complete(self, event: Event):
        """Handle federated learning training completion."""
        data = event.data
        final_metrics = data.get('final_metrics', {})
        
        self.mlflow_tracker.end_federated_experiment(final_metrics)


def main():
    """Example usage of MLflow experiment tracking."""
    print("📊 Testing MLflow Experiment Tracking")
    
    # Initialize MLflow tracker
    tracker = ZKPASMLflowTracker(
        tracking_uri="./mlruns",
        experiment_name="zkpas_demo_experiment"
    )
    
    # Start a federated experiment
    config = {
        "max_rounds": 10,
        "min_clients": 3,
        "client_fraction": 0.8,
        "learning_rate": 0.01,
        "privacy_budget": 5.0
    }
    
    run_id = tracker.start_federated_experiment(config, privacy_budget=5.0)
    print(f"Started experiment: {run_id}")
    
    # Simulate logging some rounds
    for round_num in range(1, 4):
        # Mock round info
        round_info = type('FederatedRound', (), {
            'round_number': round_num,
            'participating_clients': [f"client_{i}" for i in range(3)],
            'privacy_budget_used': 0.5,
            'start_time': datetime.now().timestamp() - 30,
            'end_time': datetime.now().timestamp()
        })()
        
        # Mock global metrics
        global_metrics = {
            'accuracy': 0.8 + round_num * 0.02 + np.random.normal(0, 0.01),
            'loss': 0.5 - round_num * 0.05 + np.random.normal(0, 0.02),
            'f1_score': 0.75 + round_num * 0.03 + np.random.normal(0, 0.01)
        }
        
        tracker.log_federated_round(round_info, global_metrics)
        print(f"Logged round {round_num}")
    
    # End experiment
    final_metrics = {
        'final_accuracy': 0.87,
        'final_loss': 0.35,
        'final_f1_score': 0.84
    }
    
    tracker.end_federated_experiment(final_metrics)
    print("Experiment completed")
    
    # Get experiment summary
    summary = tracker.get_experiment_summary()
    print(f"Experiment summary: {summary}")
    
    print("✅ MLflow Experiment Tracking test completed!")


if __name__ == "__main__":
    main()
