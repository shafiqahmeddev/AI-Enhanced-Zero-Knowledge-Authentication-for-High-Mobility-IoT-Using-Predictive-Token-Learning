"""
Task 4.2: Model Interpretability with LIME/SHAP
===============================================

This module provides comprehensive model interpretability for the ZKPAS MLOps
pipeline using LIME (Local Interpretable Model-agnostic Explanations) and 
SHAP (SHapley Additive exPlanations) frameworks.

Key Features:
- Local explanations for individual predictions (LIME)
- Global model explanations and feature importance (SHAP)
- Privacy-preserving explanation generation
- Interactive visualization dashboards
- Integration with mobility prediction models
- Explanation caching and aggregation for federated settings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import warnings

# Interpretability libraries
import lime
import lime.lime_tabular
import shap
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set up logging
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class ExplanationResult:
    """Represents an explanation result for a model prediction."""
    prediction_id: str
    model_type: str
    prediction_value: Union[float, int, List[float]]
    explanation_method: str  # 'lime' or 'shap'
    feature_importances: Dict[str, float]
    local_explanation: Optional[Dict[str, Any]]
    confidence_score: float
    timestamp: float
    privacy_preserving: bool = False


@dataclass
class GlobalExplanation:
    """Represents global model explanation across all features."""
    model_id: str
    explanation_method: str
    feature_importance_ranking: List[Tuple[str, float]]
    interaction_effects: Optional[Dict[str, Dict[str, float]]]
    summary_statistics: Dict[str, Any]
    visualization_paths: List[str]
    generated_at: float


class LIMEExplainer:
    """
    Local Interpretable Model-agnostic Explanations (LIME) wrapper.
    
    Provides local explanations for individual predictions while maintaining
    privacy when needed for federated learning scenarios.
    """
    
    def __init__(self, 
                 feature_names: List[str],
                 categorical_features: Optional[List[int]] = None,
                 mode: str = 'classification',
                 privacy_budget: float = 1.0):
        """
        Initialize LIME explainer.
        
        Args:
            feature_names: Names of input features
            categorical_features: Indices of categorical features
            mode: 'classification' or 'regression'
            privacy_budget: Privacy budget for differential privacy
        """
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.mode = mode
        self.privacy_budget = privacy_budget
        self.explainer = None
        self.is_fitted = False
        
        logger.info(f"LIME explainer initialized for {mode} with {len(feature_names)} features")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the LIME explainer on training data."""
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            categorical_features=self.categorical_features,
            mode=self.mode,
            discretize_continuous=True
        )
        self.is_fitted = True
        logger.info("LIME explainer fitted on training data")
    
    def explain_instance(self, 
                        instance: np.ndarray,
                        predict_fn: Callable,
                        num_features: int = 10,
                        num_samples: int = 5000,
                        add_privacy_noise: bool = False) -> ExplanationResult:
        """
        Explain a single prediction instance.
        
        Args:
            instance: Input instance to explain
            predict_fn: Model prediction function
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME
            add_privacy_noise: Whether to add differential privacy noise
            
        Returns:
            ExplanationResult with local explanation
        """
        if not self.is_fitted:
            raise ValueError("LIME explainer must be fitted before explaining instances")
        
        prediction_id = f"lime_{datetime.now().timestamp()}"
        
        # Get prediction
        if self.mode == 'classification':
            prediction = predict_fn(instance.reshape(1, -1))[0]
            if hasattr(prediction, '__len__') and len(prediction) > 1:
                prediction_value = prediction.tolist()
            else:
                prediction_value = float(prediction)
        else:
            prediction_value = float(predict_fn(instance.reshape(1, -1))[0])
        
        # Generate LIME explanation
        explanation = self.explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Extract feature importances
        feature_importances = {}
        local_exp = explanation.as_list()
        
        for feature_desc, importance in local_exp:
            # Parse feature name from description
            feature_name = feature_desc.split('=')[0].strip() if '=' in feature_desc else feature_desc
            
            # Add privacy noise if requested
            if add_privacy_noise:
                noise_scale = self.privacy_budget / 10.0
                noise = np.random.laplace(0, 1/noise_scale)
                importance += noise
            
            feature_importances[feature_name] = float(importance)
        
        # Calculate confidence score (simplified)
        confidence = explanation.score if hasattr(explanation, 'score') else 0.8
        
        result = ExplanationResult(
            prediction_id=prediction_id,
            model_type="lime_explained",
            prediction_value=prediction_value,
            explanation_method="lime",
            feature_importances=feature_importances,
            local_explanation={
                "lime_explanation": local_exp,
                "intercept": getattr(explanation, 'intercept', 0.0),
                "prediction_local": getattr(explanation, 'local_pred', prediction_value)
            },
            confidence_score=float(confidence),
            timestamp=datetime.now().timestamp(),
            privacy_preserving=add_privacy_noise
        )
        
        logger.debug(f"Generated LIME explanation for instance {prediction_id}")
        return result
    
    def explain_batch(self,
                     instances: np.ndarray,
                     predict_fn: Callable,
                     num_features: int = 10,
                     add_privacy_noise: bool = False) -> List[ExplanationResult]:
        """Explain multiple instances in batch."""
        explanations = []
        
        for i, instance in enumerate(instances):
            try:
                explanation = self.explain_instance(
                    instance, predict_fn, num_features, add_privacy_noise=add_privacy_noise
                )
                explanations.append(explanation)
            except Exception as e:
                logger.warning(f"Failed to explain instance {i}: {e}")
        
        logger.info(f"Generated LIME explanations for {len(explanations)}/{len(instances)} instances")
        return explanations


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) wrapper for global model interpretation.
    
    Provides both local and global explanations with theoretical guarantees
    and support for various model types.
    """
    
    def __init__(self, 
                 model: BaseEstimator,
                 feature_names: List[str],
                 explainer_type: str = 'auto'):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model to explain
            feature_names: Names of input features  
            explainer_type: Type of SHAP explainer ('tree', 'linear', 'kernel', 'auto')
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer_type = explainer_type
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
        logger.info(f"SHAP explainer initialized with {explainer_type} explainer")
    
    def fit(self, X_background: np.ndarray):
        """Initialize SHAP explainer with background data."""
        # Choose explainer type
        if self.explainer_type == 'auto':
            if hasattr(self.model, 'tree_'):
                self.explainer_type = 'tree'
            elif hasattr(self.model, 'coef_'):
                self.explainer_type = 'linear'
            else:
                self.explainer_type = 'kernel'
        
        # Create appropriate explainer
        if self.explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.explainer_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X_background)
        elif self.explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict, X_background[:100])  # Sample for efficiency
        else:
            raise ValueError(f"Unknown explainer type: {self.explainer_type}")
        
        logger.info(f"SHAP {self.explainer_type} explainer fitted")
    
    def explain_global(self, X_test: np.ndarray) -> GlobalExplanation:
        """Generate global explanation for the model."""
        if self.explainer is None:
            raise ValueError("SHAP explainer must be fitted before generating explanations")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_test)
        self.shap_values = shap_values
        
        if isinstance(shap_values, list):  # Multi-class classification
            shap_values = shap_values[0]  # Use first class for ranking
        
        # Calculate feature importance ranking
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feature_importance = list(zip(self.feature_names, mean_abs_shap))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate summary statistics
        summary_stats = {
            'mean_shap_values': np.mean(shap_values, axis=0).tolist(),
            'std_shap_values': np.std(shap_values, axis=0).tolist(),
            'max_shap_values': np.max(shap_values, axis=0).tolist(),
            'min_shap_values': np.min(shap_values, axis=0).tolist()
        }
        
        # Generate visualizations
        viz_paths = self._create_visualizations(X_test, shap_values)
        
        explanation = GlobalExplanation(
            model_id=f"shap_model_{datetime.now().timestamp()}",
            explanation_method=f"shap_{self.explainer_type}",
            feature_importance_ranking=feature_importance,
            interaction_effects=None,  # Could be extended
            summary_statistics=summary_stats,
            visualization_paths=viz_paths,
            generated_at=datetime.now().timestamp()
        )
        
        logger.info(f"Generated global SHAP explanation with {len(feature_importance)} features")
        return explanation
    
    def explain_instance(self, 
                        instance: np.ndarray,
                        add_privacy_noise: bool = False) -> ExplanationResult:
        """Generate SHAP explanation for a single instance."""
        if self.explainer is None:
            raise ValueError("SHAP explainer must be fitted before explaining instances")
        
        prediction_id = f"shap_{datetime.now().timestamp()}"
        
        # Get SHAP values for instance
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))
        
        if isinstance(shap_values, list):  # Multi-class
            shap_values = shap_values[0]  # Use first class
        
        shap_values = shap_values[0]  # Single instance
        
        # Get prediction
        prediction = self.model.predict(instance.reshape(1, -1))[0]
        
        # Create feature importance dictionary
        feature_importances = {}
        for i, (feature_name, shap_val) in enumerate(zip(self.feature_names, shap_values)):
            if add_privacy_noise:
                noise = np.random.laplace(0, 0.01)  # Small noise
                shap_val += noise
            feature_importances[feature_name] = float(shap_val)
        
        result = ExplanationResult(
            prediction_id=prediction_id,
            model_type="shap_explained",
            prediction_value=float(prediction),
            explanation_method=f"shap_{self.explainer_type}",
            feature_importances=feature_importances,
            local_explanation={
                "shap_values": shap_values.tolist(),
                "expected_value": getattr(self.explainer, 'expected_value', 0.0),
                "instance_values": instance.tolist()
            },
            confidence_score=0.95,  # SHAP provides theoretical guarantees
            timestamp=datetime.now().timestamp(),
            privacy_preserving=add_privacy_noise
        )
        
        logger.debug(f"Generated SHAP explanation for instance {prediction_id}")
        return result
    
    def _create_visualizations(self, X_test: np.ndarray, shap_values: np.ndarray) -> List[str]:
        """Create and save SHAP visualizations."""
        viz_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, show=False)
            summary_path = f"visualizations/shap_summary_{timestamp}.png"
            Path("visualizations").mkdir(exist_ok=True)
            plt.savefig(summary_path, bbox_inches='tight', dpi=150)
            plt.close()
            viz_paths.append(summary_path)
            
            # Feature importance bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, 
                             plot_type="bar", show=False)
            bar_path = f"visualizations/shap_importance_{timestamp}.png"
            plt.savefig(bar_path, bbox_inches='tight', dpi=150)
            plt.close()
            viz_paths.append(bar_path)
            
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")
        
        return viz_paths


class ModelInterpretabilityManager:
    """
    Manages model interpretability for the ZKPAS MLOps pipeline.
    
    Coordinates LIME and SHAP explanations, handles privacy-preserving
    explanation generation, and provides unified API for interpretability.
    """
    
    def __init__(self,
                 model: BaseEstimator,
                 feature_names: List[str],
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 categorical_features: Optional[List[int]] = None,
                 privacy_budget: float = 2.0):
        """
        Initialize interpretability manager.
        
        Args:
            model: Trained model to explain
            feature_names: Names of input features
            X_train: Training data for background/reference
            y_train: Training labels
            categorical_features: Indices of categorical features
            privacy_budget: Privacy budget for explanations
        """
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        self.y_train = y_train
        self.categorical_features = categorical_features or []
        self.privacy_budget = privacy_budget
        
        # Determine task type
        if hasattr(model, 'predict_proba'):
            self.task_type = 'classification'
        else:
            self.task_type = 'regression'
        
        # Initialize explainers
        self.lime_explainer = LIMEExplainer(
            feature_names=feature_names,
            categorical_features=categorical_features,
            mode=self.task_type,
            privacy_budget=privacy_budget / 2
        )
        
        self.shap_explainer = SHAPExplainer(
            model=model,
            feature_names=feature_names
        )
        
        # Fit explainers
        self.lime_explainer.fit(X_train, y_train)
        self.shap_explainer.fit(X_train)
        
        # Storage for explanations
        self.explanation_history = []
        self.global_explanations = {}
        
        logger.info(f"Model interpretability manager initialized for {self.task_type}")
    
    def explain_prediction(self,
                          instance: np.ndarray,
                          methods: List[str] = ['lime', 'shap'],
                          privacy_preserving: bool = False) -> Dict[str, ExplanationResult]:
        """
        Generate comprehensive explanations for a single prediction.
        
        Args:
            instance: Input instance to explain
            methods: Explanation methods to use
            privacy_preserving: Whether to add privacy noise
            
        Returns:
            Dictionary mapping method names to explanation results
        """
        explanations = {}
        
        # Get model prediction function
        if self.task_type == 'classification':
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict
        
        # Generate LIME explanation
        if 'lime' in methods:
            try:
                lime_explanation = self.lime_explainer.explain_instance(
                    instance=instance,
                    predict_fn=predict_fn,
                    add_privacy_noise=privacy_preserving
                )
                explanations['lime'] = lime_explanation
                self.explanation_history.append(lime_explanation)
            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")
        
        # Generate SHAP explanation
        if 'shap' in methods:
            try:
                shap_explanation = self.shap_explainer.explain_instance(
                    instance=instance,
                    add_privacy_noise=privacy_preserving
                )
                explanations['shap'] = shap_explanation
                self.explanation_history.append(shap_explanation)
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
        
        logger.info(f"Generated {len(explanations)} explanations for instance")
        return explanations
    
    def generate_global_explanation(self, 
                                  X_test: np.ndarray,
                                  explanation_id: Optional[str] = None) -> GlobalExplanation:
        """Generate global model explanation using SHAP."""
        if explanation_id is None:
            explanation_id = f"global_{datetime.now().timestamp()}"
        
        global_explanation = self.shap_explainer.explain_global(X_test)
        self.global_explanations[explanation_id] = global_explanation
        
        logger.info(f"Generated global explanation: {explanation_id}")
        return global_explanation
    
    def compare_explanations(self, 
                           explanations: Dict[str, ExplanationResult]) -> Dict[str, Any]:
        """Compare explanations from different methods."""
        if len(explanations) < 2:
            return {}
        
        comparison = {
            'methods_compared': list(explanations.keys()),
            'feature_agreement': {},
            'ranking_correlation': {},
            'prediction_consistency': {}
        }
        
        # Get feature importance rankings
        rankings = {}
        for method, explanation in explanations.items():
            # Sort features by absolute importance
            sorted_features = sorted(
                explanation.feature_importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            rankings[method] = [f[0] for f in sorted_features]
        
        # Calculate rank correlation between methods
        if len(rankings) >= 2:
            methods = list(rankings.keys())
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    # Calculate Spearman rank correlation
                    correlation = self._calculate_rank_correlation(
                        rankings[method1], rankings[method2]
                    )
                    comparison['ranking_correlation'][f"{method1}_vs_{method2}"] = correlation
        
        # Check prediction consistency
        predictions = [exp.prediction_value for exp in explanations.values()]
        if len(set(predictions)) == 1:
            comparison['prediction_consistency']['all_methods_agree'] = True
        else:
            comparison['prediction_consistency']['prediction_variance'] = np.var(predictions)
        
        return comparison
    
    def aggregate_explanations(self, 
                             explanations: List[ExplanationResult],
                             aggregation_method: str = 'mean') -> Dict[str, float]:
        """Aggregate multiple explanations for federated interpretability."""
        if not explanations:
            return {}
        
        # Group by explanation method
        method_groups = {}
        for exp in explanations:
            method = exp.explanation_method
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(exp)
        
        aggregated = {}
        
        for method, method_explanations in method_groups.items():
            # Get all feature names
            all_features = set()
            for exp in method_explanations:
                all_features.update(exp.feature_importances.keys())
            
            # Aggregate importances
            feature_aggregates = {}
            for feature in all_features:
                values = [exp.feature_importances.get(feature, 0.0) for exp in method_explanations]
                
                if aggregation_method == 'mean':
                    aggregated_value = np.mean(values)
                elif aggregation_method == 'median':
                    aggregated_value = np.median(values)
                elif aggregation_method == 'max':
                    aggregated_value = np.max(np.abs(values))
                else:
                    aggregated_value = np.mean(values)  # Default to mean
                
                feature_aggregates[feature] = float(aggregated_value)
            
            aggregated[method] = feature_aggregates
        
        logger.info(f"Aggregated {len(explanations)} explanations using {aggregation_method}")
        return aggregated
    
    def save_explanations(self, file_path: str):
        """Save explanation history to file."""
        data = {
            'explanation_history': [asdict(exp) for exp in self.explanation_history],
            'global_explanations': {k: asdict(v) for k, v in self.global_explanations.items()},
            'metadata': {
                'feature_names': self.feature_names,
                'task_type': self.task_type,
                'total_explanations': len(self.explanation_history),
                'saved_at': datetime.now().timestamp()
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved explanations to {file_path}")
    
    def load_explanations(self, file_path: str):
        """Load explanation history from file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct explanation objects
        self.explanation_history = [
            ExplanationResult(**exp_data) for exp_data in data['explanation_history']
        ]
        
        self.global_explanations = {
            k: GlobalExplanation(**v) for k, v in data['global_explanations'].items()
        }
        
        logger.info(f"Loaded {len(self.explanation_history)} explanations from {file_path}")
    
    def _calculate_rank_correlation(self, ranking1: List[str], ranking2: List[str]) -> float:
        """Calculate Spearman rank correlation between two feature rankings."""
        # Find common features
        common_features = set(ranking1) & set(ranking2)
        if len(common_features) < 2:
            return 0.0
        
        # Get ranks for common features
        ranks1 = {feature: i for i, feature in enumerate(ranking1) if feature in common_features}
        ranks2 = {feature: i for i, feature in enumerate(ranking2) if feature in common_features}
        
        # Calculate Spearman correlation
        n = len(common_features)
        sum_d_squared = sum((ranks1[f] - ranks2[f])**2 for f in common_features)
        correlation = 1 - (6 * sum_d_squared) / (n * (n**2 - 1))
        
        return correlation
    
    def get_interpretability_summary(self) -> Dict[str, Any]:
        """Get summary of interpretability analysis."""
        return {
            'total_explanations': len(self.explanation_history),
            'methods_used': list(set(exp.explanation_method for exp in self.explanation_history)),
            'global_explanations': len(self.global_explanations),
            'privacy_preserving_ratio': sum(1 for exp in self.explanation_history if exp.privacy_preserving) / max(len(self.explanation_history), 1),
            'average_confidence': np.mean([exp.confidence_score for exp in self.explanation_history]) if self.explanation_history else 0.0,
            'feature_names': self.feature_names,
            'task_type': self.task_type
        }


def create_sample_model_and_data():
    """Create sample model and data for testing."""
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=7, 
                              n_redundant=2, n_clusters_per_class=1, random_state=42)
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test, feature_names


def main():
    """Example usage and testing of model interpretability."""
    print("🔍 Testing Model Interpretability with LIME/SHAP")
    
    # Create sample model and data
    model, X_train, X_test, y_train, y_test, feature_names = create_sample_model_and_data()
    
    print(f"Model accuracy: {model.score(X_test, y_test):.3f}")
    
    # Initialize interpretability manager
    interp_manager = ModelInterpretabilityManager(
        model=model,
        feature_names=feature_names,
        X_train=X_train,
        y_train=y_train,
        privacy_budget=2.0
    )
    
    # Explain a single prediction
    test_instance = X_test[0]
    explanations = interp_manager.explain_prediction(
        instance=test_instance,
        methods=['lime', 'shap'],
        privacy_preserving=False
    )
    
    print(f"Generated {len(explanations)} explanations for test instance")
    
    for method, explanation in explanations.items():
        print(f"\n{method.upper()} Explanation:")
        print(f"  Prediction: {explanation.prediction_value}")
        print(f"  Confidence: {explanation.confidence_score:.3f}")
        print(f"  Top 3 features:")
        top_features = sorted(explanation.feature_importances.items(), 
                            key=lambda x: abs(x[1]), reverse=True)[:3]
        for feature, importance in top_features:
            print(f"    {feature}: {importance:.4f}")
    
    # Compare explanations
    comparison = interp_manager.compare_explanations(explanations)
    print(f"\nExplanation comparison:")
    for key, value in comparison.items():
        print(f"  {key}: {value}")
    
    # Generate global explanation
    global_exp = interp_manager.generate_global_explanation(X_test[:100])
    print(f"\nGlobal explanation generated with {len(global_exp.feature_importance_ranking)} features")
    print("Top 5 most important features globally:")
    for i, (feature, importance) in enumerate(global_exp.feature_importance_ranking[:5]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # Get summary
    summary = interp_manager.get_interpretability_summary()
    print(f"\nInterpretability Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("✅ Model Interpretability module test completed!")


if __name__ == "__main__":
    main()
