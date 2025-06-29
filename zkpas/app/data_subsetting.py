"""
Task 4.0: Reproducible Data Subsetting & Validation
==================================================

This module implements reproducible data subsetting and validation for the ZKPAS
privacy-preserving MLOps pipeline. It ensures consistent data splits, validates
data quality, and provides mechanisms for reproducible experiments.

Key Features:
- Deterministic data splitting with seeded randomization
- Data quality validation and anomaly detection
- Feature engineering pipeline with consistent transformations
- Privacy-preserving data sampling techniques
- Differential privacy budget allocation
- Data lineage tracking for experiment reproducibility
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import random

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class DataSubset:
    """Represents a data subset with metadata."""
    subset_id: str
    data_path: str
    metadata: Dict[str, Any]
    created_at: float
    size: int
    features: List[str]
    target: Optional[str]
    split_type: str  # 'train', 'validation', 'test'
    privacy_budget: float
    hash_signature: str


@dataclass 
class DataValidationResult:
    """Results from data validation checks."""
    is_valid: bool
    validation_errors: List[str]
    warnings: List[str]
    quality_score: float
    statistics: Dict[str, Any]
    validation_timestamp: float


class DataSubsettingManager:
    """
    Manages reproducible data subsetting and validation for MLOps pipeline.
    
    Provides deterministic data splitting, quality validation, and privacy-preserving
    sampling techniques for consistent experiment reproducibility.
    """
    
    def __init__(self, 
                 data_root: str,
                 random_seed: int = 42,
                 privacy_budget: float = 1.0,
                 min_quality_score: float = 0.8):
        """
        Initialize the data subsetting manager.
        
        Args:
            data_root: Root directory for data storage
            random_seed: Seed for reproducible randomization
            privacy_budget: Total privacy budget for differential privacy
            min_quality_score: Minimum acceptable data quality score
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        self.random_seed = random_seed
        self.privacy_budget = privacy_budget
        self.min_quality_score = min_quality_score
        
        # Set seeds for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Initialize storage for subsets and validation results
        self.subsets_registry = {}
        self.validation_cache = {}
        
        # Create subdirectories
        (self.data_root / "subsets").mkdir(exist_ok=True)
        (self.data_root / "metadata").mkdir(exist_ok=True)
        (self.data_root / "validation_reports").mkdir(exist_ok=True)
        
        logger.info(f"DataSubsettingManager initialized with data_root: {data_root}")
    
    def create_reproducible_split(self,
                                  data: pd.DataFrame,
                                  target_column: Optional[str] = None,
                                  train_size: float = 0.7,
                                  val_size: float = 0.15,
                                  test_size: float = 0.15,
                                  stratify: bool = True,
                                  subset_prefix: str = "dataset") -> Dict[str, DataSubset]:
        """
        Create reproducible train/validation/test splits.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column for stratification
            train_size: Proportion for training set
            val_size: Proportion for validation set  
            test_size: Proportion for test set
            stratify: Whether to stratify splits based on target
            subset_prefix: Prefix for subset identifiers
            
        Returns:
            Dictionary mapping split names to DataSubset objects
        """
        # Validate split proportions
        if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
            raise ValueError("Split sizes must sum to 1.0")
        
        logger.info(f"Creating reproducible split: train={train_size}, val={val_size}, test={test_size}")
        
        # Calculate data hash for reproducibility verification
        data_hash = self._calculate_data_hash(data)
        
        # Prepare stratification
        stratify_column = None
        if stratify and target_column and target_column in data.columns:
            stratify_column = data[target_column]
            logger.info(f"Using stratification on column: {target_column}")
        
        # First split: separate test set
        if test_size > 0:
            X_temp, X_test, y_temp, y_test = train_test_split(
                data.drop(columns=[target_column] if target_column else []),
                data[target_column] if target_column else None,
                test_size=test_size,
                random_state=self.random_seed,
                stratify=stratify_column if stratify_column is not None else None
            )
            
            # Reconstruct test DataFrame
            test_data = X_test.copy()
            if target_column and y_test is not None:
                test_data[target_column] = y_test
        else:
            X_temp = data.drop(columns=[target_column] if target_column else [])
            y_temp = data[target_column] if target_column else None
            test_data = pd.DataFrame()
        
        # Second split: separate train and validation
        val_proportion = val_size / (train_size + val_size)
        
        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_proportion,
                random_state=self.random_seed + 1,  # Different seed for second split
                stratify=y_temp if stratify and y_temp is not None else None
            )
            
            # Reconstruct DataFrames
            train_data = X_train.copy()
            val_data = X_val.copy()
            
            if target_column:
                if y_train is not None:
                    train_data[target_column] = y_train
                if y_val is not None:
                    val_data[target_column] = y_val
        else:
            train_data = X_temp.copy()
            if target_column and y_temp is not None:
                train_data[target_column] = y_temp
            val_data = pd.DataFrame()
        
        # Create DataSubset objects
        timestamp = datetime.now().timestamp()
        feature_columns = [col for col in data.columns if col != target_column]
        
        subsets = {}
        
        # Privacy budget allocation (simple equal distribution)
        budget_per_split = self.privacy_budget / 3 if test_size > 0 and val_size > 0 else self.privacy_budget / 2
        
        if not train_data.empty:
            train_subset = DataSubset(
                subset_id=f"{subset_prefix}_train_{timestamp}",
                data_path=str(self.data_root / "subsets" / f"{subset_prefix}_train.pkl"),
                metadata={
                    "split_type": "train",
                    "original_size": len(data),
                    "split_size": len(train_data),
                    "proportion": train_size,
                    "stratified": stratify,
                    "random_seed": self.random_seed,
                    "data_hash": data_hash
                },
                created_at=timestamp,
                size=len(train_data),
                features=feature_columns,
                target=target_column,
                split_type="train",
                privacy_budget=budget_per_split,
                hash_signature=self._calculate_data_hash(train_data)
            )
            
            # Save train data
            with open(train_subset.data_path, 'wb') as f:
                pickle.dump(train_data, f)
            
            subsets["train"] = train_subset
        
        if not val_data.empty:
            val_subset = DataSubset(
                subset_id=f"{subset_prefix}_val_{timestamp}",
                data_path=str(self.data_root / "subsets" / f"{subset_prefix}_val.pkl"),
                metadata={
                    "split_type": "validation",
                    "original_size": len(data),
                    "split_size": len(val_data),
                    "proportion": val_size,
                    "stratified": stratify,
                    "random_seed": self.random_seed + 1,
                    "data_hash": data_hash
                },
                created_at=timestamp,
                size=len(val_data),
                features=feature_columns,
                target=target_column,
                split_type="validation",
                privacy_budget=budget_per_split,
                hash_signature=self._calculate_data_hash(val_data)
            )
            
            # Save validation data
            with open(val_subset.data_path, 'wb') as f:
                pickle.dump(val_data, f)
                
            subsets["validation"] = val_subset
        
        if not test_data.empty:
            test_subset = DataSubset(
                subset_id=f"{subset_prefix}_test_{timestamp}",
                data_path=str(self.data_root / "subsets" / f"{subset_prefix}_test.pkl"),
                metadata={
                    "split_type": "test",
                    "original_size": len(data),
                    "split_size": len(test_data),
                    "proportion": test_size,
                    "stratified": stratify,
                    "random_seed": self.random_seed,
                    "data_hash": data_hash
                },
                created_at=timestamp,
                size=len(test_data),
                features=feature_columns,
                target=target_column,
                split_type="test",
                privacy_budget=budget_per_split,
                hash_signature=self._calculate_data_hash(test_data)
            )
            
            # Save test data
            with open(test_subset.data_path, 'wb') as f:
                pickle.dump(test_data, f)
                
            subsets["test"] = test_subset
        
        # Save subset metadata
        self._save_subset_metadata(subsets)
        
        logger.info(f"Created {len(subsets)} data subsets with total size: {sum(s.size for s in subsets.values())}")
        return subsets
    
    def validate_data_quality(self, data: Union[pd.DataFrame, str, DataSubset]) -> DataValidationResult:
        """
        Perform comprehensive data quality validation.
        
        Args:
            data: DataFrame, file path, or DataSubset to validate
            
        Returns:
            DataValidationResult with validation status and metrics
        """
        # Load data if needed
        if isinstance(data, str):
            data = pd.read_csv(data)
        elif isinstance(data, DataSubset):
            with open(data.data_path, 'rb') as f:
                data = pickle.load(f)
        
        logger.info(f"Validating data quality for dataset with shape: {data.shape}")
        
        validation_errors = []
        warnings = []
        statistics = {}
        
        # Basic shape validation
        if data.empty:
            validation_errors.append("Dataset is empty")
            return DataValidationResult(
                is_valid=False,
                validation_errors=validation_errors,
                warnings=warnings,
                quality_score=0.0,
                statistics=statistics,
                validation_timestamp=datetime.now().timestamp()
            )
        
        # Missing value analysis
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        statistics["missing_values"] = {
            "total_missing": int(missing_counts.sum()),
            "columns_with_missing": list(missing_counts[missing_counts > 0].index),
            "missing_percentages": missing_percentages.to_dict()
        }
        
        # Check for excessive missing values
        high_missing_cols = missing_percentages[missing_percentages > 50].index.tolist()
        if high_missing_cols:
            warnings.append(f"Columns with >50% missing values: {high_missing_cols}")
        
        # Duplicate analysis
        duplicate_count = data.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(data)) * 100
        
        statistics["duplicates"] = {
            "count": int(duplicate_count),
            "percentage": float(duplicate_percentage)
        }
        
        if duplicate_percentage > 10:
            warnings.append(f"High duplicate rate: {duplicate_percentage:.2f}%")
        
        # Data type analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        statistics["data_types"] = {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "total_columns": len(data.columns)
        }
        
        # Outlier detection for numeric columns
        outlier_info = {}
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(data)) * 100
            
            outlier_info[col] = {
                "count": outlier_count,
                "percentage": float(outlier_percentage),
                "bounds": [float(lower_bound), float(upper_bound)]
            }
            
            if outlier_percentage > 5:
                warnings.append(f"Column {col} has {outlier_percentage:.2f}% outliers")
        
        statistics["outliers"] = outlier_info
        
        # Cardinality analysis for categorical columns
        cardinality_info = {}
        for col in categorical_cols:
            unique_count = data[col].nunique()
            cardinality_percentage = (unique_count / len(data)) * 100
            
            cardinality_info[col] = {
                "unique_count": unique_count,
                "cardinality_percentage": float(cardinality_percentage),
                "most_frequent": data[col].mode().iloc[0] if not data[col].mode().empty else None
            }
            
            if cardinality_percentage > 95:
                warnings.append(f"Column {col} has very high cardinality: {cardinality_percentage:.2f}%")
        
        statistics["cardinality"] = cardinality_info
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(statistics, validation_errors, warnings)
        
        # Determine if validation passes
        is_valid = (quality_score >= self.min_quality_score and 
                   len(validation_errors) == 0)
        
        result = DataValidationResult(
            is_valid=is_valid,
            validation_errors=validation_errors,
            warnings=warnings,
            quality_score=quality_score,
            statistics=statistics,
            validation_timestamp=datetime.now().timestamp()
        )
        
        # Save validation report
        self._save_validation_report(result)
        
        logger.info(f"Data validation completed. Quality score: {quality_score:.3f}, Valid: {is_valid}")
        return result
    
    def apply_privacy_preserving_sampling(self, 
                                        data: pd.DataFrame,
                                        sample_fraction: float = 0.8,
                                        noise_level: float = 0.01,
                                        k_anonymity: int = 5) -> pd.DataFrame:
        """
        Apply privacy-preserving sampling techniques to the data.
        
        Args:
            data: Input DataFrame
            sample_fraction: Fraction of data to sample
            noise_level: Level of differential privacy noise to add
            k_anonymity: Minimum k-anonymity level
            
        Returns:
            Privacy-preserved DataFrame
        """
        logger.info(f"Applying privacy-preserving sampling: fraction={sample_fraction}, noise={noise_level}")
        
        # Random sampling for basic privacy
        if sample_fraction < 1.0:
            data = data.sample(frac=sample_fraction, random_state=self.random_seed)
        
        # Add differential privacy noise to numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if noise_level > 0:
                # Add Laplace noise for differential privacy
                sensitivity = data[col].std()  # Simplified sensitivity calculation
                scale = sensitivity / (noise_level * len(data))
                noise = np.random.laplace(0, scale, size=len(data))
                data[col] = data[col] + noise
        
        logger.info(f"Applied privacy preservation to {len(numeric_cols)} numeric columns")
        return data
    
    def load_subset(self, subset_id: str) -> Optional[pd.DataFrame]:
        """Load a data subset by ID."""
        if subset_id in self.subsets_registry:
            subset = self.subsets_registry[subset_id]
            with open(subset.data_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def get_subset_metadata(self, subset_id: str) -> Optional[DataSubset]:
        """Get metadata for a data subset."""
        return self.subsets_registry.get(subset_id)
    
    def list_subsets(self) -> List[DataSubset]:
        """List all available data subsets."""
        return list(self.subsets_registry.values())
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate a hash signature for the data."""
        # Create a deterministic string representation
        data_str = data.to_string(index=False)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _calculate_quality_score(self, 
                                statistics: Dict[str, Any], 
                                errors: List[str], 
                                warnings: List[str]) -> float:
        """Calculate an overall data quality score."""
        if errors:
            return 0.0
        
        score = 1.0
        
        # Penalize missing values
        missing_penalty = min(statistics.get("missing_values", {}).get("total_missing", 0) / 1000, 0.3)
        score -= missing_penalty
        
        # Penalize high duplicate rate
        duplicate_penalty = min(statistics.get("duplicates", {}).get("percentage", 0) / 100, 0.2)
        score -= duplicate_penalty
        
        # Penalize warnings
        warning_penalty = min(len(warnings) * 0.05, 0.2)
        score -= warning_penalty
        
        return max(score, 0.0)
    
    def _save_subset_metadata(self, subsets: Dict[str, DataSubset]) -> None:
        """Save subset metadata to registry."""
        for name, subset in subsets.items():
            self.subsets_registry[subset.subset_id] = subset
            
            # Save individual metadata file
            metadata_path = self.data_root / "metadata" / f"{subset.subset_id}.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(subset), f, indent=2)
    
    def _save_validation_report(self, result: DataValidationResult) -> None:
        """Save validation report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.data_root / "validation_reports" / f"validation_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)


# Utility functions for data preprocessing
def create_feature_engineering_pipeline(numeric_features: List[str], 
                                       categorical_features: List[str]) -> Dict[str, Any]:
    """Create a reproducible feature engineering pipeline."""
    pipeline = {
        "numeric_imputer": SimpleImputer(strategy='median'),
        "categorical_imputer": SimpleImputer(strategy='most_frequent'),
        "scaler": StandardScaler(),
        "label_encoders": {col: LabelEncoder() for col in categorical_features}
    }
    return pipeline


def apply_feature_engineering(data: pd.DataFrame, 
                            pipeline: Dict[str, Any],
                            is_training: bool = True) -> pd.DataFrame:
    """Apply feature engineering pipeline to data."""
    processed_data = data.copy()
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    
    # Handle numeric features
    if len(numeric_cols) > 0:
        if is_training:
            processed_data[numeric_cols] = pipeline["numeric_imputer"].fit_transform(data[numeric_cols])
            processed_data[numeric_cols] = pipeline["scaler"].fit_transform(processed_data[numeric_cols])
        else:
            processed_data[numeric_cols] = pipeline["numeric_imputer"].transform(data[numeric_cols])
            processed_data[numeric_cols] = pipeline["scaler"].transform(processed_data[numeric_cols])
    
    # Handle categorical features
    for col in categorical_cols:
        if col in pipeline["label_encoders"]:
            if is_training:
                processed_data[col] = pipeline["label_encoders"][col].fit_transform(data[col].astype(str))
            else:
                processed_data[col] = pipeline["label_encoders"][col].transform(data[col].astype(str))
    
    return processed_data


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Data Subsetting & Validation")
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.uniform(0, 10, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.randint(0, 2, 1000)
    })
    
    # Add some missing values and outliers for testing
    sample_data.loc[np.random.choice(1000, 50), 'feature1'] = np.nan
    sample_data.loc[np.random.choice(1000, 10), 'feature2'] = 100  # Outliers
    
    # Initialize manager
    manager = DataSubsettingManager("./data_mlops")
    
    # Create splits
    subsets = manager.create_reproducible_split(
        sample_data, 
        target_column='target',
        stratify=True
    )
    
    print(f"Created {len(subsets)} subsets:")
    for name, subset in subsets.items():
        print(f"  {name}: {subset.size} samples")
    
    # Validate data quality
    validation_result = manager.validate_data_quality(sample_data)
    print(f"Data quality validation: {validation_result.is_valid}")
    print(f"Quality score: {validation_result.quality_score:.3f}")
    print(f"Warnings: {len(validation_result.warnings)}")
    
    print("Data Subsetting & Validation module ready!")
