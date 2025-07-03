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
- Secure data persistence using Parquet format (no pickle security risks)
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
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
                data_path=str(self.data_root / "subsets" / f"{subset_prefix}_train.parquet"),
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
            train_data.to_parquet(train_subset.data_path)
            
            subsets["train"] = train_subset
        
        if not val_data.empty:
            val_subset = DataSubset(
                subset_id=f"{subset_prefix}_val_{timestamp}",
                data_path=str(self.data_root / "subsets" / f"{subset_prefix}_val.parquet"),
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
            val_data.to_parquet(val_subset.data_path)
                
            subsets["validation"] = val_subset
        
        if not test_data.empty:
            test_subset = DataSubset(
                subset_id=f"{subset_prefix}_test_{timestamp}",
                data_path=str(self.data_root / "subsets" / f"{subset_prefix}_test.parquet"),
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
            test_data.to_parquet(test_subset.data_path)
                
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
            data = pd.read_parquet(data.data_path)
        
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
            noise_level: Privacy budget parameter (epsilon) for differential privacy
            k_anonymity: Minimum k-anonymity level (each group must have at least k records)
            
        Returns:
            Privacy-preserved DataFrame
        """
        logger.info(f"Applying privacy-preserving sampling: fraction={sample_fraction}, noise={noise_level}, k-anonymity={k_anonymity}")
        
        # Random sampling for basic privacy
        if sample_fraction < 1.0:
            data = data.sample(frac=sample_fraction, random_state=self.random_seed)
        
        # Apply k-anonymity by identifying quasi-identifiers and ensuring group sizes
        if k_anonymity > 1:
            data = self._apply_k_anonymity(data, k_anonymity)
            logger.info(f"Applied k-anonymity with k={k_anonymity}")
        
        # Add differential privacy noise to numeric columns
        # Implementation follows the Laplace mechanism for differential privacy:
        # - Sensitivity (Δf): maximum change in output when one record is modified
        # - Noise scale: sensitivity / epsilon, where epsilon is the privacy parameter
        # - Smaller epsilon = stronger privacy but more noise
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if noise_level > 0:
                # Calculate sensitivity as the data range (difference between max and min values)
                # This represents the maximum possible change when one record is added/removed
                col_min = data[col].min()
                col_max = data[col].max()
                col_range = col_max - col_min
                
                # Sensitivity equals the range of possible values for bounded data
                sensitivity = col_range if col_range > 0 else 1.0
                
                # Standard Laplace mechanism: scale = sensitivity / epsilon
                # This ensures (epsilon, 0)-differential privacy
                epsilon = noise_level
                scale = sensitivity / epsilon
                
                # Generate Laplace noise and add to data
                noise = np.random.laplace(0, scale, size=len(data))
                data[col] = data[col] + noise
        
        logger.info(f"Applied privacy preservation to {len(numeric_cols)} numeric columns")
        return data
    
    def _apply_k_anonymity(self, data: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Apply k-anonymity by ensuring each group has at least k records.
        
        Args:
            data: Input DataFrame
            k: Minimum group size for k-anonymity
            
        Returns:
            DataFrame with k-anonymity applied
        """
        if k <= 1:
            return data
        
        # Identify potential quasi-identifiers (categorical and discretized numeric columns)
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # For numeric columns, create bins to treat as quasi-identifiers
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        binned_data = data.copy()
        
        # Create bins for numeric columns (treating them as quasi-identifiers)
        for col in numeric_cols:
            if col in data.columns:
                # Create 10 bins for each numeric column
                try:
                    binned_data[f"{col}_bin"] = pd.cut(data[col], bins=10, labels=False, duplicates='drop')
                    categorical_cols.append(f"{col}_bin")
                except Exception:
                    # If binning fails (e.g., all values are the same), skip this column
                    continue
        
        # Remove target columns and other non-quasi-identifier columns
        # Assume columns with high cardinality are not quasi-identifiers
        quasi_identifiers = []
        for col in categorical_cols:
            if col in binned_data.columns:
                unique_ratio = binned_data[col].nunique() / len(binned_data)
                # Only use columns with reasonable cardinality as quasi-identifiers
                if unique_ratio < 0.5:  # Less than 50% unique values
                    quasi_identifiers.append(col)
        
        if not quasi_identifiers:
            # If no suitable quasi-identifiers found, return original data
            logger.warning("No suitable quasi-identifiers found for k-anonymity")
            return data
        
        logger.info(f"Applying k-anonymity with quasi-identifiers: {quasi_identifiers}")
        
        # Group by quasi-identifiers and check group sizes
        grouped = binned_data.groupby(quasi_identifiers)
        group_sizes = grouped.size()
        
        # Find groups that violate k-anonymity
        small_groups = group_sizes[group_sizes < k]
        
        if len(small_groups) == 0:
            # All groups already satisfy k-anonymity
            return data
        
        logger.info(f"Found {len(small_groups)} groups with size < {k}")
        
        # Strategy 1: Remove records from small groups (suppression)
        valid_groups = group_sizes[group_sizes >= k].index
        
        # Filter data to keep only records from valid groups
        mask = pd.Series(False, index=binned_data.index)
        for group_key in valid_groups:
            if isinstance(group_key, tuple):
                # Multiple quasi-identifiers
                group_mask = pd.Series(True, index=binned_data.index)
                for i, qi in enumerate(quasi_identifiers):
                    group_mask &= (binned_data[qi] == group_key[i])
            else:
                # Single quasi-identifier
                group_mask = (binned_data[quasi_identifiers[0]] == group_key)
            
            mask |= group_mask
        
        result_data = data[mask].copy()
        
        # Strategy 2: If too much data is lost, try generalization
        data_loss_ratio = 1 - (len(result_data) / len(data))
        
        if data_loss_ratio > 0.3:  # If more than 30% data loss
            logger.warning(f"High data loss ({data_loss_ratio:.2%}) due to k-anonymity. Applying generalization...")
            result_data = self._apply_generalization_for_k_anonymity(data, quasi_identifiers, k)
        
        records_removed = len(data) - len(result_data)
        logger.info(f"K-anonymity applied: removed {records_removed} records ({records_removed/len(data):.2%})")
        
        return result_data
    
    def _apply_generalization_for_k_anonymity(self, data: pd.DataFrame, 
                                            quasi_identifiers: List[str], 
                                            k: int) -> pd.DataFrame:
        """
        Apply generalization strategy for k-anonymity when suppression causes too much data loss.
        
        Args:
            data: Input DataFrame
            quasi_identifiers: List of quasi-identifier columns
            k: Minimum group size
            
        Returns:
            DataFrame with generalized quasi-identifiers
        """
        generalized_data = data.copy()
        
        # Simple generalization: combine small groups by relaxing constraints
        for qi in quasi_identifiers:
            if qi.endswith('_bin') and qi in generalized_data.columns:
                # For binned numeric columns, use fewer bins
                original_col = qi.replace('_bin', '')
                if original_col in data.columns:
                    try:
                        # Use fewer bins (5 instead of 10) for more generalization
                        generalized_data[qi] = pd.cut(data[original_col], bins=5, labels=False, duplicates='drop')
                    except Exception:
                        continue
            elif qi in generalized_data.columns:
                # For categorical columns, group rare categories as "Other"
                value_counts = generalized_data[qi].value_counts()
                rare_values = value_counts[value_counts < k].index
                generalized_data.loc[generalized_data[qi].isin(rare_values), qi] = 'Other'
        
        # Check if generalization helped
        grouped = generalized_data.groupby(quasi_identifiers)
        group_sizes = grouped.size()
        small_groups = group_sizes[group_sizes < k]
        
        if len(small_groups) > 0:
            # If generalization didn't help enough, fall back to suppression
            valid_groups = group_sizes[group_sizes >= k].index
            mask = pd.Series(False, index=generalized_data.index)
            
            for group_key in valid_groups:
                if isinstance(group_key, tuple):
                    group_mask = pd.Series(True, index=generalized_data.index)
                    for i, qi in enumerate(quasi_identifiers):
                        group_mask &= (generalized_data[qi] == group_key[i])
                else:
                    group_mask = (generalized_data[quasi_identifiers[0]] == group_key)
                
                mask |= group_mask
            
            generalized_data = generalized_data[mask]
        
        return generalized_data
    
    def load_subset(self, subset_id: str) -> Optional[pd.DataFrame]:
        """Load a data subset by ID."""
        if subset_id in self.subsets_registry:
            subset = self.subsets_registry[subset_id]
            return pd.read_parquet(subset.data_path)
        return None
    
    def get_subset_metadata(self, subset_id: str) -> Optional[DataSubset]:
        """Get metadata for a data subset."""
        return self.subsets_registry.get(subset_id)
    
    def list_subsets(self) -> List[DataSubset]:
        """List all available data subsets."""
        return list(self.subsets_registry.values())
    
    def create_stratified_sample(self,
                               data: pd.DataFrame,
                               target_column: str,
                               sample_size: int,
                               random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Create a stratified sample maintaining target distribution.
        
        Args:
            data: Input DataFrame
            target_column: Column to stratify on
            sample_size: Number of samples to create
            random_state: Random state for reproducibility
            
        Returns:
            Stratified sample DataFrame
        """
        if random_state is None:
            random_state = self.random_seed
            
        # Calculate sample sizes for each stratum
        target_dist = data[target_column].value_counts(normalize=True)
        stratified_samples = []
        
        for target_value, proportion in target_dist.items():
            stratum_data = data[data[target_column] == target_value]
            stratum_sample_size = max(1, int(sample_size * proportion))
            
            if len(stratum_data) >= stratum_sample_size:
                stratum_sample = stratum_data.sample(
                    n=stratum_sample_size, 
                    random_state=random_state
                )
            else:
                stratum_sample = stratum_data
            
            stratified_samples.append(stratum_sample)
        
        result = pd.concat(stratified_samples, ignore_index=True)
        logger.info(f"Created stratified sample of {len(result)} records from {len(data)} original records")
        return result
    
    def create_privacy_preserving_sample(self,
                                       data: pd.DataFrame,
                                       sample_size: int,
                                       privacy_budget: float = 1.0,
                                       epsilon: float = 0.1,
                                       k_anonymity: int = 5) -> pd.DataFrame:
        """
        Create a privacy-preserving sample with differential privacy and k-anonymity.
        
        Args:
            data: Input DataFrame
            sample_size: Desired sample size
            privacy_budget: Total privacy budget (not currently used)
            epsilon: Privacy parameter (epsilon) for differential privacy
            k_anonymity: Minimum k-anonymity level
            
        Returns:
            Privacy-preserving sample DataFrame
        """
        # Calculate sample fraction
        sample_fraction = min(1.0, sample_size / len(data))
        
        # Apply privacy-preserving sampling
        private_sample = self.apply_privacy_preserving_sampling(
            data=data,
            sample_fraction=sample_fraction,
            noise_level=epsilon,
            k_anonymity=k_anonymity
        )
        
        # If sample is still larger than desired, randomly sample
        if len(private_sample) > sample_size:
            private_sample = private_sample.sample(
                n=sample_size, 
                random_state=self.random_seed
            )
        
        logger.info(f"Created privacy-preserving sample of {len(private_sample)} records")
        return private_sample
    
    def create_train_validation_split(self,
                                    data: pd.DataFrame,
                                    validation_split: float = 0.2,
                                    random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation split.
        
        Args:
            data: Input DataFrame
            validation_split: Proportion for validation set
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_data, validation_data)
        """
        if random_state is None:
            random_state = self.random_seed
            
        train_data, val_data = train_test_split(
            data,
            test_size=validation_split,
            random_state=random_state
        )
        
        logger.info(f"Created train/validation split: {len(train_data)}/{len(val_data)} samples")
        return train_data, val_data
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """
        Calculate a memory-efficient hash signature for the data.
        
        Uses pandas' built-in hashing for better performance with large datasets.
        """
        # Calculate hash for each row using pandas built-in hashing
        # This is memory-efficient as it processes data row by row
        row_hashes = hash_pandas_object(data, index=False, encoding='utf8')
        
        # Combine all row hashes into a single deterministic hash
        # Use the hash of the array of hashes for final result
        combined_hash = hashlib.sha256(row_hashes.values.tobytes()).hexdigest()
        
        return combined_hash[:16]
    
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
