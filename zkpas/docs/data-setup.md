# Dataset Setup Guide

This guide explains how to set up the required datasets for the ZKPAS research project.

## Excluded Datasets

The following datasets are excluded from the git repository due to their large size (2.4GB total):

### 1. Geolife Trajectories Dataset (Microsoft Research)

- **Size**: ~1.8GB
- **Location**: `Datasets/Geolife Trajectories 1.3/Data/`
- **Content**: GPS trajectories from 182 users over 5+ years
- **Paper**: "GeoLife: A Collaborative Social Networking Service among User, Location and Trajectory"

**Download Instructions:**

```bash
# Download from Microsoft Research
wget https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip

# Extract to correct location
unzip "Geolife Trajectories 1.3.zip" -d Datasets/
```

### 2. Beijing Taxi GPS Logs (T-Drive)

- **Size**: ~600MB
- **Location**: `Datasets/release/taxi_log_2008_by_id/`
- **Content**: GPS logs from ~10,000 Beijing taxis
- **Paper**: "T-Drive: Driving Directions Based on Taxi Trajectories"

**Download Instructions:**

```bash
# Download from Microsoft Research T-Drive project
# URL: https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/
# Manual download required due to license agreement
```

## Dataset Structure

After downloading, your directory structure should look like:

```
Datasets/
├── Geolife Trajectories 1.3/
│   ├── User Guide-1.3.pdf          # ✅ Included in repo
│   └── Data/                        # ❌ Excluded (too large)
│       ├── 000/
│       ├── 001/
│       └── ... (182 user directories)
├── release/
│   ├── user_guide.pdf               # ✅ Included in repo
│   └── taxi_log_2008_by_id/         # ❌ Excluded (too large)
│       ├── 1.txt
│       ├── 10.txt
│       └── ... (~10,000 files)
```

## Data Processing Scripts

Use these scripts to process the raw datasets:

### Geolife Processing

```bash
cd zkpas
python scripts/data/process_geolife.py --input ../Datasets/Geolife\ Trajectories\ 1.3/Data/ --output data/processed/geolife/
```

### T-Drive Processing

```bash
cd zkpas
python scripts/data/process_tdrive.py --input ../Datasets/release/taxi_log_2008_by_id/ --output data/processed/tdrive/
```

## Alternative: Simulated Data

For development and testing without downloading large datasets:

```bash
cd zkpas
python scripts/data/generate_synthetic.py --users 50 --days 30 --output data/synthetic/
```

This generates realistic but synthetic mobility data for testing.

## Data Usage in ZKPAS

The datasets are used for:

1. **Mobility Pattern Analysis**: Understanding real-world movement patterns
2. **Prediction Model Training**: Training ML models for mobility prediction
3. **Authentication Simulation**: Realistic mobility scenarios for testing
4. **Performance Benchmarking**: Real-scale data for performance evaluation

## Data Privacy and Ethics

- All datasets are publicly available research datasets
- Personal identifiers have been anonymized by original researchers
- Use only for academic and research purposes
- Follow original dataset license terms
- Do not attempt to re-identify individuals

## Storage Recommendations

### For Researchers

- Store datasets on external drives or cloud storage
- Use symbolic links to avoid duplication:
  ```bash
  ln -s /external/drive/Datasets ./Datasets
  ```

### For CI/CD

- Use synthetic data for automated testing
- Store small sample datasets in test fixtures
- Download datasets only for performance benchmarks

## Troubleshooting

### Common Issues

**"Dataset not found" errors:**

- Ensure datasets are downloaded and extracted correctly
- Check file permissions (should be readable)
- Verify directory structure matches expected layout

**Memory issues during processing:**

- Process datasets in chunks using `--batch-size` parameter
- Increase available memory or use smaller samples
- Consider using streaming processing for large files

**Slow processing:**

- Use SSD storage for better I/O performance
- Enable parallel processing with `--workers` parameter
- Pre-filter datasets to reduce size

### Getting Help

1. Check the user guides included in `Datasets/*/user_guide.pdf`
2. Review dataset papers for format specifications
3. Use `--help` flag with processing scripts for options
4. Check issues in the repository for common problems
