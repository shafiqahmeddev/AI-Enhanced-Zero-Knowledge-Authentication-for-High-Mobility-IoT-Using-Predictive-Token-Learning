# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ZKPAS is a Zero-Knowledge Proof Authentication System for high-mobility IoT devices with AI-enhanced predictive token learning. The system includes cryptographic protocols, machine learning components, and Byzantine fault tolerance.

## Architecture

### Core Components

- **zkpas/app/components/**: Core system entities
  - `trusted_authority.py`: Central registration and certificate issuance
  - `gateway_node.py`: Edge authentication with sliding window tokens and degraded mode
  - `iot_device.py`: Mobility-aware device with ZKP capabilities
  - `interfaces.py`: ABC-based component interfaces
  
- **zkpas/shared/**: Shared utilities and configuration
  - `config.py`: Type-safe configuration management
  - `crypto_utils.py`: ECC, hashing, encryption, ZKP primitives
  
- **zkpas/app/**: Application-level services
  - `state_machine.py`: Event-driven protocol state management
  - `mobility_predictor.py`: AI-powered movement prediction
  - `events.py`: Async event system for component communication
  - `data_subsetting.py`: K-anonymity data privacy
  - `federated_learning.py`: Distributed ML training
  - `analytics_dashboard.py`: Real-time monitoring
  - `mlflow_tracking.py`: ML experiment tracking
  - `model_interpretability.py`: Explainable AI features

### Key Design Patterns

- **Event-Driven Architecture**: Components communicate via async events
- **Graceful Degradation**: Gateways operate with cached credentials when TA unavailable
- **Sliding Window Tokens**: Reduces authentication overhead for mobile devices
- **Byzantine Fault Tolerance**: Threshold cryptography for cross-domain auth

## Common Development Commands

### Environment Setup
```bash
cd zkpas
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests (preferred)
pytest

# Run with coverage
pytest --cov=app --cov=shared --cov-report=term-missing

# Run specific test files
pytest tests/test_trusted_authority.py -v
pytest tests/test_gateway_node.py -v
pytest tests/test_iot_device.py -v

# Run integration tests
pytest tests/test_phase4_integration.py -v

# Run standalone validation scripts
python validate_phase3.py
python test_all_modules.py
python test_phase3.py
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Run all pre-commit hooks
pre-commit run --all-files
```

### Development Tools
```bash
# Update dependencies
pip-compile requirements.in

# Security audit
pip-audit

# Run MLflow server for experiment tracking
mlflow server --host 0.0.0.0 --port 5000
```

## Configuration

Configuration is managed through `shared/config.py` with environment variable support:

- **Cryptographic settings**: ECC curve, key sizes, post-quantum readiness
- **Network settings**: Timeouts, sliding window sizes
- **ML settings**: Batch sizes, learning rates, model parameters
- **Privacy settings**: K-anonymity parameters, differential privacy

## Security Considerations

The system implements defense-in-depth with:

- **ECC secp256r1** cryptography with digital signatures
- **AES-GCM** encryption for data confidentiality
- **HKDF** key derivation for secure key management
- **Constant-time** operations to prevent timing attacks
- **Post-quantum** algorithm stubs for future migration

Refer to `zkpas/adr/002-threat-model.md` for complete STRIDE threat analysis.

## Data Privacy

The system implements privacy-preserving features:

- **K-anonymity**: Implemented in `data_subsetting.py` for location privacy
- **Differential privacy**: Noise injection for aggregate statistics
- **Federated learning**: Distributed training without centralized data
- **Minimal data collection**: Only essential mobility patterns stored

## Performance Optimization

- **Memory-conscious design**: <100MB per gateway node
- **Async processing**: Non-blocking operations with bounded queues
- **Batch processing**: Configurable batch sizes for ML operations
- **Connection pooling**: Efficient resource management
- **Authentication caching**: <50ms latency for cached tokens

## Current Status

- **Phase 3 Complete**: Event-driven architecture, ML integration, formal state machines
- **Phase 4 In Progress**: Privacy-preserving MLOps, federated learning, explainable AI
- **Test Coverage**: 63% overall with comprehensive integration tests

## Working with Modified Files

The following files have recent changes (check git status):
- `zkpas/app/data_subsetting.py`
- `zkpas/app/mobility_predictor.py`

Use `git diff` to see specific changes before making additional modifications.