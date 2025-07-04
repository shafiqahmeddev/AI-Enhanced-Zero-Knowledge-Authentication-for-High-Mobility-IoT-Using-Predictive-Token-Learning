# ZKPAS - Zero Knowledge Proof Authentication System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Status: Active](https://img.shields.io/badge/Status-Active-green.svg)](https://github.com/shafiqahmed/zkpas)

> **A cutting-edge Zero-Knowledge Proof Authentication System for high-mobility IoT devices with AI-enhanced predictive token learning and Byzantine fault tolerance.**

## 🚀 Overview

ZKPAS is a revolutionary authentication system specifically designed for high-mobility IoT environments. It combines zero-knowledge proofs with machine learning to provide secure, efficient, and scalable authentication for resource-constrained devices in dynamic network conditions.

### Key Innovation
- **Zero-Knowledge Authentication**: Devices authenticate without revealing sensitive information
- **AI-Powered Mobility Prediction**: LSTM neural networks predict device movement patterns
- **Byzantine Fault Tolerance**: Resilient operation in untrusted environments
- **Lightweight Design**: Optimized for resource-constrained IoT devices (<100MB RAM)

## ✨ Features

### 🔐 Security
- **ECC secp256r1** cryptography with digital signatures
- **Zero-knowledge proof** protocols for privacy-preserving authentication
- **Constant-time operations** to prevent timing attacks
- **Post-quantum algorithm** stubs for future-proofing

### 🧠 AI-Enhanced Mobility Prediction
- **LSTM neural networks** for movement pattern analysis
- **Lightweight predictor** fallback for resource-constrained scenarios
- **Real-time adaptation** to changing mobility patterns
- **Accuracy**: 40-45% prediction accuracy (realistic for GPS-based systems)

### 🛡️ Byzantine Fault Tolerance
- **Threshold cryptography** for cross-domain authentication
- **Graceful degradation** when trusted authorities are unavailable
- **Sliding window tokens** to reduce authentication overhead
- **Distributed consensus** mechanisms

### ⚡ Performance
- **Authentication latency**: <200ms per device
- **Memory footprint**: <100MB total system usage
- **LSTM training**: ~1.3s on synthetic data
- **Scalability**: Supports 100+ concurrent IoT devices

## 🎯 Quick Start

### Prerequisites
- Python 3.7 or higher
- 4GB+ RAM (recommended)
- Network connectivity for distributed scenarios

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shafiqahmed/zkpas.git
   cd zkpas
   ```

2. **Setup the system** (one-time setup)
   ```bash
   cd zkpas
   python setup_zkpas.py
   ```

3. **Run the system** (every time)
   ```bash
   python run_zkpas.py
   ```

That's it! The system provides an interactive menu to explore all features.

## 🎮 Usage Examples

### Interactive Mode
```bash
python run_zkpas.py
# Interactive menu with options:
# 1. Basic Authentication Demo
# 2. LSTM Mobility Prediction
# 3. Security Stress Testing
# 4. Complete Integration Test
```

### Command Line Interface
```bash
# Run specific demonstrations
python run_zkpas.py --demo basic      # Basic ZKP authentication
python run_zkpas.py --demo lstm       # LSTM prediction system
python run_zkpas.py --demo security   # Security testing
python run_zkpas.py --demo all        # Complete system test

# System utilities
python run_zkpas.py --health          # Health check
python run_zkpas.py --test            # Run unit tests
```

### Advanced Usage
```bash
# Direct demo access
python demos/demo_zkpas_basic.py      # Basic authentication
python demos/demo_lstm_system.py      # LSTM prediction
python demos/demo_security_stress_testing.py  # Security testing

# Standalone components
python app/lightweight_predictor.py   # Lightweight ML predictor
python app/mobility_predictor.py      # Full LSTM predictor
```

## 🏗️ Architecture

### Core Components

```
zkpas/
├── app/                    # Core application modules
│   ├── mobility_predictor.py    # LSTM-based mobility prediction
│   ├── lightweight_predictor.py # Fallback predictor
│   ├── state_machine.py         # Protocol state management
│   └── events.py               # Event-driven architecture
├── demos/                  # Example demonstrations
│   ├── demo_zkpas_basic.py     # Basic authentication
│   ├── demo_lstm_system.py     # LSTM prediction
│   └── demo_security_stress_testing.py  # Security testing
├── shared/                 # Shared utilities
│   ├── crypto_utils.py         # Cryptographic primitives
│   └── config.py              # Configuration management
└── tests/                  # Comprehensive test suite
```

### System Architecture
- **Event-Driven Design**: Asynchronous event processing for scalability
- **Microservices Pattern**: Modular components for flexibility
- **Graceful Degradation**: Fallback mechanisms for reliability
- **Plugin Architecture**: Extensible design for new features

## 📊 Performance Metrics

### Authentication Performance
- **Latency**: <200ms per device authentication
- **Throughput**: 100+ concurrent authentications
- **Memory Usage**: <100MB total system footprint
- **CPU Usage**: <10% on modern hardware

### ML Prediction Accuracy
- **LSTM Accuracy**: 40-45% within 100m threshold
- **Prediction Error**: ~127m average error (realistic for GPS)
- **Training Time**: 1.3s on synthetic trajectory data
- **Inference Time**: <10ms per prediction

### Network Efficiency
- **Bandwidth**: <1KB per authentication request
- **Packet Overhead**: Minimal protocol overhead
- **Network Resilience**: Operates in high-latency environments

## 🔧 Configuration

The system automatically detects available dependencies and gracefully falls back to lightweight implementations when needed. No manual configuration is required for basic usage.

### Advanced Configuration
```python
# Environment variables for customization
export ZKPAS_MAX_DEVICES=100
export ZKPAS_MEMORY_LIMIT=6GB
export ZKPAS_BATCH_SIZE=32
export ZKPAS_LOG_LEVEL=INFO
```

## 🧪 Testing

### Run All Tests
```bash
python run_zkpas.py --test
```

### Specific Test Suites
```bash
# Unit tests
python -m pytest tests/test_trusted_authority.py -v
python -m pytest tests/test_gateway_node.py -v
python -m pytest tests/test_iot_device.py -v

# Integration tests
python -m pytest tests/test_phase4_integration.py -v
python -m pytest tests/test_security_stress.py -v
```

## 🛠️ Development

### Dependencies
- **Required**: Python 3.7+, hashlib, secrets, hmac
- **Recommended**: numpy, scikit-learn, cryptography
- **Optional**: TensorFlow, PyTorch (for advanced ML features)

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run code quality checks
python -m flake8 .
python -m black .
python -m mypy .

# Run comprehensive tests
python -m pytest --cov=app --cov=shared
```

## 🌟 Key Innovations

### 1. **Zero-Knowledge Proof Integration**
- Custom ZKP protocols optimized for IoT constraints
- Privacy-preserving authentication without credential exposure
- Efficient proof generation and verification

### 2. **AI-Enhanced Mobility Prediction**
- LSTM neural networks for trajectory prediction
- Adaptive learning from device movement patterns
- Lightweight fallback for resource-constrained devices

### 3. **Byzantine Fault Tolerance**
- Threshold cryptography for distributed trust
- Graceful degradation in untrusted environments
- Cross-domain authentication protocols

### 4. **Resource Optimization**
- Memory-conscious design for IoT devices
- Efficient cryptographic operations
- Scalable architecture for large deployments

## 📚 Documentation

- **Technical Documentation**: See `zkpas/docs/` directory
- **API Reference**: Comprehensive inline documentation
- **Architecture Decisions**: See `zkpas/adr/` directory
- **Implementation Guide**: Detailed setup and usage instructions

## 🤝 Contributing

This project is actively maintained and welcomes contributions from the research community. Areas of interest include:

- **Performance optimization** for resource-constrained devices
- **Advanced ML models** for mobility prediction
- **Post-quantum cryptography** integration
- **Real-world deployment** case studies

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Shafiq Ahmed**
- Email: [s.ahmed@essex.ac.uk](mailto:s.ahmed@essex.ac.uk)
- Affiliation: University of Essex
- Research Focus: IoT Security, Zero-Knowledge Proofs, Machine Learning

## 🎓 Academic Context

This work represents cutting-edge research in:
- **IoT Security**: Novel approaches to device authentication
- **Zero-Knowledge Proofs**: Practical applications in mobile environments
- **Machine Learning**: AI-enhanced security systems
- **Distributed Systems**: Byzantine fault tolerance in IoT networks

## 📈 Future Roadmap

- **Real-world deployment** testing with industrial IoT systems
- **Post-quantum cryptography** integration for future security
- **Edge computing** optimization for 5G/6G networks
- **Federated learning** for privacy-preserving ML training

---

**Built with ❤️ for the future of IoT security**

*ZKPAS - Where Zero-Knowledge meets Artificial Intelligence*