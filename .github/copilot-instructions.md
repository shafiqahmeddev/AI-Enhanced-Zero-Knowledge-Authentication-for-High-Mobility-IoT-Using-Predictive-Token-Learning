# GitHub Copilot Instructions for the ZKPAS Project

Welcome to the ZKPAS (Zero Knowledge Proof Authentication System) project! This document provides guidelines for using GitHub Copilot effectively within this repository. Following these instructions will help ensure that Copilot's suggestions are consistent with our project's architecture, coding style, and best practices.

## 1. Project Overview

ZKPAS is a Zero-Knowledge Proof Authentication System for high-mobility IoT devices. It features AI-enhanced predictive token learning and Byzantine fault tolerance. The system is designed to be secure, efficient, and scalable for resource-constrained devices.

**Key Goals:**

- Provide secure authentication without revealing sensitive information.
- Predict device movement patterns using AI.
- Ensure resilient operation in untrusted environments.
- Maintain a lightweight footprint for IoT devices.

## 2. Tech Stack

- **Primary Language:** Python 3.10+
- **Core Libraries:**
  - **Cryptography:** `cryptography` for ECC, digital signatures, and other cryptographic primitives.
  - **Machine Learning:** `tensorflow` and `scikit-learn` for mobility prediction. `numpy` and `pandas` for data manipulation.
  - **Asynchronous Programming:** `asyncio` for the event-driven architecture.
- **Testing:** `pytest`, `pytest-asyncio`, `pytest-cov`.
- **Code Quality:** `black` for formatting, `flake8` for linting, `mypy` for type checking.

When adding new dependencies, please update `requirements.in` and then run `pip-compile` to regenerate `requirements.txt`.

## 3. Coding Style & Conventions

- **Formatting:** All Python code is formatted with `black` using the default settings. Please run `black .` before committing.
- **Linting:** We use `flake8` for linting. Please ensure there are no linting errors before committing.
- **Type Hinting:** All new code **must** include type hints. We use `mypy` for static type checking in strict mode.
- **Docstrings:** Use Google-style docstrings for all modules, classes, and functions.
- **Logging:** We use the `loguru` library for logging. Please use it for all logging messages.
- **Naming Conventions:**
  - `snake_case` for variables, functions, and methods.
  - `PascalCase` for classes.
  - `UPPER_SNAKE_CASE` for constants.

## 4. Architectural Patterns

- **Event-Driven Architecture:** The system is built around an asynchronous event bus (`app/events.py`). Components communicate by publishing and subscribing to events. When adding new functionality, consider whether it can be implemented as a response to an event.
- **State Machines:** We use formal state machines (`app/state_machine.py`) to manage the state of components like the Gateway and IoT devices. When modifying the authentication protocol, please update the corresponding state machine to reflect the changes.
- **Dependency Injection:** While not strictly enforced, we prefer to pass dependencies (like the `EventBus`) into classes during initialization.

## 5. Testing

- All new features **must** be accompanied by tests.
- Unit tests are located in the `tests/` directory and should mirror the structure of the `app/` directory.
- We use `pytest` as our test runner. You can run all tests with `python run_zkpas.py --test`.
- When writing tests, please use the fixtures provided in the test files (e.g., `event_bus`).
- Aim for high test coverage. You can check coverage with `pytest --cov=app`.

## 6. Key Components

- **`run_zkpas.py`:** The main entry point for the application. It provides a CLI for running demos and tests.
- **`app/`:** The core application logic.
  - **`events.py`:** Defines the event-driven architecture.
  - **`state_machine.py`:** Implements the state machines for the protocol.
  - **`mobility_predictor.py`:** The AI-powered mobility prediction system.
  - **`components/`:** Contains the main components of the system (IoT Device, Gateway Node, etc.).
- **`shared/`:** Shared utilities, such as `crypto_utils.py` and `config.py`.
- **`demos/`:** Demonstration scripts that showcase the system's features.

When working on a specific component, please familiarize yourself with its role in the system and its interactions with other components.

Thank you for contributing to the ZKPAS project! By following these guidelines, you'll help us maintain a high-quality and consistent codebase.
