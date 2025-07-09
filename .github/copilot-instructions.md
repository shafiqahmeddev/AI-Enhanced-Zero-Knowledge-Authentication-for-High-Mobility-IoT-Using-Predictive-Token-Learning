# GitHub Copilot Instructions Template

Welcome to the project! This document provides guidelines for using GitHub Copilot effectively within this repository. Following these instructions will help ensure that Copilot's suggestions are consistent with the project's architecture, coding style, and best practices.

## 1. Project Overview Template

**Project Name:** [PROJECT_NAME]  
**Description:** [Brief description of what the project does]  
**Domain:** [e.g., Security, IoT, Machine Learning, Web Development]

**Key Goals:**
- [Primary goal 1]
- [Primary goal 2]
- [Primary goal 3]
- [Performance/quality requirements]

**Target Audience:**
- [Primary users/developers]
- [Secondary stakeholders]
- [Deployment environment]

## 2. Technology Stack

### Core Technologies
- **Primary Language:** [e.g., Python 3.8+, JavaScript/TypeScript, Java 11+]
- **Framework/Platform:** [e.g., Django, React, Spring Boot, FastAPI]
- **Database:** [e.g., PostgreSQL, MongoDB, Redis]
- **Testing Framework:** [e.g., pytest, Jest, JUnit]

### Key Libraries & Dependencies
- **Category 1:** [e.g., Authentication] - `library1`, `library2`
- **Category 2:** [e.g., Data Processing] - `library3`, `library4`
- **Category 3:** [e.g., API/Web] - `library5`, `library6`
- **Category 4:** [e.g., DevOps/Infrastructure] - `library7`, `library8`

### Dependency Management
- **Primary:** `[requirements.txt|package.json|pom.xml|Cargo.toml]`
- **Lock Files:** `[requirements-lock.txt|package-lock.json|pom.xml|Cargo.lock]`
- **Update Process:** [How to update dependencies]

## 3. Coding Style & Conventions

### Code Formatting
- **Formatter:** [e.g., Black for Python, Prettier for JS, gofmt for Go]
- **Command:** `[formatting command]`
- **Configuration:** [Config file location and key settings]

### Linting & Type Checking
- **Linter:** [e.g., flake8, ESLint, Clippy]
- **Type Checker:** [e.g., mypy, TypeScript, Flow]
- **Configuration:** [Config files and rules]

### Documentation Standards
- **Docstring Style:** [e.g., Google, NumPy, JSDoc]
- **API Documentation:** [e.g., Swagger, OpenAPI, Sphinx]
- **README Requirements:** [What should be in README files]

### Naming Conventions
- **Variables/Functions:** [e.g., snake_case, camelCase]
- **Classes/Types:** [e.g., PascalCase, UpperCamelCase]
- **Constants:** [e.g., UPPER_SNAKE_CASE, SCREAMING_SNAKE_CASE]
- **Files/Directories:** [e.g., kebab-case, snake_case]

### Logging Standards
- **Library:** [e.g., loguru, winston, log4j]
- **Levels:** [DEBUG, INFO, WARNING, ERROR, CRITICAL]
- **Format:** [Structured logging format]

## 4. Architecture Patterns

### Primary Architecture
- **Pattern:** [e.g., Event-Driven, Microservices, MVC, Clean Architecture]
- **Description:** [Brief explanation of the architectural approach]
- **Key Components:** [Main architectural components]

### Design Patterns
- **Pattern 1:** [e.g., Repository Pattern] - [When to use]
- **Pattern 2:** [e.g., Factory Pattern] - [When to use]
- **Pattern 3:** [e.g., Observer Pattern] - [When to use]

### Data Flow
- **Request Flow:** [How requests flow through the system]
- **Error Handling:** [How errors are propagated and handled]
- **State Management:** [How state is managed and shared]

### Integration Patterns
- **External APIs:** [How to integrate with external services]
- **Database Access:** [ORM/query patterns]
- **Caching Strategy:** [When and how to cache]

## 5. Testing Strategy

### Testing Requirements
- **Coverage Target:** [e.g., 80%+ line coverage]
- **Test Types:** [Unit, Integration, E2E requirements]
- **Test Naming:** [How to name test files and functions]

### Testing Frameworks
- **Unit Testing:** [Framework and conventions]
- **Integration Testing:** [Framework and patterns]
- **Mocking:** [Mocking library and patterns]

### Test Organization
- **Directory Structure:** [Where tests should be located]
- **Test Data:** [How to manage test data and fixtures]
- **Test Utilities:** [Shared test utilities and helpers]

### Running Tests
- **Command:** `[test command]`
- **Coverage:** `[coverage command]`
- **CI/CD:** [How tests run in CI/CD pipeline]

## 6. Project Structure

### Core Directories
```
project-root/
├── src/[main-directory]/          # Main application code
│   ├── [component1]/              # Component/feature 1
│   ├── [component2]/              # Component/feature 2
│   └── shared/                    # Shared utilities
├── tests/                         # Test files
├── docs/                          # Documentation
├── scripts/                       # Build/deployment scripts
├── config/                        # Configuration files
└── [other-directories]/           # Project-specific directories
```

### Key Files
- **Entry Point:** `[main.py|index.js|main.go]` - Application entry point
- **Configuration:** `[config.py|config.json|settings.yml]` - Main configuration
- **Dependencies:** `[requirements.txt|package.json|go.mod]` - Dependency management
- **Documentation:** `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`

### Component Guidelines
- **[Component1]:** [Purpose and responsibilities]
- **[Component2]:** [Purpose and responsibilities]
- **Shared:** [Shared utilities and common code]

## 7. Development Workflow

### Branch Strategy
- **Main Branch:** `[main|master|develop]`
- **Feature Branches:** `[feature/description|feat/ticket-number]`
- **Release Branches:** `[release/version|hotfix/description]`

### Commit Conventions
- **Format:** `[type(scope): description]`
- **Types:** [feat, fix, docs, style, refactor, test, chore]
- **Examples:** 
  - `feat(auth): add user authentication`
  - `fix(api): resolve null pointer exception`

### Pull Request Process
1. [Step 1 - e.g., Create feature branch]
2. [Step 2 - e.g., Implement changes with tests]
3. [Step 3 - e.g., Run linting and tests]
4. [Step 4 - e.g., Create PR with description]
5. [Step 5 - e.g., Address review feedback]

### Code Review Guidelines
- **Reviewers:** [Who should review code]
- **Checklist:** [What to check during review]
- **Standards:** [Code quality standards]

## 8. Environment Setup

### Development Environment
- **Prerequisites:** [Required software and versions]
- **Setup Commands:** 
  ```bash
  [command 1]
  [command 2]
  [command 3]
  ```

### Configuration Management
- **Environment Variables:** [How to manage env vars]
- **Configuration Files:** [Local vs production config]
- **Secrets Management:** [How to handle sensitive data]

### Development Tools
- **IDE/Editor:** [Recommended setup]
- **Extensions:** [Recommended extensions/plugins]
- **Debug Configuration:** [How to debug the application]

## 9. Deployment & Operations

### Deployment Strategy
- **Environment:** [dev, staging, production]
- **Deployment Method:** [CI/CD, manual, containers]
- **Configuration:** [Environment-specific settings]

### Monitoring & Logging
- **Monitoring Tools:** [APM, metrics, alerts]
- **Log Aggregation:** [Centralized logging]
- **Health Checks:** [Application health monitoring]

### Performance Considerations
- **Optimization Guidelines:** [Performance best practices]
- **Scalability Patterns:** [How to scale the application]
- **Resource Limits:** [Memory, CPU, storage constraints]

## 10. Security Guidelines

### Security Best Practices
- **Authentication:** [How to handle auth]
- **Authorization:** [Permission/role management]
- **Input Validation:** [Data validation patterns]
- **Error Handling:** [Secure error responses]

### Data Protection
- **Sensitive Data:** [How to handle PII, secrets]
- **Encryption:** [Data encryption requirements]
- **Compliance:** [Regulatory requirements]

### Security Testing
- **Static Analysis:** [SAST tools and configuration]
- **Dependency Scanning:** [Vulnerability scanning]
- **Penetration Testing:** [Security testing process]

## 11. Troubleshooting

### Common Issues
- **Issue 1:** [Description and solution]
- **Issue 2:** [Description and solution]
- **Issue 3:** [Description and solution]

### Debugging Tips
- **Logging:** [How to enable debug logging]
- **Debugging Tools:** [Available debugging tools]
- **Performance Profiling:** [How to profile performance]

### Getting Help
- **Documentation:** [Where to find more docs]
- **Team Contacts:** [Who to contact for help]
- **Community:** [External resources and communities]

## 12. Customization Notes

### Project-Specific Adaptations
When adapting this template for your project:

1. **Replace placeholders** in `[brackets]` with your project's specific information
2. **Remove sections** that don't apply to your project
3. **Add project-specific sections** as needed
4. **Update examples** to match your project's patterns
5. **Keep it focused** - include only what's relevant for your team

### Maintenance
- **Review regularly** - Update as the project evolves
- **Keep it practical** - Focus on actionable guidance
- **Get team input** - Ensure the team agrees with the guidelines
- **Version control** - Track changes to these instructions

---

**Thank you for contributing to this project!** By following these guidelines, you'll help maintain a high-quality and consistent codebase that benefits the entire team.

*Last Updated: July 9, 2025*  
*Version: 2.0*
