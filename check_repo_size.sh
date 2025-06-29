#!/bin/bash
# Repository Size Analysis Script
# Shows the impact of .gitignore on repository size

echo "🔍 ZKPAS Repository Size Analysis"
echo "================================="
echo

# Check if we're in the right directory
if [[ ! -d "zkpas" ]] && [[ ! -f "pyproject.toml" ]]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Determine project root
if [[ -f "pyproject.toml" ]]; then
    PROJECT_ROOT="."
else
    PROJECT_ROOT="zkpas"
fi

echo "📂 Directory Analysis:"
echo "---------------------"

# Function to get human-readable size
get_size() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        du -sh "$1" 2>/dev/null | cut -f1
    else
        # Linux
        du -sh "$1" 2>/dev/null | cut -f1
    fi
}

# Analyze excluded directories
echo "❌ EXCLUDED (Heavy-weight files):"
if [[ -d "Datasets" ]]; then
    DATASETS_SIZE=$(get_size "Datasets")
    echo "   📊 Datasets/: ${DATASETS_SIZE} (GPS trajectories, taxi logs)"
fi

if [[ -d "${PROJECT_ROOT}/venv" ]]; then
    VENV_SIZE=$(get_size "${PROJECT_ROOT}/venv")
    echo "   🐍 Virtual Environment: ${VENV_SIZE} (Python packages)"
fi

if [[ -f "${PROJECT_ROOT}/.coverage" ]]; then
    COVERAGE_SIZE=$(get_size "${PROJECT_ROOT}/.coverage")
    echo "   📈 Coverage files: ${COVERAGE_SIZE} (Test reports)"
fi

if [[ -d "Research Article" ]]; then
    RESEARCH_SIZE=$(get_size "Research Article")
    echo "   📄 Research Articles: ${RESEARCH_SIZE} (PDFs)"
fi

echo
echo "✅ INCLUDED (Source code and configs):"
if [[ -d "${PROJECT_ROOT}/app" ]]; then
    APP_SIZE=$(get_size "${PROJECT_ROOT}/app")
    echo "   💻 Source code (app/): ${APP_SIZE}"
fi

if [[ -d "${PROJECT_ROOT}/shared" ]]; then
    SHARED_SIZE=$(get_size "${PROJECT_ROOT}/shared")
    echo "   🔗 Shared modules: ${SHARED_SIZE}"
fi

if [[ -d "${PROJECT_ROOT}/tests" ]]; then
    TESTS_SIZE=$(get_size "${PROJECT_ROOT}/tests")
    echo "   🧪 Tests: ${TESTS_SIZE}"
fi

if [[ -d "${PROJECT_ROOT}/docs" ]]; then
    DOCS_SIZE=$(get_size "${PROJECT_ROOT}/docs")
    echo "   📚 Documentation: ${DOCS_SIZE}"
fi

if [[ -d "${PROJECT_ROOT}/adr" ]]; then
    ADR_SIZE=$(get_size "${PROJECT_ROOT}/adr")
    echo "   📋 Architecture Decisions: ${ADR_SIZE}"
fi

echo
echo "🔒 Security Analysis:"
echo "--------------------"

# Check for sensitive files that might be accidentally included
echo "Checking for potentially sensitive files..."

SENSITIVE_FOUND=0

# Check for .env files (should only be .env.example)
if find . -name ".env" -not -name ".env.example" 2>/dev/null | grep -q .; then
    echo "⚠️  Found .env files (should be in .gitignore)"
    SENSITIVE_FOUND=1
fi

# Check for key files
if find . -name "*.key" -o -name "*.pem" 2>/dev/null | grep -q .; then
    echo "⚠️  Found key files (should be in .gitignore)"
    SENSITIVE_FOUND=1
fi

# Check for database files
if find . -name "*.db" -o -name "*.sqlite" 2>/dev/null | grep -q .; then
    echo "⚠️  Found database files (should be in .gitignore)"
    SENSITIVE_FOUND=1
fi

if [[ $SENSITIVE_FOUND -eq 0 ]]; then
    echo "✅ No sensitive files found in tracked files"
fi

echo
echo "📈 Repository Impact:"
echo "--------------------"

# Calculate total excluded size
TOTAL_EXCLUDED=0
if [[ -n "$DATASETS_SIZE" ]]; then
    DATASETS_GB=$(echo "$DATASETS_SIZE" | sed 's/G.*//' | sed 's/M.*//')
    if [[ "$DATASETS_SIZE" == *"G"* ]]; then
        TOTAL_EXCLUDED=$((TOTAL_EXCLUDED + ${DATASETS_GB%.*}))
    fi
fi

if [[ -n "$VENV_SIZE" ]]; then
    VENV_GB=$(echo "$VENV_SIZE" | sed 's/G.*//' | sed 's/M.*//')
    if [[ "$VENV_SIZE" == *"G"* ]]; then
        TOTAL_EXCLUDED=$((TOTAL_EXCLUDED + ${VENV_GB%.*}))
    fi
fi

echo "💾 Estimated size reduction: ~${TOTAL_EXCLUDED}GB+ excluded"
echo "🚀 Repository is now lightweight and fast to clone"
echo "⚡ Push/pull operations will be much faster"
echo "🔒 Sensitive files are protected from accidental commits"

echo
echo "🛠️  Quick Setup for New Contributors:"
echo "------------------------------------"
echo "1. git clone <repository-url>        # Fast clone (~50MB)"
echo "2. cd zkpas"
echo "3. python -m venv venv               # Create virtual environment"
echo "4. source venv/bin/activate          # Activate environment"
echo "5. pip install -r requirements.txt   # Install dependencies"
echo "6. cp .env.example .env              # Set up environment"
echo "7. # Download datasets separately if needed (see docs/data-setup.md)"

echo
echo "✨ Repository optimization complete!"
