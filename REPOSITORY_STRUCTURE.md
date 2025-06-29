# Repository Structure and File Exclusions

This document explains the `.gitignore` configuration and what files are excluded from version control.

## Heavy-Weight Files Excluded (Size Optimization)

### Datasets (~2.4GB)

- `Datasets/Geolife Trajectories 1.3/Data/` - GPS trajectory data (180 users, ~24k trajectories)
- `Datasets/release/taxi_log_2008_by_id/` - Beijing taxi GPS logs (~10k files)
- **Reason**: These datasets are massive and should be downloaded separately or stored in LFS
- **What's kept**: User guides and documentation PDFs

### Virtual Environment (~1.7GB)

- `zkpas/venv/` - Python virtual environment with all dependencies
- **Reason**: Virtual environments are environment-specific and easily recreated
- **Recreation**: `cd zkpas && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

### Build Artifacts

- Python bytecode (`__pycache__/`, `*.pyc`)
- Distribution files (`build/`, `dist/`, `*.egg-info/`)
- Test coverage reports (`.coverage`, `htmlcov/`)

### Academic and Research Files

- **Research Article folder** - Contains published papers and PDFs
- **LaTeX files** (`.tex`) - Academic paper source files
- **Implementation Blueprints** - Detailed specification documents
- **Reason**: These are intellectual property, large files, or contain sensitive research details

## Sensitive Files Excluded (Security)

### Environment Variables

- `.env` files containing secrets, API keys, database URLs
- **What's kept**: `.env.example` template files

### Cryptographic Materials

- Private keys (`*.key`, `*.pem`)
- Certificates (`*.crt`, `*.cert`)
- Key stores (`*.p12`, `*.pfx`)

### Database Files

- SQLite databases (`*.db`, `*.sqlite`)
- Production data directories

## Development Files Excluded

### IDE and Editor Files

- VSCode settings (`.vscode/settings.json`)
- Vim swap files (`*.swp`)
- System files (`.DS_Store`, `Thumbs.db`)

### Logs and Temporary Files

- Application logs (`*.log`)
- Temporary directories (`tmp/`, `temp/`)
- Simulation output files

## What IS Included

### Essential Configuration

- `pyproject.toml` - Build and tool configuration
- `requirements.txt` - Pinned dependencies
- `.env.example` - Environment template

### Source Code

- All Python modules in `app/`, `shared/`, `tests/`
- Documentation in `docs/` and `adr/`
- Shell scripts in `scripts/`

### Documentation

- Markdown files (`*.md`)
- Architecture Decision Records (ADRs)
- Essential PDF documentation (user guides)

## Repository Size Impact

**Before `.gitignore`**: ~4.2GB (includes datasets + venv + artifacts + research docs)
**After `.gitignore`**: ~20MB (source code + essential docs + configs only)

**Size reduction**: ~99.5% smaller, making cloning and syncing extremely fast.

## Quick Setup for New Contributors

```bash
# Clone the repository (fast - only ~50MB)
git clone <repository-url>
cd zkpas

# Set up Python environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your local settings

# Download datasets separately (if needed for research)
# Instructions in docs/data-setup.md
```

## Security Notes

- Never commit real credentials or API keys
- Use `.env.example` as a template for required environment variables
- Sensitive configuration should be provided through environment variables
- All cryptographic materials are excluded by default

## Data Management

- Large datasets should be stored in external systems (cloud storage, LFS)
- Use data versioning tools for reproducible research
- Document data sources and acquisition methods in `docs/`
- Consider using data download scripts in `scripts/data/`
