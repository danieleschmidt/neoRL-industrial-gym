#!/bin/bash

# Post-create script for neoRL-industrial-gym development container
# This script runs after the container is created

set -e

echo "🚀 Setting up neoRL-industrial-gym development environment..."

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate neorl

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "✅ Created .env file. Please customize it for your environment."
fi

# Install the package in development mode
echo "📦 Installing neoRL-industrial-gym in development mode..."
pip install -e .[dev,test,docs]

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p data
mkdir -p logs
mkdir -p mlruns
mkdir -p cache
mkdir -p checkpoints

# Set up git configuration helpers
echo "🔧 Configuring git helpers..."
git config --global --add safe.directory /workspace
git config core.autocrlf false

# Install additional development dependencies
echo "📚 Installing additional development tools..."
pip install \
    jupyter-contrib-nbextensions \
    nbconvert \
    nbformat \
    ipywidgets

# Enable jupyter extensions
jupyter contrib nbextension install --user
jupyter nbextension enable --py widgetsnbextension

# Create jupyter config
echo "📓 Configuring Jupyter..."
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

# Set up shell aliases
echo "🔧 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# neoRL-industrial-gym development aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'

# Python development
alias py='python'
alias pip='python -m pip'
alias test='python -m pytest'
alias testv='python -m pytest -v'
alias testcov='python -m pytest --cov=neorl_industrial --cov-report=html'
alias lint='ruff check .'
alias format='black . && isort .'
alias typecheck='mypy src/'

# Git shortcuts
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'

# MLflow
alias mlflow-ui='mlflow ui --host 0.0.0.0 --port 8080'

# Jupyter
alias notebook='jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias lab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

# neoRL specific
alias train='python scripts/train.py'
alias eval='python scripts/evaluate.py'
alias benchmark='python scripts/benchmark_suite.py'
alias safety-check='python scripts/validate_safety.py'

EOF

# Create development helper scripts
echo "📜 Creating development helper scripts..."
cat > scripts/dev-setup.sh << 'EOF'
#!/bin/bash
# Development environment setup script

echo "🔄 Updating development environment..."

# Update dependencies
pip install -e .[dev,test,docs] --upgrade

# Update pre-commit hooks
pre-commit autoupdate

# Run safety checks
python scripts/validate_safety.py --quick

echo "✅ Development environment updated!"
EOF

chmod +x scripts/dev-setup.sh

# Display helpful information
echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Quick start commands:"
echo "  • test         - Run tests"
echo "  • lint         - Check code quality"
echo "  • format       - Format code"
echo "  • mlflow-ui    - Start MLflow UI"
echo "  • notebook     - Start Jupyter notebook"
echo "  • safety-check - Run safety validation"
echo ""
echo "📖 Documentation: https://neorl-industrial.readthedocs.io"
echo "🐛 Issues: https://github.com/terragon-labs/neoRL-industrial-gym/issues"
echo ""
echo "Happy coding! 🚀"