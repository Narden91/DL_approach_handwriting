#!/bin/bash
# Handwriting Analysis - Quick Setup Script (Linux/macOS)
# This script automates the environment setup using uv

echo "🖋️  Handwriting Analysis - Environment Setup"
echo "=========================================="
echo ""

# Check if uv is installed
echo "Checking for uv installation..."
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing now..."
    echo ""
    
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    if [ $? -eq 0 ]; then
        echo "✅ uv installed successfully!"
        echo ""
        # Source the uv environment
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        echo "❌ Failed to install uv. Please install manually:"
        echo "   https://docs.astral.sh/uv/"
        exit 1
    fi
else
    echo "✅ uv is already installed"
fi

# Check current directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found. Please run this script from the project root."
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
uv venv

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment created successfully"
else
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies with uv (this may take a few minutes)..."
uv pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Display next steps
echo ""
echo "=========================================="
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Set up your S3 environment variables:"
echo '   export S3_ENDPOINT_URL="<your-endpoint>"'
echo '   export AWS_ACCESS_KEY_ID="<your-access-key>"'
echo '   export AWS_SECRET_ACCESS_KEY="<your-secret-key>"'
echo '   export S3_BUCKET="<your-bucket-name>"'
echo ""
echo "3. Run the training pipeline:"
echo "   python main.py"
echo ""
echo "For more information, see README.md"
echo "=========================================="
