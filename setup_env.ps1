# Handwriting Analysis - Quick Setup Script
# This script automates the environment setup using uv

Write-Host "🖋️  Handwriting Analysis - Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if uv is installed
Write-Host "Checking for uv installation..." -ForegroundColor Yellow
$uvInstalled = Get-Command uv -ErrorAction SilentlyContinue

if (-not $uvInstalled) {
    Write-Host "❌ uv is not installed. Installing now..." -ForegroundColor Red
    Write-Host ""
    
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        Write-Host "✅ uv installed successfully!" -ForegroundColor Green
        Write-Host ""
        
        # Reload PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    }
    catch {
        Write-Host "❌ Failed to install uv. Please install manually:" -ForegroundColor Red
        Write-Host "   https://docs.astral.sh/uv/" -ForegroundColor White
        exit 1
    }
}
else {
    Write-Host "✅ uv is already installed" -ForegroundColor Green
}

# Check current directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "❌ requirements.txt not found. Please run this script from the project root." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
uv venv

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Virtual environment created successfully" -ForegroundColor Green
}
else {
    Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate virtual environment and install dependencies
Write-Host ""
Write-Host "Installing dependencies with uv (this may take a few minutes)..." -ForegroundColor Yellow
uv pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
}
else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Display next steps
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "🎉 Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Activate the virtual environment:" -ForegroundColor White
Write-Host "   .venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Set up your S3 environment variables:" -ForegroundColor White
Write-Host '   $env:S3_ENDPOINT_URL="<your-endpoint>"' -ForegroundColor Cyan
Write-Host '   $env:AWS_ACCESS_KEY_ID="<your-access-key>"' -ForegroundColor Cyan
Write-Host '   $env:AWS_SECRET_ACCESS_KEY="<your-secret-key>"' -ForegroundColor Cyan
Write-Host '   $env:S3_BUCKET="<your-bucket-name>"' -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Run the training pipeline:" -ForegroundColor White
Write-Host "   python main.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "For more information, see README.md" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
