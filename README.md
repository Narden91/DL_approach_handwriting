# Handwriting Analysis using Deep Learning and stroke approach for Alzheimer's Detection

## Project Overview
This project implements a deep learning pipeline for detecting Alzheimer's disease through handwriting analysis. The system processes handwriting data from multiple tasks and uses various deep learning architectures (LSTM, Attention) for binary classification.

## Features
- Multi-task handwriting analysis
- Support for various deep learning architectures (LSTM, Attention)
- Sliding window approach for handling variable sequence lengths
- Comprehensive experiment tracking with Weights & Biases
- Hydra-based configuration management
- PyTorch Lightning implementation

## Project Structure
```
handwriting_analysis/
├── conf/                     # Configuration files
│   ├── config.yaml          # Main configuration
│   ├── data/                # Data-related configs
│   ├── model/               # Model architectures configs
│   └── training/            # Training hyperparameters
├── src/                     # Source code
│   ├── main.py             # Main training script
│   ├── data/               # Data loading and processing
│   ├── models/             # Model implementations
│   └── utils/              # Utility functions
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/...
cd handwriting_analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation
Place your CSV files in the data directory with the following naming convention:
- T01.csv to T34.csv
- Each file should contain columns for:
  - Subject_ID
  - Feature vectors (velocity, acceleration, slant)
  - Label (0: healthy, 1: Alzheimer's)
  - Segment (temporal order)

## Configuration
The project uses Hydra for configuration management. Key configuration files:
- `conf/config.yaml`: Main configuration
- `conf/data/data.yaml`: Data loading parameters
- `conf/model/lstm.yaml`: LSTM model parameters
- `conf/model/attention.yaml`: Attention model parameters
- `conf/training/training.yaml`: Training hyperparameters

## Usage

### Training
```bash
# Train with default configuration (LSTM)
python src/main.py

# Train with attention model
python src/main.py model=attention

# Modify hyperparameters
python src/main.py model=lstm data.batch_size=64 training.learning_rate=0.001
```

### Experiment Tracking
The project uses Weights & Biases for experiment tracking. To view results:
1. Login to your W&B account: `wandb login`
2. Run training
3. View results in the W&B dashboard

## Model Architectures

### LSTM Model
- Bidirectional LSTM layers
- Task-specific embeddings
- Dropout for regularization

### Attention Model
- Multi-head attention mechanism
- Positional encoding
- Task-specific embeddings

## Results
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
