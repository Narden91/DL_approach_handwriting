# Handwriting Analysis with Deep Learning

A PyTorch Lightning-based project for analyzing handwriting data using RNN architectures for health status classification.

## Project Overview

This project implements a deep learning approach to analyze handwriting data for health status classification. It uses a bidirectional RNN architecture with advanced preprocessing and cross-validation techniques to identify patterns in handwriting that may indicate health conditions.

## Project Structure

```
├── conf/
│   └── config.yaml                        # Hydra configuration file
├── s3_operations/
│   ├── s3_handler.py                      # S3 initialization
│   └── s3_io.py                           # S3 IO operations handling
├── src/
│   ├── data/
│   │   └── datamodule.py                  # Data loading and preprocessing
│   ├── models/
│   │   ├── base.py                        # Base model class
│   │   ├── RNN.py                         # RNN model
│   │   └── GRU.py                         # GRU model
│   │   └── LSTM.py                        # LSTM model 
│   └── utils/
│       ├── trainer_visualizer.py          # Plot Functions
│       ├── majority_vote.py               # Implement Majority Vote Strategy
│       └── print_info.py                  # Utility functions for information display
├── main.py                                # Main training script
```

## Key Components

### Data Module (`datamodule.py`)
- `HandwritingDataset`: Custom PyTorch dataset for handwriting data
  - Handles sliding window creation
  - Implements feature normalization
  - Manages data preprocessing
- `HandwritingDataModule`: PyTorch Lightning data module
  - Manages train/val/test splits
  - Implements k-fold cross-validation
  - Handles data loading and preprocessing

### Models
- `BaseModel` (`base.py`): Abstract base class for models
  - Defines common training and validation steps
  - Implements metric tracking
- `RNN` (`RNN.py`): Main model implementation
  - Bidirectional RNN architecture
  - Layer normalization and dropout
  - Advanced gradient checking and debugging capabilities

### Training (`main.py`)
- Implements 5-fold cross-validation
- Handles model training and validation
- Includes early stopping and checkpointing
- Provides detailed metrics logging

## Features

- **Data Preprocessing**
  - Robust handling of categorical variables
  - Advanced feature scaling (Standard/Robust)
  - Sliding window approach for temporal data
  - Automatic handling of missing values

- **Model Architecture**
  - Bidirectional RNN
  - Batch normalization
  - Dropout regularization
  - Xavier initialization

- **Training Features**
  - K-fold cross-validation
  - Early stopping
  - Gradient clipping
  - Comprehensive metrics tracking
  - GPU acceleration support

## Configuration

The project uses Hydra for configuration management. Key configurations in `config.yaml`

## Models Implemented
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory Networks (LSTM)
- Gated Recurrent Units (GRU)

## Metrics

The model tracks multiple performance metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## Usage

1. Configure the data path in `config.yaml`
2. Run training:
```bash
python main.py
```

## License

This project is licensed under the Università degli Studi di Cassino license.

## Contributors

@Narden91