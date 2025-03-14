# 🖋️ Handwriting Analysis with Deep Learning

A deep learning approach to analyze handwriting data for health status classification with DL techniques.

## 📋 Overview

This project implements a comprehensive deep learning pipeline for analyzing handwriting data to identify potential health conditions. It uses bidirectional recurrent neural networks and advanced preprocessing techniques to detect patterns in handwriting that may indicate health issues.

### Key Features

- **Multiple Neural Architectures**: Implementation of RNN, LSTM, GRU, Transformer, and attention-based models
- **Robust Cross-Validation**: 5-fold subject-level stratified cross-validation
- **Data Preprocessing Pipeline**: Advanced feature normalization, sliding windows, and augmentation
- **Gradient-Based Explainability**: Feature and task importance analysis
- **S3 Integration**: Seamless data handling with S3 storage
- **Configurable Pipeline**: Hydra-based configuration for easy experimentation

## 🏗️ Project Structure

```
├── conf/                               # Configuration files
│   ├── config.yaml                     # Main Hydra configuration
│   ├── model/                          # Model-specific configs
│   └── data/                           # Data processing configs
├── s3_operations/                      # S3 integration
│   ├── s3_handler.py                   # S3 initialization
│   └── s3_io.py                        # S3 IO operations
├── src/
│   ├── data/                           # Data processing modules
│   │   ├── datamodule.py               # PyTorch Lightning data module
│   │   ├── data_augmentation.py        # Data augmentation techniques
│   │   ├── balanced_batch.py           # Balanced batch sampler
│   │   └── stratified_k_fold.py        # Stratified cross-validation
│   ├── models/                         # Model implementations
│   │   ├── base.py                     # Base model class
│   │   ├── RNN.py                      # RNN implementation
│   │   ├── LSTM.py                     # LSTM implementation
│   │   ├── GRU.py                      # GRU implementation
│   │   ├── XLSTM.py                    # Extended LSTM implementation
│   │   ├── transformer_model.py        # Transformer implementation
│   │   ├── attention_RNN.py            # Attention RNN implementation
│   │   ├── han.py                      # Hierarchical Attention Network
│   │   ├── hat_net.py                  # Hierarchical Attention-Temporal Network
│   │   ├── liquid_neural_net.py        # Liquid Neural Network
│   │   └── simpleRNN.py                # Simple RNN with regularization
│   ├── explainability/                 # Explainability tools
│   │   └── model_explainer.py          # Feature and task importance analysis
│   └── utils/                          # Utility functions
│       ├── model_factory.py            # Factory pattern for model creation
│       ├── callbacks.py                # Custom PyTorch Lightning callbacks
│       ├── majority_vote.py            # Majority vote aggregation
│       ├── trainer_visualizer.py       # Training visualization utilities
│       └── print_info.py               # Information display utilities
├── main.py                             # Main training script
├── Dockerfile                          # Docker configuration
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

## 🧠 Models Implemented

- **Recurrent Neural Networks (RNN)**: Base recurrent architecture with customizable nonlinearities
- **Long Short-Term Memory Networks (LSTM)**: LSTM architecture with optional layer normalization and attention
- **Gated Recurrent Units (GRU)**: Efficient recurrent architecture with gating mechanisms
- **Extended LSTM (XLSTM)**: Enhanced LSTM with residual connections and improved regularization
- **Transformer Models**: Implementation of transformer architecture for handwriting analysis
- **Hierarchical Attention Networks (HAN)**: Multi-level attention mechanisms for feature importance
- **Liquid Neural Networks (LNN)**: Dynamic time-constant neural networks
- **Attention-Enhanced RNNs**: RNNs with task-aware attention mechanisms

## 📊 Data Processing

- **Feature Normalization**: Robust scaling and standardization
- **Sliding Window Approach**: Configurable window sizes and strides
- **Data Augmentation**: Time warping, noise addition, and smoothing
- **Class Balancing**: Weighted sampling for imbalanced data

## 📈 Performance Metrics

The model tracks multiple evaluation metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity or true positive rate
- **Specificity**: True negative rate
- **F1 Score**: Harmonic mean of precision and recall
- **Matthews Correlation Coefficient (MCC)**: Balanced measure for binary classification

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd handwriting-analysis

# Install dependencies
pip install -r requirements.txt

# Set up environment variables for S3 access
export S3_ENDPOINT_URL=<your-endpoint>
export AWS_ACCESS_KEY_ID=<your-access-key>
export AWS_SECRET_ACCESS_KEY=<your-secret-key>
export S3_BUCKET=<your-bucket-name>
```

### Running the Training Pipeline

```bash
# Basic training run
python main.py

# Specify model type and hyperparameters
python main.py model.type=lstm data.window_sizes=[60] data.strides=[2]

# Enable data augmentation
python main.py data.enable_augmentation=true

# Run in test mode (single fold)
python main.py test_mode=true
```

## 📝 Configuration

The project uses Hydra for configuration management. Key configurations in `config.yaml`:

```yaml
seed: 42
verbose: false
num_folds: 5
test_mode: false

data:
  enable_augmentation: false
  window_sizes: [50]
  strides: [20]
  batch_size: 64
  
model:
  type: "rnn"  # Options: rnn, lstm, gru, xlstm, simpleRNN, attention_rnn, han, lnn, transformer
  hidden_size: 256
  num_layers: 6
  
training:
  max_epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.00005
```

## 👥 Contributors

- [@Narden91](https://github.com/Narden91)

## 📄 License

This project is licensed under the Università degli Studi di Cassino e del Lazio Meridionale license.