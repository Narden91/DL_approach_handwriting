# Handwriting Classification with Deep Learning

This project implements a deep learning approach for handwriting classification using LSTM networks. It processes raw handwriting data, trains an LSTM model, and evaluates its performance using cross-validation.

## Project Structure

```
handwriting_classifier/
├── config/
│   ├── main.yaml
│   ├── model/
│   │   ├── model1.yaml
│   │   └── model2.yaml
│   └── process/
│       ├── process1.yaml
│       └── process2.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
├── src/
│   ├── __init__.py
│   ├── process.py
│   ├── train_model.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_process.py
│   └── test_train_model.py
├── notebooks/
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/Narden91/DL_approach_handwriting.git
   cd handwriting_classifier
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Place your raw data files (T01.csv to T34.csv) in the `data/raw/` directory.

## Usage

The project is divided into two main steps: data processing and model training/evaluation.

### Data Processing

To process the raw data:

```
python src/process.py
```

This will read the raw CSV files, preprocess the data, and save the processed datasets in the `data/processed/` directory.

You can modify the processing parameters in `config/process/process1.yaml` or create a new configuration file and specify it when running:

```
python src/process.py process=process2
```

### Model Training and Evaluation

To train the model and evaluate its performance:

```
python src/train_model.py
```

This will train the LSTM model on the processed data, evaluate it on the test set, perform cross-validation, and save the trained model in the `data/final/` directory.

You can modify the model and training parameters in `config/model/model1.yaml` or create a new configuration file and specify it when running:

```
python src/train_model.py model=model2
```

## Configuration

The project uses Hydra for configuration management. The main configuration file is `config/main.yaml`, which includes:

- Data paths
- Training parameters
- Model selection
- Processing parameters

You can override any configuration parameter from the command line. For example:

```
python src/train_model.py model.hidden_size=256 training.learning_rate=0.0005
```

## Project Components

- `src/process.py`: Handles data loading, preprocessing, and splitting.
- `src/train_model.py`: Manages model training, evaluation, and cross-validation.
- `src/utils.py`: Contains utility functions and classes, including the LSTM model definition.
- `config/`: Contains all configuration files for data processing, model architecture, and training parameters.

## Testing

Run the tests using:

```
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

University of Cassino and the Southern Lazio