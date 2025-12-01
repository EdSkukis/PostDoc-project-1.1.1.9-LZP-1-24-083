# Polymer Property Prediction

This project is a machine learning pipeline for predicting the properties of polymers, specifically their Glass Transition Temperature (Tg) and Polymer Class, based on their SMILES representation.

## Features

- **Data Loading & Cleaning:** Loads data from Kaggle, cleans it, and validates SMILES strings.
- **SMILES Featurization:** Converts SMILES strings into numerical features using Morgan Fingerprints and RDKit descriptors.
- **Multi-Task Modeling:** A single model predicts both a continuous value (Tg) and a categorical value (PolymerClass).
- **Training & Evaluation:** A complete pipeline for training the model, evaluating its performance, and saving the artifacts.
- **Data Analysis:** Scripts to analyze and visualize the quality of the SMILES data.
- **Unit Tests:** A suite of tests to ensure the core logic is working correctly.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <xxxx>
    cd polymer-ml
    ```

2.  **Create and activate a virtual environment:**
    It is recommended to use a virtual environment to manage dependencies. The user has specified `postdoc-env` as their environment.

    *   **Using `venv`:**
        ```bash
        python3 -m venv postdoc-env
        source postdoc-env/bin/activate
        ```
    *   **Using `conda`:**
        ```bash
        conda create -n postdoc-env python=3.9
        conda activate postdoc-env
        ```

3.  **Install dependencies:**
    The required libraries are listed in `requirements.txt`. You can install them using pip.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: I will create the `requirements.txt` file in the next step.)*

## Usage

### 1. Data Analysis

To analyze the quality of the SMILES data and generate plots, run the `analyze_data.py` script:

```bash
python3 scripts/analyze_data.py
```
This will save several plots in the `smiles_check` directory.

### 2. Model Training

To train the model, run the `train_multi_task.py` script:

```bash
python3 scripts/train_multi_task.py
```
This will:
- Load and process the data.
- Train the multi-task model.
- Evaluate the model and print the metrics.
- Save the trained model to `models_artifacts/multi_output_polymer_model.joblib`.
- Save a confusion matrix to `models_artifacts/confusion_matrix_polymer_class.png`.

### 3. Making Predictions

To make predictions on new data, you can use the `predict_example.py` script. You'll need to modify it to load your own data.

```bash
python3 scripts/predict_example.py
```

## Testing

To run the unit tests, use the `unittest` module:

```bash
python3 -m unittest discover tests
```
This command will discover and run all tests in the `tests` directory.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── scripts
│   ├── analyze_data.py
│   ├── predict_example.py
│   └── train_multi_task.py
├── src
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation
│   │   └── metrics.py
│   ├── featurizers
│   │   └── ...
│   ├── models
│   │   └── multi_task.py
│   ├── training
│   │   └── train_multi_task.py
│   └── visualization
│       └── smiles_quality.py
└── tests
    ├── test_data_loader.py
    ├── test_featurizer.py
    └── ...
```
