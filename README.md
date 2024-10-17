# Concrete Compressive Strength Prediction using Keras


## Project Overview

This project aims to predict the **compressive strength** of concrete using a regression model built with the **Keras** deep learning library. Predicting concrete compressive strength is vital for ensuring the reliability and safety of concrete structures, and accurate predictions can help optimize material use and reduce costs.

### Key Features:

- Data sourced from [UCI Machine Learning Repository](https://cocl.us/concrete_data).
- Built a **neural network regression model** using Keras.
- Implemented data preprocessing techniques including **normalization**.
- Evaluated model performance using **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)**.
  
---

## Dataset Description

The dataset contains information on the composition of concrete mixes and their corresponding compressive strength, measured in megapascals (MPa).

### Features (Predictors):

- **Cement** (kg in a m³ mixture)
- **Blast Furnace Slag** (kg in a m³ mixture)
- **Fly Ash** (kg in a m³ mixture)
- **Water** (kg in a m³ mixture)
- **Superplasticizer** (kg in a m³ mixture)
- **Coarse Aggregate** (kg in a m³ mixture)
- **Fine Aggregate** (kg in a m³ mixture)
- **Age** (days)

### Target Variable:

- **Strength**: Concrete compressive strength (MPa).

---


---

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **pip** package manager

### Clone the Repository

```bash
git clone https://github.com/yourusername/Concrete-Strength-Prediction.git
cd Concrete-Strength-Prediction
```

## Usage

### Running the Jupyter Notebook

1. Navigate to the notebooks directory:
```bash
cd notebooks
```
2. Launch Jupyter Notebook:

```bash
jupyter notebook
```
3. Open Concrete_Strength_Model.ipynb and run the cells sequentially to execute the project.


## Methodology

1. Data Preparation
- Import Libraries: Essential libraries such as pandas, numpy, matplotlib, tensorflow.keras, and scikit-learn are imported for data manipulation, visualization, model building, and evaluation.

- Load Data: The dataset is loaded from the provided URL using pandas.read_csv().

- Data Inspection: The first few rows and statistical summaries are examined to understand the data distribution and check for anomalies.

2. Exploratory Data Analysis
- Missing Values: Checked for any missing or null values to ensure data quality.

- Statistical Summary: Analyzed mean, standard deviation, and other statistical metrics to understand feature distributions.

- Visualization: Plotted histograms, scatter plots, and correlation matrices to visualize relationships between features and the target variable.

3. Data Preprocessing
- Feature Selection: Selected relevant predictor variables and the target variable (Strength).

- Normalization: Applied StandardScaler to normalize the feature data, ensuring that all features contribute equally to the model training.

- Train-Test Split: Split the data into training and testing sets (70% training, 30% testing) to evaluate model performance on unseen data.

4. Building the Regression Model
- Model Architecture: Constructed a Sequential neural network with:

    - Input Layer: Corresponding to the number of predictors.
    - Hidden Layers: Two hidden layers with 50 neurons each and ReLU activation.
    - Output Layer: Single neuron with linear activation for regression output.
      
- Compilation: Used the 'adam' optimizer and mean_squared_error as the loss function.

5. Training the Model
- Model Training: Trained the model for 100 epochs with a batch size of 10, using 20% of the training data for validation.

- Training Visualization: Plotted training and validation loss over epochs to monitor learning progress and detect overfitting or underfitting.

6. Evaluating the Model
- Predictions: Generated predictions on the test set.

- Performance Metrics: Calculated Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) to quantify model accuracy.

- Result Interpretation: Analyzed whether the obtained MSE is acceptable based on the dataset's context and potential application requirements.
