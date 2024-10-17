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
