Parametric Curve Fitting and Optimization
Overview

This project performs parametric curve fitting on a given dataset to model the relationship between x and y values. The workflow begins from uploading the dataset in CSV format, visualizing the raw data, applying an initial curve fitting model, and finally optimizing the model parameters to minimize the error.

The goal is to find the best-fitting curve that accurately represents the data distribution using non-linear least squares optimization.

Project Workflow
Step 1: Upload the Dataset

The dataset file should be in .csv format.

It must contain at least two columns:

x — Independent variable

y — Dependent variable

Example CSV format:

x,y
0.0,0.0
0.5,0.4
1.0,0.9
1.5,1.3
2.0,1.8


Upload the file in the project directory.

Step 2: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


These libraries handle numerical computation, data loading, plotting, and curve optimization.

Step 3: Load and Inspect the Data
data = pd.read_csv("xy_data.csv")
print(data.head())


Reads the CSV file.

Displays the first few rows for verification.

Ensures the dataset contains the necessary columns.

Step 4: Visualize the Original Data

Before applying any model, visualize the data distribution:

plt.figure(figsize=(8, 5))
plt.scatter(data['x'], data['y'], color='blue', label='Original Data')
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Raw Data Plot - Before Optimization")
plt.legend()
plt.grid(True)
plt.show()


This step helps in understanding the data trend before fitting a model.

Step 5: Define the Model Function

A mathematical function is assumed to represent the underlying relationship between x and y.
For example, a linear model or a non-linear model like an exponential or polynomial curve:

def model_func(x, a, b, c):
    return a * np.exp(b * x) + c


Here:

a, b, c are parameters to be optimized.

Step 6: Fit the Model to the Data

Use the curve_fit function from scipy.optimize to estimate parameters:

x = data['x'].values
y = data['y'].values

params, covariance = curve_fit(model_func, x, y)
a, b, c = params
print(f"Optimized parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}")


This step computes the best-fit parameters that minimize the difference between the actual and predicted values.

Step 7: Visualize the Fitted Curve

Compare the fitted curve with the original data:

y_pred = model_func(x, a, b, c)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Original Data')
plt.plot(x, y_pred, color='red', linewidth=2, label='Fitted Curve')
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Curve Fitting Result - After Optimization")
plt.legend()
plt.grid(True)
plt.show()


This provides a clear visualization of how well the curve fits the data after optimization.

Step 8: Evaluate Model Performance

Compute the Root Mean Square Error (RMSE) and R² (Coefficient of Determination):

rmse = np.sqrt(np.mean((y - y_pred) ** 2))
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


RMSE shows the average error magnitude.

R² indicates how much variance in y is explained by the model.

Output Summary

After successful execution:

The optimized parameters (a, b, c) are displayed.

Two graphs are shown:

Raw Data Plot — before curve fitting.

Fitted Curve Plot — after optimization.

Evaluation metrics (RMSE, R²) summarize model accuracy.

File Structure
project_directory/
│
├── xy_data.csv                # Input data file
├── curve_fit_optimization.py  # Main Python script
├── README.md                  # Documentation
└── outputs/                   # (Optional) Folder to save graphs and results

Dependencies

Ensure the following Python packages are installed:

pip install numpy pandas matplotlib scipy

How to Run

Place the dataset (xy_data.csv) in the same directory as the script.

Run the script:

python curve_fit_optimization.py


View the generated plots and parameter results in the console.

Key Learning Outcomes

Understanding parametric curve fitting and model optimization.

Using SciPy’s curve_fit for non-linear regression.

Evaluating model performance using statistical metrics.

Visualizing data before and after optimization for interpretation.
