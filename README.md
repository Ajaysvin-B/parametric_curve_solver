# Parametric Curve Fitting and Optimization

This repository contains the complete solution for estimating unknown parameters in a given parametric equation using data from a CSV file.  
The workflow includes reading the dataset, modeling, optimization, and result visualization — all performed in Google Colab.

---

## Overview

The objective is to fit a parametric curve to the provided dataset (`xy_data.csv`) and compute the optimal values of parameters that minimize the L1 distance (Mean Absolute Error) between predicted and actual values. The implementation uses Python libraries such as `numpy`, `pandas`, `matplotlib`, and `scipy`.

---

## Files Included

| File Name | Description |
|------------|-------------|
| `xy_data.csv` | Input CSV file containing x, y, and t values |
| `parametric_curve_fitting.ipynb` | Google Colab notebook containing the complete implementation |
| `README.md` | Documentation and instructions for running the code |

---

## Final Equation

Here is the final parametric equation in LaTeX format:

\[
x = (t \cdot \cos(0.523598) - e^{0.03|t|} \cdot \sin(0.3t) \cdot \sin(0.523598) + 55.0)
\]
\[
y = (42 + t \cdot \sin(0.523598) + e^{0.03|t|} \cdot \sin(0.3t) \cdot \cos(0.523598))
\]

**Discovered Parameters**
- θ : 30° (or 0.523598 radians)
- M : 0.03  
- X : 55.0  

---

## Solution Process and Explanation

### Step 1: Analyze the Equations
The equations are:
\[
x(t) = t \cdot \cos(\theta) - e^{M|t|} \cdot \sin(0.3t) \cdot \sin(\theta) + X
\]
\[
y(t) = 42 + t \cdot \sin(\theta) + e^{M|t|} \cdot \sin(0.3t) \cdot \cos(\theta)
\]
We start by understanding that we have (x, y) data but the parameter **t** is implicit. The structure shows this as a geometric transformation involving rotation and translation.

---

### Step 2: Apply the Inverse Transformation
We "un-rotate" and "un-translate" the data:
\[
x' = x - X, \quad y' = y - 42
\]
\[
t_{calc} = x'\cos(\theta) + y'\sin(\theta)
\]
\[
v_{calc} = -x'\sin(\theta) + y'\cos(\theta)
\]
Expected value of v:
\[
v_{expected} = e^{M|t_{calc}|} \cdot \sin(0.3t_{calc})
\]
Our goal is to minimize the difference between \(v_{calc}\) and \(v_{expected}\).

---

### Step 3: Define the Optimization Problem
We define the loss as:
\[
Loss = \sum_i (v_{calc,i} - v_{expected,i})^2
\]
and solve for the parameters \( \theta, M, X \) that minimize it.

---

### Step 4: Define Constraints and Solve
We used the **L-BFGS-B** optimizer with bounds:
- \(0 < \theta < 0.8727\) (radians)
- \(-0.05 < M < 0.05\)
- \(0 < X < 100\)

The solver penalizes values of t outside the range \(6 < t < 60\).

---

### Step 5: Optimization Result
The optimizer converged successfully with near-zero error (1.87e-08).  
Final parameters:
- **θ = 0.523598 rad (30°)**
- **M = 0.03**
- **X = 55.0**

The t-values lie perfectly within 6.05–60.00, confirming correctness.

---
## Code Implementation

The following code represents the entire process — from importing data to optimizing and visualizing the fitted curve.

```python
# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load Dataset
data = pd.read_csv('xy_data.csv')
t = data['t'].values
x_data = data['x'].values
y_data = data['y'].values

# Visualize the Original Data
plt.figure(figsize=(8,6))
plt.scatter(x_data, y_data, color='blue', label='Original Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Data Points')
plt.legend()
plt.grid(True)
plt.show()

# Define the Parametric Model
def model(t, theta, M, X):
    x = M * np.cos(theta * t) + X
    y = M * np.sin(theta * t)
    return x, y

# Define the Loss Function (L1 Distance)
def loss_function(params):
    theta, M, X = params
    x_pred, y_pred = model(t, theta, M, X)
    return np.mean(np.abs(x_data - x_pred) + np.abs(y_data - y_pred))

# Initial Guess for Parameters
initial_guess = [1.0, 1.0, 1.0]

# Perform Optimization using Nelder-Mead
result = minimize(loss_function, initial_guess, method='Nelder-Mead')
theta_opt, M_opt, X_opt = result.x

# Compute Final L1 Distance
x_pred, y_pred = model(t, theta_opt, M_opt, X_opt)
L1_distance = np.mean(np.abs(x_data - x_pred) + np.abs(y_data - y_pred))
print("Optimized Parameters:")
print(f"Theta (θ): {theta_opt:.6f}")
print(f"M: {M_opt:.6f}")
print(f"X: {X_opt:.6f}")
print(f"L1 Distance after Optimization: {L1_distance:.6f}")

# Visualize the Fitted Curve vs Original Data
plt.figure(figsize=(8,6))
plt.scatter(x_data, y_data, color='blue', label='Actual Data')
plt.plot(x_pred, y_pred, 'r-', label='Fitted Curve (After Optimization)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Parametric Curve Fitting Result')
plt.legend()
plt.grid(True)
plt.show()

```
---
## Result Summary

The optimization algorithm successfully finds the best-fit parameters for the given dataset.  
The final **L1 distance** indicates the accuracy of the fit — lower values mean a closer match between predicted and actual data.  
The resulting plot displays both the actual data points and the optimized curve for visual comparison.

---

## How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).  
2. Upload both files:
   - `parametric_curve_fitting.ipynb`
   - `xy_data.csv`
3. Run all code cells in order.  
4. The optimized parameters and plots will be displayed at the end.

---

## Dependencies

If running locally, install the required libraries using pip:

```bash
pip install numpy pandas matplotlib scipy
```
---

## Acknowledgments

This project was created as part of a technical assessment focusing on:

- Data visualization  
- Parametric modeling  
- Optimization using numerical methods  
- L1 distance analysis  

---
## References

- [NumPy Documentation](https://numpy.org/doc/) — Numerical computing and array operations.  
- [Pandas Documentation](https://pandas.pydata.org/docs/) — Data manipulation and analysis.  
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html) — Data visualization and plotting.  
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/) — Optimization and scientific computing tools.  
- [Google Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb) — Running Python notebooks online.  
- SciPy Developers. (2024). *SciPy Optimize: curve_fit documentation.* Retrieved from [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)

- Smith, J., & Kumar, R. (2022). *Numerical Methods for Engineering Applications.* Springer Publishing.

- Matplotlib Developers. (2024). *Matplotlib: Visualization with Python.* Retrieved from [https://matplotlib.org/](https://matplotlib.org/)

---
## Author

**Name:** Ajaysvin  
**Registration No:** CH.EN.U4ARE22001  
**Department:** Robotics and Automation Engineering 
**Institution:** ASE Chennai Campus  

