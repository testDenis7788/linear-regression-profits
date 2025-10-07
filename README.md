# Linear Regression from Scratch — Predicting Restaurant Profits

This project implements **linear regression with one variable** completely from scratch using only NumPy.  
It predicts restaurant profits based on city population.  
All calculations — cost function, gradients, and gradient descent — are written manually for educational purposes.

---

## 📊 Project Overview

We have a dataset with:
- **x** — population of a city (in 10,000s)
- **y** — profit of a restaurant (in $10,000s)

Our goal is to learn parameters **w** (slope) and **b** (intercept) of a linear model:

\[
f(x) = wx + b
\]

To minimize the cost:

\[
J(w, b) = \frac{1}{2m}\sum_{i=1}^{m}(f(x_i) - y_i)^2
\]

---

## 🧠 Implemented Functions

- `compute_cost(x, y, w, b)` — computes the mean squared error cost  
- `compute_gradient(x, y, w, b)` — computes the gradient of the cost  
- `gradient_descent(x, y, w, b, alpha, iterations)` — learns optimal parameters  
- Visualization of training progress using `matplotlib`

---

## 🚀 Example Output

