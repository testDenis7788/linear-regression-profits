#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linear Regression from Scratch — Predicting Restaurant Profits
Author: Denis Naumov
Description:
    This script implements a simple linear regression model to predict restaurant profits 
    based on city population using NumPy and Matplotlib. All core functions (cost, gradient, 
    gradient descent) are written manually.
"""

import numpy as np
import matplotlib.pyplot as plt


def load_data():
    """Loads dataset (population vs profit)."""
    data = np.loadtxt("data/ex1data1.txt", delimiter=",")
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def compute_cost(x, y, w, b):
    """Compute the cost function for linear regression."""
    m = len(x)
    cost = 0.0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    total_cost = cost / (2 * m)
    return total_cost


def compute_gradient(x, y, w, b):
    """Compute the gradient of the cost function with respect to w and b."""
    m = len(x)
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]
        dj_dw += error * x[i]
        dj_db += error
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


def gradient_descent(x, y, w_init, b_init, alpha, iterations):
    """Perform gradient descent to find optimal parameters."""
    w = w_init
    b = b_init
    cost_history = []

    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost = compute_cost(x, y, w, b)
        cost_history.append(cost)

        if i % (iterations // 10) == 0:
            print(f"Iteration {i:4d}: Cost={cost:.4f}, w={w:.4f}, b={b:.4f}")

    return w, b, cost_history


def predict(x, w, b):
    """Predict profit for given input x."""
    return w * x + b


def main():
    # Load data
    x_train, y_train = load_data()

    # Visualize data
    plt.scatter(x_train, y_train, color='red', marker='x')
    plt.title("City Population vs. Profit")
    plt.xlabel("Population (10,000s)")
    plt.ylabel("Profit ($10,000s)")
    plt.show()

    # Initialize parameters
    initial_w = 0.0
    initial_b = 0.0
    alpha = 0.01
    iterations = 1500

    print("Training the model...")
    w, b, cost_history = gradient_descent(x_train, y_train, initial_w, initial_b, alpha, iterations)

    print(f"\nFinal parameters:\nw = {w:.4f}, b = {b:.4f}")
    print(f"Final cost: {cost_history[-1]:.4f}")

    # Predictions
    predict1 = predict(3.5, w, b)
    predict2 = predict(7.0, w, b)
    print(f"Predicted profit for 35,000 people: {predict1:.2f}")
    print(f"Predicted profit for 70,000 people: {predict2:.2f}")

    # Plot regression line
    plt.scatter(x_train, y_train, color='red', marker='x', label='Training Data')
    plt.plot(x_train, predict(x_train, w, b), label='Linear Regression', color='blue')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
