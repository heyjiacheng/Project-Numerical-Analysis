# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 23:58:28 2025

@author: jxy31
"""

import numpy as np

from matplotlib import pyplot as plt

# Define 4 form functions:
def form(x):
    form_1 = 1 - 3*x**2 + 2*x**3
    form_2 = x * ((x - 1)**2)
    form_3 = 3*x**2 - 2*x**3
    form_4 = x**2 * (x - 1)
    return np.array([form_1, form_2, form_3, form_4])
    
def plot_piecewise_polynomial(u: np.ndarray, x_nodes: np.ndarray) -> None:
    x_all = []
    w_all = []
    n = len(x_nodes)
    for i in range(n-1): # For each element in [x_i, x_{i+1}]:
        x_left = x_nodes[i]
        x_right = x_nodes[i+1]
        h = x_right - x_left
        
        # Discrete sampling in this element
        x_local = np.linspace(x_left, x_right, 50)
        
        # Standardization the sampling points: mapping to [0, 1]:
        xi = (x_local - x_left) / h
        
        # Get the 4 values from array_u:
        u1 = u[2*i]
        u2 = u[2*i + 1]
        u3 = u[2*i + 2]
        u4 = u[2*i + 3]
        
        # interpolated function values:
        phi = form(xi)  # np.array with shape (4, len(xi))
        w_local = u1 * phi[0] + h * u2 * phi[1] + u3 * phi[2] + h * u4 * phi[3]
        
        x_all.extend(x_local)
        w_all.extend(w_local)
    
    # Visualization:
    plt.figure(figsize=(10, 6))
    plt.plot(x_all, w_all, 'b-', linewidth=2, label='Piecewise Polynomial')
    plt.xlabel('x')
    plt.ylabel('w(x)')
    plt.title('Piecewise Polynomial')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()



"""
Validation using the original function:
y = sinx
"""

if __name__ == "__main__":
    n = 10
    x_nodes = np.linspace(0, 2 * np.pi, n)
    
    # build the vector u with length 2*n:
    u = np.zeros(2*n)
    for i in range(n):
        u[2 * i] = np.sin(x_nodes[i])
        u[2 * i + 1] = np.cos(x_nodes[i])
        
    print(x_nodes)
    print(u)
    plot_piecewise_polynomial(u, x_nodes)



