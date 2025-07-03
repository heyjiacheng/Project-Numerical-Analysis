import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from my_plot_fun import form

"""
BEAM PARAMETERS
"""

# Parameters
L = 10 # Length of the beam
n = 25  # Number of elements

# Constants
E = 200e9  # Young's modulus (Pa)
I = 1  # Beam moment of inertia (m^4)
rho = 7800.  # Density (kg/m^3)

# Other info
n_nodes = n + 1
x_nodes = np.linspace(0, L, n_nodes)  # Node positions
h = L / n  # Length of each element

# Element connectivity
elements = []
for i in range(n):
    elements.append([i, i + 1])  # Node indices of each element, [start, end]

# LOCAL MATRICES
def local_matrices(E_val, I_val, rho_val, h_val):
    # Define symbols
    E, I, h, rho, x = sp.symbols('E I h rho x')

    # Get symbolic form functions
    forms = form(x)

    # Initialize the stiffness and mass matrices
    Stiffness = sp.Matrix.zeros(4, 4)
    Mass = sp.Matrix.zeros(4, 4)

    # Compute S and M
    for i in range(4):
        for j in range(i, 4):
            # Stiffness matrix
            d2bi_dx2 = sp.diff(forms[i], x, 2)  # Second derivative of form function i
            d2bj_dx2 = sp.diff(forms[j], x, 2)  # Second derivative of form function j

            Stiffness[i, j] = E * I * sp.integrate(d2bi_dx2 * d2bj_dx2, (x, 0, h))
            Stiffness[j, i] = Stiffness[i, j]  # Symmetry

            # Mass matrix
            Mass[i, j] = rho * sp.integrate(forms[i] * forms[j], (x, 0, h))
            Mass[j, i] = Mass[i, j]  # Symmetry

    # Substitute numerical values into the symbolic matrices
    local_s = Stiffness.subs({h: h_val, E: E_val, I: I_val})
    local_m = Mass.subs({h: h_val, rho: rho_val})

    # Convert the resulting symbolic matrices to NumPy arrays
    local_s = np.array(local_s).astype(np.float64)
    local_m = np.array(local_m).astype(np.float64)

    return local_s, local_m

# GLOBAL MATRICES
def global_matrices():
    local_s, local_m = local_matrices(E, I, rho, h)  # Compute local matrices

    # Initialize global stiffness and mass matrices
    global_s = np.zeros((2 * n_nodes, 2 * n_nodes))
    global_m = np.zeros((2 * n_nodes, 2 * n_nodes))

    for element in elements:
        start = element[0]  # Assemble with 2 degrees of freedom
        global_s[2 * start:2 * start + 4, 2 * start:2 * start + 4] += local_s
        global_m[2 * start:2 * start + 4, 2 * start:2 * start + 4] += local_m

    return global_s, global_m

# Print the global stiffness matrix
global_S, global_M = global_matrices()
print("Global stiffness matrix S:")
print(np.round(global_S, 2))


