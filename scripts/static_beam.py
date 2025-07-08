import numpy as np
import matplotlib.pyplot as plt
from my_plot_fun import plot_piecewise_polynomial

def get_element_matrices(E, I, rho_A, h):
    """
    Calculates the local stiffness and mass matrices for a beam element
    using the standard analytical formulation.

    Args:
        E (float): Young's modulus
        I (float): Moment of inertia
        rho_A (float): Mass per unit length (rho * cross-sectional area)
        h (float): Length of the element

    Returns:
        (np.ndarray, np.ndarray): Local stiffness and mass matrices.
    """
    c1 = E * I / h**3
    local_s = c1 * np.array([
        [12, 6 * h, -12, 6 * h],
        [6 * h, 4 * h**2, -6 * h, 2 * h**2],
        [-12, -6 * h, 12, -6 * h],
        [6 * h, 2 * h**2, -6 * h, 4 * h**2]
    ])

    # Consistent mass matrix
    c2 = rho_A * h / 420
    local_m = c2 * np.array([
        [156, 22 * h, 54, -13 * h],
        [22 * h, 4 * h**2, 13 * h, -3 * h**2],
        [54, 13 * h, 156, -22 * h],
        [-13 * h, -3 * h**2, -22 * h, 4 * h**2]
    ])

    return local_s, local_m

def get_element_force_vector(q, h):
    """
    Calculates the consistent nodal force vector for a uniformly distributed load q.
    The returned vector is [F1, M1, F2, M2]^T.
    """
    return q * h / 12.0 * np.array([6, h, 6, -h])[:, np.newaxis]

def assemble_globals(n_elements, n_nodes, local_s, local_m, element_f):
    """
    Assembles the global stiffness matrix, mass matrix, and force vector.
    """
    n_dofs = 2 * n_nodes
    global_s = np.zeros((n_dofs, n_dofs))
    global_m = np.zeros((n_dofs, n_dofs))
    global_f = np.zeros((n_dofs, 1))

    for i in range(n_elements):
        dof_indices = np.arange(2 * i, 2 * i + 4)
        ix = np.ix_(dof_indices, dof_indices)
        global_s[ix] += local_s
        global_m[ix] += local_m
        global_f[dof_indices] += element_f

    return global_s, global_m, global_f

def apply_bcs_and_solve(K_global, f_global, beam_type, n_nodes):
    """
    Applies boundary conditions by partitioning the matrices and solves the
    static system K*u = f for the unknown displacements.
    """
    n_dofs = 2 * n_nodes

    if beam_type == "cantilever":
        # Fixed at node 0 (left end): displacement u_0=0, rotation theta_0=0
        fixed_dofs = [0, 1]
    elif beam_type == "simply_supported":
        # Pinned at node 0, roller at the end node.
        # Vertical displacement u_0=0, u_n=0
        fixed_dofs = [0, n_dofs - 2]
    else:
        raise ValueError("Unknown beam type. Use 'cantilever' or 'simply_supported'.")

    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Partition the matrices to extract the sub-system for free DOFs
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    f_free = f_global[free_dofs]

    # Solve for displacements at free DOFs
    u_free = np.linalg.solve(K_free, f_free)

    # Reconstruct the full displacement vector (including fixed DOFs, which are zero)
    u_full = np.zeros((n_dofs, 1))
    u_full[free_dofs] = u_free

    return u_full

def main():
    """
    Main function to define parameters and run the beam analysis.
    """
    # --- BEAM AND ANALYSIS PARAMETERS ---
    L = 10.0          # Length of the beam (m)
    n_elements = 25   # Number of elements
    E = 200e9         # Young's modulus (Pa)
    I = 1.0           # Moment of inertia (m^4)
    rho_A = 7800.0    # Mass per unit length (kg/m). Original `rho` was likely this.
    q = 10.0          # Uniform load (N/m)
    beam_type = "cantilever"  # "cantilever" or "simply_supported"

    # --- MESH GENERATION ---
    n_nodes = n_elements + 1
    x_nodes = np.linspace(0, L, n_nodes)
    h = L / n_elements

    # --- FINITE ELEMENT COMPUTATION ---
    # 1. Get local matrices (since all elements are identical, we compute once)
    local_s, local_m = get_element_matrices(E, I, rho_A, h)
    element_f = get_element_force_vector(q, h)

    # 2. Assemble global matrices and force vector
    global_s, global_m, global_f = assemble_globals(
        n_elements, n_nodes, local_s, local_m, element_f
    )

    # 3. Apply boundary conditions and solve for the nodal displacement vector
    u_vector = apply_bcs_and_solve(global_s, global_f, beam_type, n_nodes)

    # --- POST-PROCESSING AND VISUALIZATION ---
    print(f"Static analysis complete for a {beam_type} beam.")
    print("Global stiffness matrix S and mass matrix M have been assembled.")

    # Extract vertical displacements for plotting
    displacements = u_vector[0::2].flatten()

    # Plot nodal displacements
    plt.figure(figsize=(10, 6))
    plt.plot(x_nodes, displacements, 'o-', label='FEA Nodal Displacement')
    plt.xlabel("Position along beam (m)")
    plt.ylabel("Vertical Displacement (m)")
    plt.title(f"{beam_type.capitalize()} Beam under Uniform Load (q={q} N/m)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # Invert y-axis so that positive load (downwards) results in a downward plot
    plt.gca().invert_yaxis()
    plt.show()

    # Use the provided function to plot the smooth, piecewise polynomial result
    print("Displaying the smooth, piecewise polynomial solution shape.")
    plot_piecewise_polynomial(u_vector.flatten(), x_nodes)


if __name__ == "__main__":
    main()