import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from my_plot_fun import form

"""
=============================================================================
STATIC BEAM ANALYSIS USING FINITE ELEMENT METHOD
=============================================================================
This module performs static analysis of beams using the finite element method.
Supports both cantilever and simply supported beams under uniform loading.
"""

# =============================================================================
# BEAM PARAMETERS AND MATERIAL PROPERTIES
# =============================================================================

class BeamParameters:
    """Container for beam geometry and material properties."""
    
    def __init__(self):
        # Geometry
        self.length = 10.0                    # Total beam length (m)
        self.num_elements = 25                # Number of finite elements
        self.num_nodes = self.num_elements + 1
        self.element_length = self.length / self.num_elements
        
        # Material properties
        self.youngs_modulus = 200e9           # Young's modulus (Pa)
        self.moment_inertia = 1.0             # Moment of inertia (m^4)
        self.density = 7800.0                 # Material density (kg/m^3)
        
        # Derived properties
        self.node_positions = np.linspace(0, self.length, self.num_nodes)
        self.elements = self._create_element_connectivity()
    
    def _create_element_connectivity(self):
        """Create element connectivity matrix."""
        elements = []
        for i in range(self.num_elements):
            elements.append([i, i + 1])  # [start_node, end_node]
        return elements

# =============================================================================
# FINITE ELEMENT MATRIX COMPUTATION
# =============================================================================

class FiniteElementMatrices:
    """Handles computation of local and global finite element matrices."""
    
    def __init__(self, beam_params):
        self.beam = beam_params
        self.dofs_per_node = 2  # Displacement and rotation per node
        self.total_dofs = self.dofs_per_node * self.beam.num_nodes
    
    def compute_local_matrices(self):
        """
        Compute local stiffness and mass matrices for beam elements.
        
        Returns:
            tuple: (local_stiffness_matrix, local_mass_matrix)
        """
        # Define symbolic variables
        E, I, h, rho, x = sp.symbols('E I h rho x')
        
        # Get shape functions from external module
        shape_functions = form(x)
        
        # Initialize matrices
        stiffness_matrix = sp.Matrix.zeros(4, 4)
        mass_matrix = sp.Matrix.zeros(4, 4)
        
        # Compute matrix elements
        for i in range(4):
            for j in range(i, 4):
                # Stiffness matrix: K_ij = ∫ EI * d²N_i/dx² * d²N_j/dx² dx
                d2_shape_i = sp.diff(shape_functions[i], x, 2)
                d2_shape_j = sp.diff(shape_functions[j], x, 2)
                
                stiffness_matrix[i, j] = E * I * sp.integrate(
                    d2_shape_i * d2_shape_j, (x, 0, h)
                )
                stiffness_matrix[j, i] = stiffness_matrix[i, j]  # Symmetry
                
                # Mass matrix: M_ij = ∫ ρ * N_i * N_j dx
                mass_matrix[i, j] = rho * sp.integrate(
                    shape_functions[i] * shape_functions[j], (x, 0, h)
                )
                mass_matrix[j, i] = mass_matrix[i, j]  # Symmetry
        
        # Substitute numerical values
        substitutions = {
            h: self.beam.element_length,
            E: self.beam.youngs_modulus,
            I: self.beam.moment_inertia,
            rho: self.beam.density
        }
        
        local_stiffness = np.array(stiffness_matrix.subs(substitutions)).astype(np.float64)
        local_mass = np.array(mass_matrix.subs(substitutions)).astype(np.float64)
        
        return local_stiffness, local_mass
    
    def assemble_global_matrices(self):
        """
        Assemble global stiffness and mass matrices from local matrices.
        
        Returns:
            tuple: (global_stiffness_matrix, global_mass_matrix)
        """
        local_stiffness, local_mass = self.compute_local_matrices()
        
        # Initialize global matrices
        global_stiffness = np.zeros((self.total_dofs, self.total_dofs))
        global_mass = np.zeros((self.total_dofs, self.total_dofs))
        
        # Assemble element contributions
        for element in self.beam.elements:
            start_node = element[0]
            start_dof = self.dofs_per_node * start_node
            end_dof = start_dof + 4
            
            # Add element matrices to global matrices
            global_stiffness[start_dof:end_dof, start_dof:end_dof] += local_stiffness
            global_mass[start_dof:end_dof, start_dof:end_dof] += local_mass
        
        return global_stiffness, global_mass

# =============================================================================
# LOAD APPLICATION
# =============================================================================

class LoadApplication:
    """Handles application of loads to the beam structure."""
    
    def __init__(self, beam_params):
        self.beam = beam_params
        self.total_dofs = 2 * beam_params.num_nodes
    
    def apply_uniform_load(self, load_intensity):
        """
        Apply uniform distributed load to the beam.
        
        Args:
            load_intensity (float): Load intensity (N/m)
            
        Returns:
            np.ndarray: Force vector
        """
        force_vector = np.zeros(self.total_dofs)
        
        # Distribute load to nodes (simplified approach)
        force_per_node = load_intensity * self.beam.element_length / 2
        
        # Apply forces to displacement DOFs only (not rotation DOFs)
        for i in range(self.beam.num_elements):
            start_node_dof = 2 * i      # Displacement DOF of start node
            end_node_dof = 2 * (i + 1)  # Displacement DOF of end node
            
            force_vector[start_node_dof] += force_per_node
            force_vector[end_node_dof] += force_per_node
        
        return force_vector

# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

class BoundaryConditions:
    """Handles different types of boundary conditions."""
    
    @staticmethod
    def apply_cantilever_bc(stiffness_matrix, force_vector):
        """
        Apply cantilever boundary conditions (fixed at one end).
        
        Args:
            stiffness_matrix (np.ndarray): Global stiffness matrix
            force_vector (np.ndarray): Force vector
            
        Returns:
            tuple: (modified_stiffness, modified_force)
        """
        # Remove first two DOFs (displacement and rotation at fixed end)
        fixed_dofs = [0, 1]
        
        K_modified = np.delete(stiffness_matrix, fixed_dofs, axis=0)
        K_modified = np.delete(K_modified, fixed_dofs, axis=1)
        F_modified = np.delete(force_vector, fixed_dofs)
        
        return K_modified, F_modified
    
    @staticmethod
    def apply_simply_supported_bc(stiffness_matrix, force_vector, num_nodes):
        """
        Apply simply supported boundary conditions.
        
        Args:
            stiffness_matrix (np.ndarray): Global stiffness matrix
            force_vector (np.ndarray): Force vector
            num_nodes (int): Number of nodes
            
        Returns:
            tuple: (modified_stiffness, modified_force)
        """
        # Remove displacement DOFs at both ends (nodes 0 and n)
        fixed_dofs = [0, 2 * num_nodes - 2]  # First and last displacement DOFs
        
        K_modified = np.delete(stiffness_matrix, fixed_dofs, axis=0)
        K_modified = np.delete(K_modified, fixed_dofs, axis=1)
        F_modified = np.delete(force_vector, fixed_dofs)
        
        return K_modified, F_modified

# =============================================================================
# SOLVER AND ANALYSIS
# =============================================================================

class BeamAnalysis:
    """Main class for beam analysis."""
    
    def __init__(self, beam_type="cantilever", load_intensity=10.0):
        """
        Initialize beam analysis.
        
        Args:
            beam_type (str): Type of beam ("cantilever" or "simply_supported")
            load_intensity (float): Uniform load intensity (N/m)
        """
        self.beam_type = beam_type
        self.load_intensity = load_intensity
        
        # Initialize components
        self.beam_params = BeamParameters()
        self.fe_matrices = FiniteElementMatrices(self.beam_params)
        self.load_handler = LoadApplication(self.beam_params)
        
        # Analysis results
        self.displacement = None
        self.global_stiffness = None
        self.global_mass = None
    
    def solve(self):
        """Perform the complete beam analysis."""
        print(f"Analyzing {self.beam_type} beam with load intensity {self.load_intensity} N/m")
        
        # 1. Compute global matrices
        self.global_stiffness, self.global_mass = self.fe_matrices.assemble_global_matrices()
        print(f"Global stiffness matrix (shape: {self.global_stiffness.shape}):")
        print(np.round(self.global_stiffness, 2))
        
        # 2. Apply loads
        force_vector = self.load_handler.apply_uniform_load(self.load_intensity)
        
        # 3. Apply boundary conditions and solve
        self.displacement = self._solve_system(force_vector)
        
        print("Analysis completed successfully!")
        return self.displacement
    
    def _solve_system(self, force_vector):
        """Solve the system of equations with appropriate boundary conditions."""
        if self.beam_type == "cantilever":
            return self._solve_cantilever(force_vector)
        elif self.beam_type == "simply_supported":
            return self._solve_simply_supported(force_vector)
        else:
            raise ValueError(f"Unknown beam type: {self.beam_type}")
    
    def _solve_cantilever(self, force_vector):
        """Solve cantilever beam system."""
        K_reduced, F_reduced = BoundaryConditions.apply_cantilever_bc(
            self.global_stiffness, force_vector
        )
        
        # Solve reduced system
        displacement_reduced = np.linalg.solve(K_reduced, F_reduced)
        
        # Reconstruct full displacement vector (extract displacement DOFs only)
        full_displacement = np.zeros(self.beam_params.num_nodes)
        full_displacement[0] = 0.0  # Fixed end displacement
        full_displacement[1:] = displacement_reduced[::2]  # Every other element (displacement DOFs)
        
        return full_displacement
    
    def _solve_simply_supported(self, force_vector):
        """Solve simply supported beam system."""
        K_reduced, F_reduced = BoundaryConditions.apply_simply_supported_bc(
            self.global_stiffness, force_vector, self.beam_params.num_nodes
        )
        
        # Solve reduced system
        displacement_reduced = np.linalg.solve(K_reduced, F_reduced)
        
        # Reconstruct full displacement vector
        full_displacement = np.zeros(self.beam_params.num_nodes)
        full_displacement[0] = 0.0  # Fixed end displacement
        full_displacement[-1] = 0.0  # Fixed end displacement
        full_displacement[1:-1] = displacement_reduced[1::2]  # Interior displacement DOFs
        
        return full_displacement
    
    def plot_results(self):
        """Plot the displacement results."""
        if self.displacement is None:
            print("No results to plot. Run solve() first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.beam_params.node_positions, self.displacement, 
                'b-o', linewidth=2, markersize=6, label='Displacement')
        
        plt.xlabel('Position along beam (m)', fontsize=12)
        plt.ylabel('Displacement (m)', fontsize=12)
        plt.title(f'{self.beam_type.replace("_", " ").title()} Beam - '
                 f'Load = {self.load_intensity} N/m', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration
    BEAM_TYPE = "cantilever"  # Options: "cantilever" or "simply_supported"
    LOAD_INTENSITY = 10.0     # Load intensity in N/m
    
    # Perform analysis
    analysis = BeamAnalysis(beam_type=BEAM_TYPE, load_intensity=LOAD_INTENSITY)
    displacement_results = analysis.solve()
    
    # Display results
    print(f"\nDisplacement results:")
    print(f"Maximum displacement: {np.max(np.abs(displacement_results)):.6f} m")
    print(f"Displacement at free end: {displacement_results[-1]:.6f} m")
    
    # Plot results
    analysis.plot_results()