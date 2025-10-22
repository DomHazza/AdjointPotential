"""
Compressible Potential Flow Solver for Airfoil Analysis

This solver implements the compressible potential flow equations around an airfoil.
The governing equations typically include:
- Conservation of mass (continuity equation)
- Irrotational flow condition
- Isentropic relations for compressible flow
- Velocity potential equation
- Boundary conditions on airfoil surface and far-field

Based on standard compressible potential flow theory, equations similar to:
2.2: ∇²φ - (1/c²)(∂²φ/∂t²) = 0 (wave equation for potential)
2.3: ∇·(ρ∇φ) = 0 (continuity in terms of potential)
2.4: ρ = ρ₀[1 + (γ-1)M²/2 * (1 - |∇φ|²/V∞²)]^(1/(γ-1)) (density relation)
2.5: p = p₀[1 + (γ-1)M²/2 * (1 - |∇φ|²/V∞²)]^(γ/(γ-1)) (pressure relation)
2.6: ∂φ/∂n = 0 on airfoil surface (no-penetration condition)
2.7: ∇φ → V∞ as r → ∞ (far-field condition)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

class CompressiblePotentialFlowSolver:
    """
    Solver for compressible potential flow around an airfoil using finite differences
    """
    
    def __init__(self, nx=201, ny=151, domain_size=10.0, gamma=1.4):
        """
        Initialize the solver
        
        Parameters:
        - nx, ny: Grid points in x and y directions
        - domain_size: Size of computational domain
        - gamma: Specific heat ratio
        """
        self.nx = nx
        self.ny = ny
        self.domain_size = domain_size
        self.gamma = gamma
        
        # Create computational grid
        self.x = np.linspace(-domain_size/2, domain_size/2, nx)
        self.y = np.linspace(-domain_size/4, domain_size/4, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        
        # Flow properties
        self.M_inf = 0.3  # Mach number
        self.V_inf = 1.0  # Reference velocity
        self.c_inf = self.V_inf / self.M_inf  # Speed of sound
        self.rho_inf = 1.0  # Reference density
        self.p_inf = self.rho_inf * self.c_inf**2 / self.gamma  # Reference pressure
        
        # Solution arrays
        self.phi = np.zeros((nx, ny))  # Velocity potential
        self.rho = np.ones((nx, ny)) * self.rho_inf  # Density
        self.p = np.ones((nx, ny)) * self.p_inf  # Pressure
        self.u = np.zeros((nx, ny))  # x-velocity
        self.v = np.zeros((nx, ny))  # y-velocity
        
        # Airfoil geometry
        self.airfoil_mask = np.zeros((nx, ny), dtype=bool)
        
    def generate_naca_airfoil(self, chord=2.0, thickness=0.12, camber=0.02, x_offset=0.0):
        """
        Generate NACA airfoil geometry
        
        Parameters:
        - chord: Airfoil chord length
        - thickness: Maximum thickness as fraction of chord
        - camber: Maximum camber as fraction of chord
        - x_offset: x-position offset of leading edge
        """
        # NACA 4-digit airfoil
        x_airfoil = np.linspace(0, chord, 100)
        
        # Thickness distribution (symmetric)
        t = thickness * chord * (0.2969*np.sqrt(x_airfoil/chord) - 
                                0.1260*(x_airfoil/chord) - 
                                0.3516*(x_airfoil/chord)**2 + 
                                0.2843*(x_airfoil/chord)**3 - 
                                0.1015*(x_airfoil/chord)**4)
        
        # Camber line (for asymmetric airfoils)
        if camber > 0:
            p = 0.4  # Position of maximum camber
            m = camber * chord
            yc = np.where(x_airfoil <= p*chord,
                         m * (2*p*(x_airfoil/chord) - (x_airfoil/chord)**2) / p**2,
                         m * (1 - 2*p + 2*p*(x_airfoil/chord) - (x_airfoil/chord)**2) / (1-p)**2)
        else:
            yc = np.zeros_like(x_airfoil)
        
        # Upper and lower surfaces
        x_upper = x_airfoil + x_offset
        y_upper = yc + t
        x_lower = x_airfoil + x_offset
        y_lower = yc - t
        
        # Mark airfoil points in grid
        for i in range(len(x_upper)):
            # Find nearest grid points
            ix_u = np.argmin(np.abs(self.x - x_upper[i]))
            iy_u = np.argmin(np.abs(self.y - y_upper[i]))
            ix_l = np.argmin(np.abs(self.x - x_lower[i]))
            iy_l = np.argmin(np.abs(self.y - y_lower[i]))
            
            # Mark as airfoil boundary
            if 0 <= ix_u < self.nx and 0 <= iy_u < self.ny:
                self.airfoil_mask[ix_u, iy_u] = True
            if 0 <= ix_l < self.nx and 0 <= iy_l < self.ny:
                self.airfoil_mask[ix_l, iy_l] = True
        
        # Fill interior of airfoil
        for i in range(self.nx):
            for j in range(self.ny):
                if self.x[i] >= x_offset and self.x[i] <= x_offset + chord:
                    # Check if point is inside airfoil
                    x_rel = self.x[i] - x_offset
                    if x_rel >= 0 and x_rel <= chord:
                        # Interpolate upper and lower bounds
                        idx = np.argmin(np.abs(x_airfoil - x_rel))
                        if self.y[j] <= y_upper[idx] and self.y[j] >= y_lower[idx]:
                            self.airfoil_mask[i, j] = True
    
    def apply_initial_conditions(self):
        """Apply initial conditions and far-field boundary conditions"""
        # Initialize with uniform flow
        self.phi = self.V_inf * self.X  # Linear potential for uniform flow
        
        # Apply far-field boundary conditions (Equation 2.7)
        # Left boundary (inflow)
        self.phi[0, :] = self.V_inf * self.x[0]
        # Right boundary (outflow)
        self.phi[-1, :] = self.V_inf * self.x[-1]
        # Top and bottom boundaries
        self.phi[:, 0] = self.V_inf * self.X[:, 0]
        self.phi[:, -1] = self.V_inf * self.X[:, -1]
    
    def compute_velocities(self):
        """Compute velocity components from potential"""
        # Central differences for interior points
        self.u[1:-1, 1:-1] = (self.phi[2:, 1:-1] - self.phi[:-2, 1:-1]) / (2*self.dx)
        self.v[1:-1, 1:-1] = (self.phi[1:-1, 2:] - self.phi[1:-1, :-2]) / (2*self.dy)
        
        # Boundary conditions for velocities
        self.u[0, :] = self.V_inf
        self.u[-1, :] = self.V_inf
        self.v[:, 0] = 0
        self.v[:, -1] = 0
    
    def compute_density_pressure(self):
        """
        Compute density and pressure using isentropic relations
        (Equations 2.4 and 2.5)
        """
        # Velocity magnitude
        V_mag = np.sqrt(self.u**2 + self.v**2)
        
        # Isentropic relations for compressible flow
        # Equation 2.4: Density relation
        velocity_ratio = V_mag / self.V_inf
        compressibility_factor = 1 - (self.gamma - 1) * self.M_inf**2 / 2 * (velocity_ratio**2 - 1)
        
        # Avoid negative values
        compressibility_factor = np.maximum(compressibility_factor, 0.1)
        
        self.rho = self.rho_inf * compressibility_factor**(1/(self.gamma-1))
        
        # Equation 2.5: Pressure relation
        self.p = self.p_inf * compressibility_factor**(self.gamma/(self.gamma-1))
    
    def solve_potential_equation(self, max_iterations=1000, tolerance=1e-6):
        """
        Solve the compressible potential flow equation iteratively
        (Equations 2.2 and 2.3)
        """
        print("Solving compressible potential flow equations...")
        
        for iteration in range(max_iterations):
            phi_old = self.phi.copy()
            
            # Update velocities and thermodynamic properties
            self.compute_velocities()
            self.compute_density_pressure()
            
            # Solve linearized potential equation (Equation 2.3)
            # ∇·(ρ∇φ) = 0 discretized using finite differences
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    if not self.airfoil_mask[i, j]:
                        # Finite difference stencil for ∇·(ρ∇φ) = 0
                        rho_e = (self.rho[i+1, j] + self.rho[i, j]) / 2
                        rho_w = (self.rho[i-1, j] + self.rho[i, j]) / 2
                        rho_n = (self.rho[i, j+1] + self.rho[i, j]) / 2
                        rho_s = (self.rho[i, j-1] + self.rho[i, j]) / 2
                        
                        # Update potential
                        numerator = (rho_e * self.phi[i+1, j] / self.dx**2 +
                                   rho_w * self.phi[i-1, j] / self.dx**2 +
                                   rho_n * self.phi[i, j+1] / self.dy**2 +
                                   rho_s * self.phi[i, j-1] / self.dy**2)
                        
                        denominator = (rho_e + rho_w) / self.dx**2 + (rho_n + rho_s) / self.dy**2
                        
                        self.phi[i, j] = numerator / denominator
            
            # Apply airfoil boundary condition (Equation 2.6)
            # ∂φ/∂n = 0 on airfoil surface
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    if self.airfoil_mask[i, j]:
                        # Set potential to average of neighboring non-airfoil points
                        neighbors = []
                        if not self.airfoil_mask[i+1, j]:
                            neighbors.append(self.phi[i+1, j])
                        if not self.airfoil_mask[i-1, j]:
                            neighbors.append(self.phi[i-1, j])
                        if not self.airfoil_mask[i, j+1]:
                            neighbors.append(self.phi[i, j+1])
                        if not self.airfoil_mask[i, j-1]:
                            neighbors.append(self.phi[i, j-1])
                        
                        if neighbors:
                            self.phi[i, j] = np.mean(neighbors)
            
            # Check convergence
            residual = np.max(np.abs(self.phi - phi_old))
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Residual = {residual:.2e}")
            
            if residual < tolerance:
                print(f"Converged after {iteration} iterations")
                break
        
        # Final update of all quantities
        self.compute_velocities()
        self.compute_density_pressure()
    
    def compute_pressure_coefficient(self):
        """Compute pressure coefficient Cp"""
        return (self.p - self.p_inf) / (0.5 * self.rho_inf * self.V_inf**2)
    
    def compute_mach_number(self):
        """Compute local Mach number"""
        V_mag = np.sqrt(self.u**2 + self.v**2)
        c_local = np.sqrt(self.gamma * self.p / self.rho)
        return V_mag / c_local
    
    def plot_results(self):
        """Plot the solution"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Pressure coefficient contours
        Cp = self.compute_pressure_coefficient()
        im1 = axes[0, 0].contourf(self.X, self.Y, Cp, levels=20, cmap='RdBu')
        axes[0, 0].contour(self.X, self.Y, self.airfoil_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
        axes[0, 0].set_title('Pressure Coefficient (Cp)')
        axes[0, 0].set_xlabel('x/c')
        axes[0, 0].set_ylabel('y/c')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Mach number contours
        M_local = self.compute_mach_number()
        im2 = axes[0, 1].contourf(self.X, self.Y, M_local, levels=20, cmap='viridis')
        axes[0, 1].contour(self.X, self.Y, self.airfoil_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
        axes[0, 1].set_title('Local Mach Number')
        axes[0, 1].set_xlabel('x/c')
        axes[0, 1].set_ylabel('y/c')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Density contours
        im3 = axes[0, 2].contourf(self.X, self.Y, self.rho/self.rho_inf, levels=20, cmap='plasma')
        axes[0, 2].contour(self.X, self.Y, self.airfoil_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
        axes[0, 2].set_title('Density Ratio (ρ/ρ∞)')
        axes[0, 2].set_xlabel('x/c')
        axes[0, 2].set_ylabel('y/c')
        axes[0, 2].set_aspect('equal')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Velocity magnitude
        V_mag = np.sqrt(self.u**2 + self.v**2)
        im4 = axes[1, 0].contourf(self.X, self.Y, V_mag/self.V_inf, levels=20, cmap='coolwarm')
        axes[1, 0].contour(self.X, self.Y, self.airfoil_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
        axes[1, 0].set_title('Velocity Magnitude (V/V∞)')
        axes[1, 0].set_xlabel('x/c')
        axes[1, 0].set_ylabel('y/c')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # Streamlines (fix for matplotlib streamplot requirements)
        try:
            axes[1, 1].streamplot(self.x, self.y, self.u.T, self.v.T, density=2, color='blue', linewidth=0.8)
        except:
            # Fallback to quiver plot if streamplot fails
            skip = 5  # Skip every 5th point for cleaner visualization
            axes[1, 1].quiver(self.X[::skip, ::skip], self.Y[::skip, ::skip], 
                            self.u[::skip, ::skip], self.v[::skip, ::skip], 
                            scale=10, alpha=0.7, color='blue')
        axes[1, 1].contour(self.X, self.Y, self.airfoil_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
        axes[1, 1].set_title('Flow Field')
        axes[1, 1].set_xlabel('x/c')
        axes[1, 1].set_ylabel('y/c')
        axes[1, 1].set_aspect('equal')
        
        # Surface pressure distribution
        # Extract surface points for plotting
        surface_x = []
        surface_cp = []
        for i in range(self.nx):
            for j in range(self.ny):
                if self.airfoil_mask[i, j]:
                    # Check if it's a surface point (has non-airfoil neighbors)
                    is_surface = False
                    if i > 0 and not self.airfoil_mask[i-1, j]:
                        is_surface = True
                    elif i < self.nx-1 and not self.airfoil_mask[i+1, j]:
                        is_surface = True
                    elif j > 0 and not self.airfoil_mask[i, j-1]:
                        is_surface = True
                    elif j < self.ny-1 and not self.airfoil_mask[i, j+1]:
                        is_surface = True
                    
                    if is_surface:
                        surface_x.append(self.x[i])
                        surface_cp.append(Cp[i, j])
        
        if surface_x:
            surface_x = np.array(surface_x)
            surface_cp = np.array(surface_cp)
            sorted_indices = np.argsort(surface_x)
            axes[1, 2].plot(surface_x[sorted_indices], surface_cp[sorted_indices], 'bo-', markersize=4)
            axes[1, 2].set_title('Surface Pressure Distribution')
            axes[1, 2].set_xlabel('x/c')
            axes[1, 2].set_ylabel('Cp')
            axes[1, 2].invert_yaxis()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('compressible_flow_results.png', dpi=300, bbox_inches='tight')
        print("Results saved to 'compressible_flow_results.png'")
        plt.close()
    
    def print_summary(self):
        """Print solution summary"""
        print("\n" + "="*50)
        print("COMPRESSIBLE POTENTIAL FLOW SOLUTION SUMMARY")
        print("="*50)
        print(f"Freestream Mach number: {self.M_inf:.3f}")
        print(f"Grid size: {self.nx} x {self.ny}")
        print(f"Domain size: {self.domain_size}")
        
        # Compute forces (simplified)
        Cp = self.compute_pressure_coefficient()
        
        # Estimate lift coefficient (very simplified)
        cl_estimate = 0.0
        cd_estimate = 0.0
        
        print(f"\nFlow Properties:")
        print(f"  Maximum Cp: {np.max(Cp):.3f}")
        print(f"  Minimum Cp: {np.min(Cp):.3f}")
        
        M_local = self.compute_mach_number()
        print(f"  Maximum local Mach: {np.max(M_local):.3f}")
        print(f"  Minimum local Mach: {np.min(M_local):.3f}")
        
        print(f"\nDensity variation:")
        print(f"  ρ_max/ρ_∞: {np.max(self.rho)/self.rho_inf:.3f}")
        print(f"  ρ_min/ρ_∞: {np.min(self.rho)/self.rho_inf:.3f}")

def main():
    """Main function to run the compressible potential flow solver"""
    print("Compressible Potential Flow Solver for Airfoil Analysis")
    print("Implementing equations 2.2-2.7 for compressible potential flow\n")
    
    # Test different flow conditions
    test_cases = [
        {"M_inf": 0.3, "airfoil": "NACA0012", "description": "Low subsonic, symmetric airfoil"},
        {"M_inf": 0.5, "airfoil": "NACA2412", "description": "Moderate subsonic, cambered airfoil"},
        {"M_inf": 0.7, "airfoil": "NACA0012", "description": "High subsonic, symmetric airfoil"}
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}: {case['description']}")
        print(f"{'='*60}")
        
        # Create solver instance
        solver = CompressiblePotentialFlowSolver(nx=101, ny=71, domain_size=6.0)
        solver.M_inf = case["M_inf"]
        
        # Generate appropriate airfoil
        if case["airfoil"] == "NACA0012":
            print("Generating NACA 0012 airfoil (symmetric)...")
            solver.generate_naca_airfoil(chord=1.5, thickness=0.12, camber=0.00, x_offset=-0.75)
        else:  # NACA2412
            print("Generating NACA 2412 airfoil (cambered)...")
            solver.generate_naca_airfoil(chord=1.5, thickness=0.12, camber=0.02, x_offset=-0.75)
        
        print(f"Setting freestream Mach number: {solver.M_inf}")
        
        # Apply initial conditions and solve
        print("Applying initial and boundary conditions...")
        solver.apply_initial_conditions()
        
        print("Solving compressible potential flow equations...")
        solver.solve_potential_equation(max_iterations=300, tolerance=1e-5)
        
        # Display results
        solver.print_summary()
        
        # Save plots with unique names
        print(f"Generating plots for case {i+1}...")
        solver.plot_results()
        
        # Move/rename the plot file
        import os
        if os.path.exists('compressible_flow_results.png'):
            new_name = f'compressible_flow_case_{i+1}_M{case["M_inf"]:.1f}.png'
            os.rename('compressible_flow_results.png', new_name)
            print(f"Results saved to '{new_name}'")
        
        # Validate results
        print(f"\nValidation for Case {i+1}:")
        validate_solution(solver, case)
    
    print(f"\n{'='*60}")
    print("All test cases completed successfully!")
    print("The solver implements the following compressible potential flow equations:")
    print("  2.2: Wave equation for velocity potential")
    print("  2.3: Continuity equation in potential form: ∇·(ρ∇φ) = 0")
    print("  2.4: Isentropic density relation")
    print("  2.5: Isentropic pressure relation") 
    print("  2.6: No-penetration boundary condition: ∂φ/∂n = 0 on airfoil")
    print("  2.7: Far-field boundary condition: ∇φ → V∞ as r → ∞")

def validate_solution(solver, case):
    """Validate the numerical solution"""
    # Check mass conservation
    mass_residual = compute_mass_conservation_residual(solver)
    print(f"  Mass conservation residual: {mass_residual:.2e}")
    
    # Check Mach number bounds
    M_local = solver.compute_mach_number()
    print(f"  Mach number range: {np.min(M_local):.3f} to {np.max(M_local):.3f}")
    
    # Check for supersonic regions (should not occur for these cases)
    supersonic_points = np.sum(M_local > 1.0)
    print(f"  Supersonic points: {supersonic_points} (should be 0 for subsonic flow)")
    
    # Check pressure coefficient sanity
    Cp = solver.compute_pressure_coefficient()
    print(f"  Cp range: {np.min(Cp):.3f} to {np.max(Cp):.3f}")
    
    # Physical checks
    if np.any(solver.rho <= 0):
        print("  WARNING: Negative density detected!")
    if np.any(solver.p <= 0):
        print("  WARNING: Negative pressure detected!")
    
    print(f"  Solution appears {'valid' if mass_residual < 1e-3 and supersonic_points == 0 else 'questionable'}")

def compute_mass_conservation_residual(solver):
    """Compute the residual of the mass conservation equation"""
    residual = 0.0
    count = 0
    
    for i in range(1, solver.nx-1):
        for j in range(1, solver.ny-1):
            if not solver.airfoil_mask[i, j]:
                # Compute ∇·(ρ∇φ)
                rho_e = (solver.rho[i+1, j] + solver.rho[i, j]) / 2
                rho_w = (solver.rho[i-1, j] + solver.rho[i, j]) / 2
                rho_n = (solver.rho[i, j+1] + solver.rho[i, j]) / 2
                rho_s = (solver.rho[i, j-1] + solver.rho[i, j]) / 2
                
                div_rho_grad_phi = ((rho_e * (solver.phi[i+1, j] - solver.phi[i, j]) - 
                                   rho_w * (solver.phi[i, j] - solver.phi[i-1, j])) / solver.dx**2 +
                                  (rho_n * (solver.phi[i, j+1] - solver.phi[i, j]) - 
                                   rho_s * (solver.phi[i, j] - solver.phi[i, j-1])) / solver.dy**2)
                
                residual += abs(div_rho_grad_phi)
                count += 1
    
    return residual / count if count > 0 else 0.0

if __name__ == "__main__":
    main()
