import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json

from trainers.base_model import BiotCoupled2D, BiotCoupledTrainer
class BiotCoupled2D_Heterogeneous(BiotCoupled2D):
    """
    Heterogeneous Biot model for CO2 storage in layered reservoir.
    Includes caprock layer and heterogeneous reservoir properties.
    """
    
    @staticmethod
    def init_params(E_base=5000.0, nu=0.25, alpha=0.8, k_base=1.0, mu=1.0):
        """Initialize with heterogeneous material parameters for CO2 storage"""
        
        # Base parameters (will be made spatially varying)
        E_base = jnp.array(E_base, dtype=jnp.float32)
        nu = jnp.array(nu, dtype=jnp.float32)
        alpha = jnp.array(alpha, dtype=jnp.float32)
        k_base = jnp.array(k_base, dtype=jnp.float32)
        mu = jnp.array(mu, dtype=jnp.float32)
        
        # Note: G and lam will be computed locally based on heterogeneous E
        # Store base values for reference
        G_base = E_base / (2.0 * (1.0 + nu))
        lam_base = E_base * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        static_params = {
            "dims": (3, 2),  # 3 outputs (u_x, u_y, p), 2 inputs (x, y)
            # Base parameters
            "E_base": E_base,
            "nu": nu,
            "G_base": G_base,
            "lam_base": lam_base,
            "k_base": k_base,
            "mu": mu,
            "alpha": alpha,
            # Heterogeneity flags
            "heterogeneous": True,
            "caprock_depth": 0.8,  # y-coordinate where caprock starts
            "transition_depth": 0.6  # y-coordinate where transition zone starts
        }
        trainable_params = {}
        return static_params, trainable_params
    
    @staticmethod
    def compute_heterogeneous_fields(x_batch, params):
        """
        Compute spatially varying material properties for CO2 storage formation.
        
        Returns:
            k: Permeability field [mD]
            E: Young's modulus field [Pa]
            G: Shear modulus field
            lam: Lame parameter field
        """
        x = x_batch[:, 0]
        y = x_batch[:, 1]
        
        caprock_depth = params["caprock_depth"]
        transition_depth = params["transition_depth"]
        E_base = params["E_base"]
        k_base = params["k_base"]
        nu = params["nu"]
        
        # Permeability field (key for CO2 storage!)
        # Caprock: very low permeability (sealing layer)
        # Transition: intermediate
        # Reservoir: heterogeneous high permeability
        k_caprock = 1e-5 * k_base  # 0.00001 mD - sealing
        k_transition = 1e-2 * k_base  # 0.01 mD - semi-permeable
        k_reservoir = 100 * k_base * (1.0 + 0.3*jnp.sin(4*jnp.pi*x))  # Heterogeneous
        
        k = jnp.where(y > caprock_depth, k_caprock,
                      jnp.where(y > transition_depth, k_transition, k_reservoir))
        
        # Young's modulus field
        # Caprock: stiffer (more competent rock)
        # Reservoir: softer (more porous)
        E_caprock = 3.0 * E_base  # Stiffer caprock
        E_reservoir = E_base * (0.8 + 0.2*jnp.sin(2*jnp.pi*x))  # Slight heterogeneity

        E = jnp.where(y > caprock_depth, E_caprock, E_reservoir)
        
        # Compute derived elastic parameters
        G = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
        return k, E, G, lam
    
    @staticmethod
    def loss_fn(all_params, constraints):
        """
        Loss function with heterogeneous parameters for CO2 storage simulation.
        """
        # Get base parameters
        nu = all_params["static"]["problem"]["nu"]
        alpha = all_params["static"]["problem"]["alpha"]
        
        # Physics constraints
        x_batch_phys = constraints[0][0]
        
        # Compute heterogeneous fields at physics points
        k, E, G, lam = BiotCoupled2D_Heterogeneous.compute_heterogeneous_fields(
            x_batch_phys, all_params["static"]["problem"]
        )
        
        # Unpack derivatives (computed after constraining_fn)
        duxdx, d2uxdx2, d2uxdy2, d2uxdxdy, duydy, d2uydx2, d2uydy2, d2uydxdy, dpdx, dpdy, d2pdx2, d2pdy2 = constraints[0][1:13]
        
        # Compute divergence of displacement
        div_u = duxdx + duydy
        
        # Mechanics equations with heterogeneous parameters
        equilibrium_x = ((2*G + lam)*d2uxdx2 + lam*d2uydxdy + 
                        G*d2uxdy2 + G*d2uydxdy + alpha*dpdx)
        
        equilibrium_y = (G*d2uxdxdy + G*d2uydx2 + 
                        lam*d2uxdxdy + (2*G + lam)*d2uydy2 + alpha*dpdy)
        
        # mechanics_loss = jnp.mean(equilibrium_x**2) + jnp.mean(equilibrium_y**2)
        
        # Flow equation with heterogeneous permeability
        # Need to account for spatial variation of k
        # ∇·(k∇p) = k∇²p + ∇k·∇p
        laplacian_p = d2pdx2 + d2pdy2
        
        # Compute permeability gradients (using finite differences approximation)
        eps = 1e-5
        x_plus = x_batch_phys + jnp.array([eps, 0])
        x_minus = x_batch_phys - jnp.array([eps, 0])
        y_plus = x_batch_phys + jnp.array([0, eps])
        y_minus = x_batch_phys - jnp.array([0, eps])
        
        k_x_plus, _, _, _ = BiotCoupled2D_Heterogeneous.compute_heterogeneous_fields(x_plus, all_params["static"]["problem"])
        k_x_minus, _, _, _ = BiotCoupled2D_Heterogeneous.compute_heterogeneous_fields(x_minus, all_params["static"]["problem"])
        k_y_plus, _, _, _ = BiotCoupled2D_Heterogeneous.compute_heterogeneous_fields(y_plus, all_params["static"]["problem"])
        k_y_minus, _, _, _ = BiotCoupled2D_Heterogeneous.compute_heterogeneous_fields(y_minus, all_params["static"]["problem"])
        
        dk_dx = (k_x_plus - k_x_minus) / (2 * eps)
        dk_dy = (k_y_plus - k_y_minus) / (2 * eps)
        
        # Full heterogeneous flow equation
        flow_residual = -(k * laplacian_p + dk_dx * dpdx + dk_dy * dpdy) + alpha * div_u
        # flow_loss = jnp.mean(flow_residual**2)

        mechanics_char = jnp.maximum(E, 1.0)  # Local E (must be positive, avoid zero)
        flow_char = jnp.maximum(k, 1e-7)  # Local k (must be positive, avoid zero)

        mechanics_loss = jnp.mean((equilibrium_x/mechanics_char)**2 + 
                              (equilibrium_y/mechanics_char)**2)
        flow_loss = jnp.mean((flow_residual/flow_char)**2)

        # No BC penalties needed - they're hard enforced
        total_loss = mechanics_loss + flow_loss
        
        return total_loss
    
    @staticmethod
    def visualize_heterogeneous_fields(trainer):
        """
        Visualize the heterogeneous material properties of the CO2 storage formation.
        """
        # Create grid for visualization
        x = jnp.linspace(0, 1, 100)
        y = jnp.linspace(0, 1, 100)
        X, Y = jnp.meshgrid(x, y)
        points = jnp.column_stack([X.flatten(), Y.flatten()])
        
        # Compute fields
        # Use the same parameters as in the model
        params = {
            "caprock_depth": 0.8,
            "transition_depth": 0.6,
            "E_base": 5000.0,
            "k_base": 1.0,
            "nu": 0.25
        }
        
        k, E, G, lam = BiotCoupled2D_Heterogeneous.compute_heterogeneous_fields(
            points, {"caprock_depth": 0.8, "transition_depth": 0.6, 
                    "E_base": 5000.0, "k_base": 1.0, "nu": 0.25}
        )
        
        # Reshape for plotting
        k_grid = k.reshape(100, 100)
        E_grid = E.reshape(100, 100)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Permeability field (log scale)
        im1 = axes[0].contourf(X, Y, jnp.log10(k_grid), levels=20, cmap='viridis')
        axes[0].set_title('Log10 Permeability Field [log10(mD)]')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].axhline(y=0.8, color='r', linestyle='--', label='Caprock')
        axes[0].axhline(y=0.6, color='orange', linestyle='--', label='Transition')
        axes[0].legend()
        plt.colorbar(im1, ax=axes[0])
        
        # Young's modulus field
        im2 = axes[1].contourf(X, Y, E_grid, levels=20, cmap='plasma')
        axes[1].set_title("Young's Modulus Field [Pa]")
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].axhline(y=0.8, color='r', linestyle='--')
        axes[1].axhline(y=0.6, color='orange', linestyle='--')
        plt.colorbar(im2, ax=axes[1])
        
        # CO2 storage schematic
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].set_aspect('equal')
        
        # Draw layers
        axes[2].fill_between([0, 1], [0.8, 0.8], [1, 1], 
                            color='gray', alpha=0.7, label='Caprock (seal)')
        axes[2].fill_between([0, 1], [0.6, 0.6], [0.8, 0.8], 
                            color='yellow', alpha=0.5, label='Transition zone')
        axes[2].fill_between([0, 1], [0, 0], [0.6, 0.6], 
                            color='sandybrown', alpha=0.5, label='Reservoir')
        
        # Add CO2 plume schematic
        circle = plt.Circle((0.2, 0.4), 0.15, color='green', alpha=0.6, label='CO2 plume')
        axes[2].add_patch(circle)
        
        # Add injection point
        axes[2].plot(0, 0.4, 'rv', markersize=12, label='Injection')
        
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].set_title('CO2 Storage Formation Schematic')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('Heterogeneous CO2 Storage Formation Properties', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig, axes
    
    @staticmethod
    def visualize_co2_injection_response(trainer, injection_x=0.2):
        """
        Visualize pressure and deformation response to CO₂ injection
        Shows vertical slice through injection point
        """
        # Create vertical slice through injection point
        y = jnp.linspace(0, 1, 100)
        x = injection_x * jnp.ones_like(y)
        points = jnp.column_stack([x, y])
        
        # Get predictions
        pred = trainer.predict(points)
        ux = pred[:, 0]
        uy = pred[:, 1]
        p = pred[:, 2]
        
        # Get material properties along this line
        params = {"caprock_depth": 0.8, "transition_depth": 0.6, 
                "E_base": 5000.0, "k_base": 1.0, "nu": 0.25}
        k, E, _, _ = BiotCoupled2D_Heterogeneous.compute_heterogeneous_fields(points, params)
        
        # Create figure with multiple panels
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Material properties
        ax1 = plt.subplot(2, 4, 1)
        ax1.semilogy(k, y, 'b-', linewidth=2)
        ax1.set_ylabel('Depth (y)')
        ax1.set_xlabel('Permeability [mD]')
        ax1.set_title('Formation Properties')
        ax1.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Caprock')
        ax1.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Transition')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        ax1.legend()
        
        # 2. Pressure profile
        ax2 = plt.subplot(2, 4, 2)
        ax2.plot(p, y, 'r-', linewidth=2)
        ax2.set_xlabel('Pressure [MPa]')
        ax2.set_title('Pressure Distribution')
        ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        # 3. Horizontal displacement
        ax3 = plt.subplot(2, 4, 3)
        ax3.plot(ux*1000, y, 'g-', linewidth=2)
        ax3.set_xlabel('Horizontal Displacement [mm]')
        ax3.set_title('Lateral Expansion')
        ax3.axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        ax3.invert_yaxis()
        
        # 4. Vertical displacement (CRITICAL for caprock integrity!)
        ax4 = plt.subplot(2, 4, 4)
        ax4.plot(uy*1000, y, 'm-', linewidth=2)
        ax4.set_xlabel('Vertical Displacement [mm]')
        ax4.set_title('Uplift (Critical for Integrity)')
        ax4.axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
        ax4.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5)
        ax4.axvline(x=10, color='red', linestyle=':', alpha=0.5, label='Safety limit')
        ax4.grid(True, alpha=0.3)
        ax4.invert_yaxis()
        ax4.legend()
        
        # 5. 2D pressure field
        ax5 = plt.subplot(2, 4, 5)
        x_2d = jnp.linspace(0, 1, 50)
        y_2d = jnp.linspace(0, 1, 50)
        X_2d, Y_2d = jnp.meshgrid(x_2d, y_2d)
        points_2d = jnp.column_stack([X_2d.flatten(), Y_2d.flatten()])
        pred_2d = trainer.predict(points_2d)
        P_2d = pred_2d[:, 2].reshape(50, 50)
        
        im5 = ax5.contourf(X_2d, Y_2d, P_2d, levels=20, cmap='coolwarm')
        ax5.axhline(y=0.8, color='white', linestyle='--', linewidth=2)
        ax5.axhline(y=0.6, color='yellow', linestyle='--', linewidth=2)
        ax5.axvline(x=injection_x, color='lime', linestyle='-', linewidth=2)
        ax5.set_xlabel('x')
        ax5.set_ylabel('y')
        ax5.set_title('Pressure Field (Full Domain)')
        plt.colorbar(im5, ax=ax5, label='Pressure [MPa]')
        
        # 6. Displacement magnitude
        ax6 = plt.subplot(2, 4, 6)
        U_mag = jnp.sqrt(pred_2d[:, 0]**2 + pred_2d[:, 1]**2).reshape(50, 50)
        im6 = ax6.contourf(X_2d, Y_2d, U_mag*1000, levels=20, cmap='viridis')
        ax6.axhline(y=0.8, color='white', linestyle='--', linewidth=2)
        ax6.axhline(y=0.6, color='yellow', linestyle='--', linewidth=2)
        ax6.set_xlabel('x')
        ax6.set_ylabel('y')
        ax6.set_title('Total Displacement [mm]')
        plt.colorbar(im6, ax=ax6, label='|u| [mm]')
        
        # 7. CO₂ plume estimation (based on pressure)
        ax7 = plt.subplot(2, 4, 7)
        # CO₂ extent estimated from pressure threshold
        CO2_extent = (P_2d > 0.3).astype(float)  # Threshold for CO₂ presence
        im7 = ax7.contourf(X_2d, Y_2d, CO2_extent, levels=[0, 0.5, 1], 
                        colors=['white', 'lightgreen'], alpha=0.7)
        ax7.contour(X_2d, Y_2d, P_2d, levels=[0.3], colors='green', linewidths=2)
        ax7.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Caprock')
        ax7.set_xlabel('x')
        ax7.set_ylabel('y')
        ax7.set_title('Estimated CO₂ Plume Extent')
        ax7.legend()
        
        # 8. Safety assessment
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')
        
        # Calculate safety metrics
        max_caprock_uplift = jnp.max(uy[y > 0.8]) * 1000  # mm
        max_pressure = jnp.max(p)
        pressure_at_caprock = p[jnp.argmin(jnp.abs(y - 0.8))]
        
        safety_text = f"""
        CO₂ STORAGE SAFETY ASSESSMENT
        
        Injection Point: x = {injection_x:.2f}
        
        Mechanical Integrity:
        • Max caprock uplift: {max_caprock_uplift:.2f} mm
        • Status: {'SAFE' if max_caprock_uplift < 10 else '⚠ MONITOR'}
        
        Pressure Containment:
        • Max pressure: {max_pressure:.3f} MPa
        • Pressure at caprock: {pressure_at_caprock:.3f} MPa
        • Breakthrough risk: {'Low' if pressure_at_caprock < 0.5 else 'Moderate'}
        
        CO₂ Migration:
        • Lateral extent: ~{jnp.sum(CO2_extent > 0) / CO2_extent.size * 100:.1f}% of domain
        • Vertical containment: {'Effective' if jnp.max(CO2_extent[Y_2d > 0.8]) < 0.5 else 'Check seal'}
        
        Recommendation:
        {'Continue injection' if max_caprock_uplift < 10 and pressure_at_caprock < 0.5 else 'Reduce injection rate'}
        """
        
        ax8.text(0.1, 0.5, safety_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.suptitle(f'CO₂ Injection Response Analysis (Vertical Slice at x={injection_x:.2f})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig

# Create trainer for heterogeneous problem
class HeterogeneousTrainer(BiotCoupledTrainer):
    """Trainer for heterogeneous CO2 storage problem"""
    
    def __init__(self):
        super().__init__()
        # Update problem class
        self.config.problem = BiotCoupled2D_Heterogeneous
        self.config.problem_init_kwargs = {
            'E_base': 5000.0, 
            'nu': 0.25, 
            'alpha': 0.8, 
            'k_base': 1.0, 
            'mu': 1.0
        }
        # Might need more training steps for heterogeneous problem
        self.config.n_steps = 10000

    def verify_bcs(self, n_points=100):
        """
        BC verification with CO2 storage context
        """
        print("\n" + "="*60)
        print("BOUNDARY CONDITION VERIFICATION - CO₂ STORAGE MODEL")
        print("="*60)
        print("Physical interpretation:")
        print("  • p(x=0) = 1: CO₂ injection pressure")
        print("  • p(x=1) = 0: Far-field pressure")
        print("  • ux(x=0) = 0: Fixed lateral boundary")
        print("  • uy(y=0) = 0: Fixed bottom (bedrock)")
        print("-"*60)
        
        # Call parent method
        return super().verify_bcs(n_points)

    def compute_physics_metrics(self, n_points=50, method='finite_diff'):
        """
        Complete physics validation metrics for HETEROGENEOUS CO2 storage model.
        
        Args:
            n_points: Number of points along each dimension for evaluation grid
            method: Method for computing derivatives ('finite_diff' recommended for heterogeneous)
            
        Returns:
            Dictionary containing all computed metrics including relative errors
        """
        if self.all_params is None:
            raise ValueError("Model not trained yet")
        
        print("\n" + "="*60)
        print(f"PHYSICS VALIDATION METRICS - HETEROGENEOUS MODEL")
        print(f"Method: {method}")
        print("="*60)
        
        # Create evaluation grid (avoid boundaries for finite differences)
        x = jnp.linspace(0.1, 0.9, n_points)
        y = jnp.linspace(0.1, 0.9, n_points)
        X, Y = jnp.meshgrid(x, y)
        points = jnp.column_stack([X.flatten(), Y.flatten()])
        
        # Get spatially-varying material properties at ALL evaluation points
        k_local, E_local, G_local, lam_local = BiotCoupled2D_Heterogeneous.compute_heterogeneous_fields(
            points, self.all_params["static"]["problem"]
        )
        
        # Get constant parameters
        alpha = self.all_params["static"]["problem"]["alpha"]
        nu = self.all_params["static"]["problem"]["nu"]
        
        # Predict solution at all points
        u_pred = self.predict(points)
        ux = u_pred[:, 0]
        uy = u_pred[:, 1]
        p = u_pred[:, 2]
        
        # Compute characteristic scales for normalization
        L = 1.0  # Domain size
        p_scale = 1.0  # Maximum pressure (BC value)
        u_scale = jnp.maximum(jnp.max(jnp.abs(ux)), jnp.max(jnp.abs(uy)))
        if u_scale < 1e-10:
            u_scale = p_scale * L / jnp.mean(E_local)  # Use average E
        
        # Use average properties for scaling
        E_avg = jnp.mean(E_local)
        k_avg = jnp.mean(k_local)
        elastic_force_scale = E_avg * u_scale / L**2
        pressure_gradient_scale = k_avg * p_scale / L**2
        
        # Compute derivatives using finite differences
        print("Computing derivatives using finite differences...")
        
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Reshape for finite differences
        ux_grid = ux.reshape(n_points, n_points)
        uy_grid = uy.reshape(n_points, n_points)
        p_grid = p.reshape(n_points, n_points)
        
        # First derivatives
        dux_dx = jnp.gradient(ux_grid, dx, axis=1).flatten()
        dux_dy = jnp.gradient(ux_grid, dy, axis=0).flatten()
        duy_dx = jnp.gradient(uy_grid, dx, axis=1).flatten()
        duy_dy = jnp.gradient(uy_grid, dy, axis=0).flatten()
        dp_dx = jnp.gradient(p_grid, dx, axis=1).flatten()
        dp_dy = jnp.gradient(p_grid, dy, axis=0).flatten()
        
        # Second derivatives
        d2ux_dx2 = jnp.gradient(jnp.gradient(ux_grid, dx, axis=1), dx, axis=1).flatten()
        d2ux_dy2 = jnp.gradient(jnp.gradient(ux_grid, dy, axis=0), dy, axis=0).flatten()
        d2ux_dxdy = jnp.gradient(jnp.gradient(ux_grid, dx, axis=1), dy, axis=0).flatten()
        
        d2uy_dx2 = jnp.gradient(jnp.gradient(uy_grid, dx, axis=1), dx, axis=1).flatten()
        d2uy_dy2 = jnp.gradient(jnp.gradient(uy_grid, dy, axis=0), dy, axis=0).flatten()
        d2uy_dxdy = jnp.gradient(jnp.gradient(uy_grid, dx, axis=1), dy, axis=0).flatten()
        
        d2p_dx2 = jnp.gradient(jnp.gradient(p_grid, dx, axis=1), dx, axis=1).flatten()
        d2p_dy2 = jnp.gradient(jnp.gradient(p_grid, dy, axis=0), dy, axis=0).flatten()
        
        # Compute physics residuals WITH HETEROGENEOUS PROPERTIES
        div_u = dux_dx + duy_dy
        
        # Mechanics equations with spatially-varying G and lam
        equilibrium_x = ((2*G_local + lam_local)*d2ux_dx2 + lam_local*d2uy_dxdy + 
                        G_local*d2ux_dy2 + G_local*d2uy_dxdy + alpha*dp_dx)
        
        equilibrium_y = (G_local*d2ux_dxdy + G_local*d2uy_dx2 + 
                        lam_local*d2ux_dxdy + (2*G_local + lam_local)*d2uy_dy2 + alpha*dp_dy)
        
        # Flow equation with spatially-varying k (need gradient of k for full equation)
        laplacian_p = d2p_dx2 + d2p_dy2
        
        # Compute permeability gradients
        k_grid = k_local.reshape(n_points, n_points)
        dk_dx = jnp.gradient(k_grid, dx, axis=1).flatten()
        dk_dy = jnp.gradient(k_grid, dy, axis=0).flatten()
        
        # Full heterogeneous flow equation: ∇·(k∇p) = k∇²p + ∇k·∇p
        flow_residual = -(k_local * laplacian_p + dk_dx * dp_dx + dk_dy * dp_dy) + alpha * div_u
        
        # Compute absolute error metrics
        mechanics_x_rms = jnp.sqrt(jnp.mean(equilibrium_x**2))
        mechanics_y_rms = jnp.sqrt(jnp.mean(equilibrium_y**2))
        mechanics_rms = jnp.sqrt((mechanics_x_rms**2 + mechanics_y_rms**2) / 2)
        flow_rms = jnp.sqrt(jnp.mean(flow_residual**2))
        
        # Compute L-infinity norms (maximum errors)
        mechanics_x_linf = jnp.max(jnp.abs(equilibrium_x))
        mechanics_y_linf = jnp.max(jnp.abs(equilibrium_y))
        mechanics_linf = jnp.maximum(mechanics_x_linf, mechanics_y_linf)
        flow_linf = jnp.max(jnp.abs(flow_residual))
        
        # Compute L2 norms
        mechanics_x_l2 = jnp.sqrt(jnp.sum(equilibrium_x**2))
        mechanics_y_l2 = jnp.sqrt(jnp.sum(equilibrium_y**2))
        mechanics_l2 = jnp.sqrt(mechanics_x_l2**2 + mechanics_y_l2**2)
        flow_l2 = jnp.sqrt(jnp.sum(flow_residual**2))
        
        # Compute relative error metrics
        mechanics_x_rel = mechanics_x_rms / elastic_force_scale
        mechanics_y_rel = mechanics_y_rms / elastic_force_scale
        mechanics_rel = mechanics_rms / elastic_force_scale
        flow_rel = flow_rms / pressure_gradient_scale
        
        # Determine overall status based on relative errors
        total_rel = (mechanics_rel + flow_rel) / 2
        if total_rel < 1e-3:
            status = "EXCELLENT"
        elif total_rel < 1e-2:
            status = "VERY GOOD"
        elif total_rel < 5e-2:
            status = "GOOD"
        elif total_rel < 1e-1:
            status = "ACCEPTABLE"
        else:
            status = "NEEDS IMPROVEMENT"
        
        # Store residuals for spatial distribution plotting
        self.last_residuals = {
            'x': X,
            'y': Y,
            'mechanics_x': equilibrium_x.reshape(n_points, n_points),
            'mechanics_y': equilibrium_y.reshape(n_points, n_points),
            'flow': flow_residual.reshape(n_points, n_points),
            'div_u': div_u.reshape(n_points, n_points),
            'k_field': k_local.reshape(n_points, n_points),
            'E_field': E_local.reshape(n_points, n_points)
        }
        
        # Print detailed results in a table
        print("\n┌─────────────────┬──────────────┬──────────────┬──────────────┐")
        print("│ Equation        │ RMS Absolute │ RMS Relative │ L-inf Norm   │")
        print("├─────────────────┼──────────────┼──────────────┼──────────────┤")
        print(f"│ Mechanics (x)   │ {mechanics_x_rms:.2e} │ {mechanics_x_rel:.2e} │ {mechanics_x_linf:.2e} │")
        print(f"│ Mechanics (y)   │ {mechanics_y_rms:.2e} │ {mechanics_y_rel:.2e} │ {mechanics_y_linf:.2e} │")
        print(f"│ Mechanics (avg) │ {mechanics_rms:.2e} │ {mechanics_rel:.2e} │ {mechanics_linf:.2e} │")
        print(f"│ Flow            │ {flow_rms:.2e} │ {flow_rel:.2e} │ {flow_linf:.2e} │")
        print("└─────────────────┴──────────────┴──────────────┴──────────────┘")
        
        # Print heterogeneous material property ranges
        print(f"\nHeterogeneous Material Properties:")
        print(f"  Permeability range: {jnp.min(k_local):.2e} to {jnp.max(k_local):.2e} mD")
        print(f"  Young's modulus range: {jnp.min(E_local):.0f} to {jnp.max(E_local):.0f} Pa")
        print(f"  Permeability contrast: {jnp.max(k_local)/jnp.min(k_local):.2e}")
        
        # Layer-specific analysis
        caprock_mask = points[:, 1] > 0.8
        reservoir_mask = points[:, 1] < 0.6
        
        if jnp.any(caprock_mask):
            caprock_flow_error = jnp.mean(jnp.abs(flow_residual[caprock_mask]))
            print(f"\nCaprock flow residual: {caprock_flow_error:.2e}")
        
        if jnp.any(reservoir_mask):
            reservoir_flow_error = jnp.mean(jnp.abs(flow_residual[reservoir_mask]))
            print(f"Reservoir flow residual: {reservoir_flow_error:.2e}")
        
        # Print scale information
        print(f"\nCharacteristic Scales:")
        print(f"  Domain size (L): {L:.2f}")
        print(f"  Pressure scale: {p_scale:.2f}")
        print(f"  Displacement scale: {u_scale:.2e}")
        print(f"  Elastic force scale: {elastic_force_scale:.2e}")
        print(f"  Pressure gradient scale: {pressure_gradient_scale:.2e}")
        
        print(f"\nOverall physics satisfaction: {status} (avg relative error: {total_rel:.2e})")
        
        # Conservation check
        print("\nPhysics Conservation Check:")
        print(f"  Mass conservation (avg residual): {flow_rms:.2e}")
        print(f"  Momentum conservation (avg residual): {mechanics_rms:.2e}")
        
        # CO2 storage specific metrics
        print("\nCO2 Storage Relevant Metrics:")
        pressure_buildup = jnp.max(p[reservoir_mask]) if jnp.any(reservoir_mask) else 0
        caprock_integrity = jnp.max(jnp.abs(uy[caprock_mask])) * 1000 if jnp.any(caprock_mask) else 0
        print(f"  Max pressure in reservoir: {pressure_buildup:.3f} MPa")
        print(f"  Max caprock uplift: {caprock_integrity:.2f} mm")
        print(f"  Seal integrity: {'SAFE' if caprock_integrity < 10 else 'MONITOR REQUIRED'}")
        
        print("="*60)
        
        # Return comprehensive metrics dictionary
        return {
            # Absolute metrics
            "mechanics_x_rms": float(mechanics_x_rms),
            "mechanics_y_rms": float(mechanics_y_rms),
            "mechanics_rms": float(mechanics_rms),
            "flow_rms": float(flow_rms),
            "mechanics_x_l2": float(mechanics_x_l2),
            "mechanics_y_l2": float(mechanics_y_l2),
            "mechanics_l2": float(mechanics_l2),
            "flow_l2": float(flow_l2),
            "mechanics_x_linf": float(mechanics_x_linf),
            "mechanics_y_linf": float(mechanics_y_linf),
            "mechanics_linf": float(mechanics_linf),
            "flow_linf": float(flow_linf),
            
            # Relative metrics
            "mechanics_x_rel": float(mechanics_x_rel),
            "mechanics_y_rel": float(mechanics_y_rel),
            "mechanics_rel": float(mechanics_rel),
            "flow_rel": float(flow_rel),
            "total_rel": float(total_rel),
            
            # Scales and status
            "status": status,
            "u_scale": float(u_scale),
            "p_scale": float(p_scale),
            "method": method,
            
            # Heterogeneous properties
            "k_min": float(jnp.min(k_local)),
            "k_max": float(jnp.max(k_local)),
            "E_min": float(jnp.min(E_local)),
            "E_max": float(jnp.max(E_local)),
            
            # CO2 storage metrics
            "max_pressure": float(pressure_buildup),
            "caprock_uplift_mm": float(caprock_integrity)
        }
    
    def plot_solution(self, n_points=50):
        """
        Enhanced solution plotting for heterogeneous CO2 storage model
        """
        x = jnp.linspace(0, 1, n_points)
        y = jnp.linspace(0, 1, n_points)
        X, Y = jnp.meshgrid(x, y)
        points = jnp.column_stack([X.flatten(), Y.flatten()])
        
        # Get predictions
        pred = self.predict(points)
        UX = pred[:, 0].reshape(n_points, n_points)
        UY = pred[:, 1].reshape(n_points, n_points)
        P = pred[:, 2].reshape(n_points, n_points)
        
        # Get material properties for overlay
        params = {"caprock_depth": 0.8, "transition_depth": 0.6,
                "E_base": 5000.0, "k_base": 1.0, "nu": 0.25}
        k, E, _, _ = BiotCoupled2D_Heterogeneous.compute_heterogeneous_fields(points, params)
        K_grid = k.reshape(n_points, n_points)
        
        # Create figure with 2 rows
        fig = plt.figure(figsize=(15, 10))
        
        # Row 1: Standard fields with layer boundaries
        ax1 = plt.subplot(2, 3, 1)
        c1 = ax1.contourf(X, Y, UX*1000, levels=20, cmap='RdBu')
        ax1.axhline(y=0.8, color='white', linestyle='--', linewidth=2, label='Caprock')
        ax1.axhline(y=0.6, color='yellow', linestyle='--', linewidth=2, label='Transition')
        ax1.set_title('X-Displacement [mm]')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend(loc='upper right')
        plt.colorbar(c1, ax=ax1)
        
        ax2 = plt.subplot(2, 3, 2)
        c2 = ax2.contourf(X, Y, UY*1000, levels=20, cmap='RdBu')
        ax2.axhline(y=0.8, color='white', linestyle='--', linewidth=2)
        ax2.axhline(y=0.6, color='yellow', linestyle='--', linewidth=2)
        ax2.set_title('Y-Displacement (Uplift) [mm]')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(c2, ax=ax2)
        
        ax3 = plt.subplot(2, 3, 3)
        c3 = ax3.contourf(X, Y, P, levels=20, cmap='coolwarm')
        ax3.axhline(y=0.8, color='white', linestyle='--', linewidth=2)
        ax3.axhline(y=0.6, color='yellow', linestyle='--', linewidth=2)
        ax3.set_title('Pressure [MPa]')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(c3, ax=ax3)
        
        # Row 2: CO2 storage specific visualizations
        ax4 = plt.subplot(2, 3, 4)
        # Displacement magnitude (important for integrity)
        U_mag = jnp.sqrt(UX**2 + UY**2)
        c4 = ax4.contourf(X, Y, U_mag*1000, levels=20, cmap='viridis')
        ax4.axhline(y=0.8, color='white', linestyle='--', linewidth=2)
        ax4.axhline(y=0.6, color='yellow', linestyle='--', linewidth=2)
        ax4.set_title('Total Displacement Magnitude [mm]')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        plt.colorbar(c4, ax=ax4)
        
        ax5 = plt.subplot(2, 3, 5)
        # Pressure with permeability overlay (shows trapping)
        c5 = ax5.contourf(X, Y, P, levels=20, cmap='coolwarm', alpha=0.7)
        ax5.contour(X, Y, jnp.log10(K_grid), levels=[-5, -2, 0, 2], 
                    colors='black', linewidths=1, linestyles='--')
        ax5.set_title('Pressure with Permeability Contours')
        ax5.set_xlabel('x')
        ax5.set_ylabel('y')
        plt.colorbar(c5, ax=ax5, label='Pressure [MPa]')
        
        ax6 = plt.subplot(2, 3, 6)
        # CO2 saturation proxy (based on pressure threshold)
        CO2_sat = jnp.where(P > 0.3, (P - 0.3) / (jnp.max(P) - 0.3), 0)
        c6 = ax6.contourf(X, Y, CO2_sat, levels=20, cmap='Greens')
        ax6.axhline(y=0.8, color='red', linestyle='--', linewidth=2)
        ax6.axhline(y=0.6, color='orange', linestyle='--', linewidth=2)
        ax6.set_title('Estimated CO₂ Saturation')
        ax6.set_xlabel('x')
        ax6.set_ylabel('y')
        plt.colorbar(c6, ax=ax6, label='Saturation [-]')
        
        plt.suptitle('Heterogeneous CO₂ Storage Simulation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print enhanced statistics
        print("\n" + "="*60)
        print("SOLUTION FIELD STATISTICS - HETEROGENEOUS MODEL")
        print("="*60)
        
        print("\nDisplacement Fields:")
        print(f"  ux: min={jnp.min(UX)*1000:.2f} mm, max={jnp.max(UX)*1000:.2f} mm")
        print(f"  uy: min={jnp.min(UY)*1000:.2f} mm, max={jnp.max(UY)*1000:.2f} mm")
        print(f"  |u|: max={jnp.max(U_mag)*1000:.2f} mm")
        
        print("\nPressure Field:")
        print(f"  Overall: min={jnp.min(P):.3f} MPa, max={jnp.max(P):.3f} MPa")
        
        # Layer-specific analysis
        caprock_mask = Y > 0.8
        transition_mask = (Y > 0.6) & (Y <= 0.8)
        reservoir_mask = Y <= 0.6
        
        print(f"  Caprock: min={jnp.min(P[caprock_mask]):.3f}, max={jnp.max(P[caprock_mask]):.3f} MPa")
        print(f"  Transition: min={jnp.min(P[transition_mask]):.3f}, max={jnp.max(P[transition_mask]):.3f} MPa")
        print(f"  Reservoir: min={jnp.min(P[reservoir_mask]):.3f}, max={jnp.max(P[reservoir_mask]):.3f} MPa")
        
        print("\nCO₂ Storage Assessment:")
        max_caprock_uplift = jnp.max(UY[caprock_mask]) * 1000
        pressure_gradient = (jnp.max(P[reservoir_mask]) - jnp.min(P[caprock_mask])) / 0.4  # Over distance
        
        print(f"  Max caprock uplift: {max_caprock_uplift:.2f} mm")
        print(f"  Vertical pressure gradient: {pressure_gradient:.3f} MPa/m")
        print(f"  CO₂ containment: {'Effective' if jnp.max(P[caprock_mask]) < 0.5 else 'Check seal'}")
        print(f"  Mechanical integrity: {'SAFE' if max_caprock_uplift < 10 else 'MONITOR'}")
        
        print("="*60)
        
        return fig
    
    def plot_residual_spatial_distribution(self):
        """
        Enhanced residual visualization showing layer-specific errors
        """
        if not hasattr(self, 'last_residuals'):
            print("Computing residuals first...")
            self.compute_physics_metrics(method='finite_diff')
        
        residuals = self.last_residuals
        
        # Parent class visualization first
        fig = super().plot_residual_spatial_distribution()
        
        # Add layer-specific analysis
        print("\n" + "="*60)
        print("LAYER-SPECIFIC RESIDUAL ANALYSIS")
        print("="*60)
        
        # Identify errors by layer
        caprock_mask = residuals['y'] > 0.8
        transition_mask = (residuals['y'] > 0.6) & (residuals['y'] <= 0.8)
        reservoir_mask = residuals['y'] <= 0.6
        
        total_residual = jnp.sqrt(residuals['mechanics_x']**2 + 
                                residuals['mechanics_y']**2 + 
                                residuals['flow']**2)
        
        print(f"\nAverage residuals by layer:")
        print(f"  Caprock: {jnp.mean(total_residual[caprock_mask]):.2e}")
        print(f"  Transition: {jnp.mean(total_residual[transition_mask]):.2e}")
        print(f"  Reservoir: {jnp.mean(total_residual[reservoir_mask]):.2e}")
        
        # Check interface errors
        interface1_mask = jnp.abs(residuals['y'] - 0.8) < 0.05
        interface2_mask = jnp.abs(residuals['y'] - 0.6) < 0.05
        
        print(f"\nInterface residuals:")
        print(f"  Caprock-transition (y≈0.8): {jnp.mean(total_residual[interface1_mask]):.2e}")
        print(f"  Transition-reservoir (y≈0.6): {jnp.mean(total_residual[interface2_mask]):.2e}")
        
        if jnp.mean(total_residual[interface1_mask]) > 2 * jnp.mean(total_residual):
            print("  ⚠ High errors at caprock interface - consider interface refinement")
        
        print("="*60)
        
        return fig
    
    def visualize_formation(self):
        """Visualize the CO2 storage formation properties"""
        return BiotCoupled2D_Heterogeneous.visualize_heterogeneous_fields(self)
    
    def analyze_injection(self, injection_x=0.2):
        """Analyze CO₂ injection response"""
        return BiotCoupled2D_Heterogeneous.visualize_co2_injection_response(self, injection_x)
    
    def save_checkpoint(self, filepath):
        """
        Enhanced checkpoint saving with heterogeneous metadata
        """
        checkpoint = {
            'all_params': self.all_params,
            'config': self.config,
            'model_type': 'heterogeneous_co2_storage',
            'properties': {
                'k_contrast': 1e7,
                'caprock_depth': 0.8,
                'transition_depth': 0.6,
                'E_base': self.config.problem_init_kwargs['E_base'],
                'k_base': self.config.problem_init_kwargs['k_base']
            }
        }
        
        # Add metrics if available
        if hasattr(self, 'last_metrics'):
            checkpoint['final_metrics'] = self.last_metrics
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Save a companion JSON for easy inspection
        import json
        json_path = filepath.replace('.pkl', '_info.json')
        json_data = {
            'model_type': 'heterogeneous_co2_storage',
            'k_contrast': '10^7',
            'layers': ['caprock', 'transition', 'reservoir'],
            'training_steps': self.config.n_steps
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Model saved to {filepath}")
        print(f"Metadata saved to {json_path}")
    
    