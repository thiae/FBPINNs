import jax.numpy as jnp
import matplotlib.pyplot as plt

from poroelasticity.trainers.base_model import BiotCoupled2D, BiotCoupledTrainer
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
        params = trainer.config.problem_init_kwargs
        params.update({
            "caprock_depth": 0.8,
            "transition_depth": 0.6,
            "E_base": 5000.0,
            "k_base": 1.0,
            "nu": 0.25
        })
        
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
    
    def visualize_formation(self):
        """Visualize the CO2 storage formation properties"""
        return BiotCoupled2D_Heterogeneous.visualize_heterogeneous_fields(self)
    
    def analyze_injection(self, injection_x=0.2):
        """Analyze CO₂ injection response"""
        return BiotCoupled2D_Heterogeneous.visualize_co2_injection_response(self, injection_x)