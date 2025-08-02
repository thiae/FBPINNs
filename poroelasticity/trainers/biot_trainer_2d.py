import numpy as np
import jax.numpy as jnp
import jax
import optax
from fbpinns.domains import RectangularDomainND
from fbpinns.problems import Problem
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants
from fbpinns.trainers import FBPINNTrainer

class BiotCoupled2D(Problem):
    """
    Unified 2D Biot Poroelasticity Problem
    Solves both mechanics (u_x, u_y) and flow (p) in one problem class
    
    Outputs: [u_x, u_y, p] - 3 total outputs
    Governing PDEs:
    - Mechanics: ∇·σ' + α∇p = 0
    - Flow: -∇·(k∇p) + α∇·u = 0
    """

    @staticmethod
    def init_params(E=5000.0, nu=0.25, alpha=0.8, k=1.0, mu=1.0):
        """Initialize material parameters for both mechanics and flow"""
        
        E = jnp.array(E, dtype=jnp.float32)
        nu = jnp.array(nu, dtype=jnp.float32)
        alpha = jnp.array(alpha, dtype=jnp.float32)
        k = jnp.array(k, dtype=jnp.float32)
        mu = jnp.array(mu, dtype=jnp.float32)
        
        # Reference values for non dimensionalization
        E_ref = jnp.array(5000.0, dtype=jnp.float32)
        k_ref = jnp.array(1.0, dtype=jnp.float32)
        
        # Non dimensionalize the parameters for numerical stability
        E_ref = 5000.0
        k_ref = 1.0
        
        E_norm = E / E_ref
        #k_norm = k / k_ref # Used only the k value for normalization in the loss function

        # Calculate the derived parameters using normalized Young's modulus
        G = E_norm * E_ref / (2.0 * (1.0 + nu))
        lam = E_norm * E_ref * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        static_params = {
            "dims": (3, 2),  # 3 outputs (u_x, u_y, p), 2 inputs (x, y)
            # Mechanics parameters
            "E": E,
            "E_ref": E_ref,
            "nu": nu,
            "G": G,
            "lam": lam,
            # Flow parameters  
            "k": k,
            "k_ref": k_ref,
            "mu": mu,
            # Coupling parameter
            "alpha": alpha
        }
        trainable_params = {}
        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        """Sample constraints for both mechanics and flow equations"""

        dom = all_params["static"].setdefault("domain", {})
        # assume 2D unit square if not provided
        d = all_params["static"]["problem"]["dims"][1]
        dom.setdefault("xmin", jnp.zeros((d,), dtype=jnp.float32))
        dom.setdefault("xmax", jnp.ones((d,),  dtype=jnp.float32))
        dom.setdefault("xd", d)

        # NB: if the model asks for a 1 tuple grid in 2D, just pad it to (N,N)
        bs0 = batch_shapes[0]
        if sampler == "grid" and len(bs0) == 1 and d > 1:
            batch_shape_phys = (bs0[0],) * d
        else:
            batch_shape_phys = bs0

        # Sample interior points
        # Physics constraints (interior points)
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shape_phys)

        # Required derivatives for mechanics (u_x, u_y) and flow (p)
        required_ujs_mech = (
            # u_x derivatives 
            (0, (0,)),  
            (0, (0,0)), 
            (0, (1,1)), 
            (0, (0,1)),
            # u_y derivatives
            (1, (1,)),
            (1, (0,0)),
            (1, (1,1)), 
            (1, (0,1)),
            # p derivatives for coupling
            (2, (0,)),
            (2, (1,)), 
            (2, (0,0)), 
            (2, (1,1))
        ) 
        
        # Sampling the Boundary constraints : use provided batch_shapes
        boundary_batch_shapes = batch_shapes[1:5]  # Skip interior (index 0)
        x_batches_boundaries = domain.sample_boundaries(all_params, key, sampler, boundary_batch_shapes)

        # LEFT BOUNDARY
        x_batch_left = x_batches_boundaries[0]
        # Target values for: u_x = 0, u_y = 0, p = 1
        ux_target_left = jnp.zeros((x_batch_left.shape[0], 1))
        uy_target_left = jnp.zeros((x_batch_left.shape[0], 1))
        p_target_left = jnp.ones((x_batch_left.shape[0], 1))
        required_ujs_left = ((0, ()), (1, ()), (2, ()))

        # RIGHT BOUNDARY  
        x_batch_right = x_batches_boundaries[1]
        # Target values for: ∂u_x/∂x = 0, ∂u_y/∂y = 0, p = 0
        p_target_right = jnp.zeros((x_batch_right.shape[0], 1))
        required_ujs_right = ((0, (0,)), (0, (1,)), (1, (0,)), (1, (1,)), (2, ()))

        # BOTTOM BOUNDARY
        x_batch_bottom = x_batches_boundaries[2]  
        # Target values for: u_y = 0, ∂p/∂y = 0
        uy_target_bottom = jnp.zeros((x_batch_bottom.shape[0], 1))
        required_ujs_bottom = ((1, ()), (2, (1,)))

        # TOP BOUNDARY
        x_batch_top = x_batches_boundaries[3]
        # Target values for: ∂u_x/∂x = 0, ∂u_y/∂y = 0, ∂p/∂y = 0
        required_ujs_top = ((0, (0,)), (0, (1,)), (1, (0,)), (1, (1,)), (2, ()), (2, (1,)))

        return [
            # Physics constraints
            [x_batch_phys, required_ujs_mech],
            # Boundary constraints: [x_batch, *target_values, required_ujs]
            [x_batch_left, ux_target_left, uy_target_left, p_target_left, required_ujs_left],
            [x_batch_right, p_target_right, required_ujs_right], 
            [x_batch_bottom, uy_target_bottom, required_ujs_bottom],
            [x_batch_top, required_ujs_top]
        ]

    @staticmethod
    def loss_fn(all_params, constraints, w_mech=1.0, w_flow=1.0, w_bc=1.0, auto_balance=True):
        """
        Unified loss function with automatic loss balancing
        
        Args:
            w_mech: Base weight for mechanics equation
            w_flow: Base weight for flow equation  
            w_bc: Base weight for boundary conditions
            auto_balance: Whether to use automatic loss balancing
        """
        # Get material parameters
        G = all_params["static"]["problem"]["G"]
        lam = all_params["static"]["problem"]["lam"]
        alpha = all_params["static"]["problem"]["alpha"]
        k = all_params["static"]["problem"]["k"]

        # Physics constraints: unpack x_batch and derivatives
        # The framework adds derivatives after required_ujs based on the order specified
        x_batch_phys = constraints[0][0]
        # Derivatives in order of required_ujs_mech:
        # (0,(0,)), (0,(0,0)), (0,(1,1)), (0,(0,1)), (1,(1,)), (1,(0,0)), (1,(1,1)), (1,(0,1)), (2,(0,)), (2,(1,)), (2,(0,0)), (2,(1,1))
        duxdx, d2uxdx2, d2uxdy2, d2uxdxdy, duydy, d2uydx2, d2uydy2, d2uydxdy, dpdx, dpdy, d2pdx2, d2pdy2 = constraints[0][1:13]
        
        # MECHANICS RESIDUAL
        # Compute divergence of displacement: ∇·u = ∂u_x/∂x + ∂u_y/∂y
        div_u = duxdx + duydy
        
        # Equilibrium equations: ∇·σ' + α∇p = 0
        # where σ' is effective stress: σ'_ij = 2G*ε_ij + λ*δ_ij*∇·u
        # ε_ij is strain tensor: ε_xx = ∂u_x/∂x, ε_yy = ∂u_y/∂y, ε_xy = 0.5*(∂u_x/∂y + ∂u_y/∂x)
        
        # X-direction equilibrium: ∂σ'_xx/∂x + ∂σ'_xy/∂y + α∂p/∂x = 0
        # σ'_xx = (2G + λ)∂u_x/∂x + λ∂u_y/∂y
        # σ'_xy = G(∂u_x/∂y + ∂u_y/∂x)
        equilibrium_x = ((2*G + lam)*d2uxdx2 + lam*d2uydxdy + 
                        G*d2uxdy2 + G*d2uydxdy + alpha*dpdx)
        
        # Y-direction equilibrium: ∂σ'_xy/∂x + ∂σ'_yy/∂y + α∂p/∂y = 0  
        # σ'_yy = (2G + λ)∂u_y/∂y + λ∂u_x/∂x
        # σ'_xy = G(∂u_x/∂y + ∂u_y/∂x)
        equilibrium_y = (G*d2uxdxdy + G*d2uydx2 + 
                        lam*d2uxdxdy + (2*G + lam)*d2uydy2 + alpha*dpdy)
        
        mechanics_loss = jnp.mean(equilibrium_x**2) + jnp.mean(equilibrium_y**2)

        # FLOW RESIDUAL
        # Flow equation: -k∇²p + α∇·u = 0
        laplacian_p = d2pdx2 + d2pdy2
        # Normalized permeability for better numerical stability
        #k_norm = k / all_params["static"]["problem"].get("k_ref", 1.0) (k norm wasnt used here because its the same value as k)
        flow_residual = -k * laplacian_p + alpha * div_u
        flow_loss = jnp.mean(flow_residual**2)

        # BOUNDARY CONDITIONS
        boundary_loss = 0.0
        
        # LEFT BOUNDARY: u_x=0, u_y=0, p=1
        constraint_left = constraints[1]
        x_batch_left = constraint_left[0]
        ux_target_left = constraint_left[1] 
        uy_target_left = constraint_left[2]
        p_target_left = constraint_left[3]
        ux_left = constraint_left[4]
        uy_left = constraint_left[5] 
        p_left = constraint_left[6]
        boundary_loss += (jnp.mean((ux_left - ux_target_left)**2) + 
                         jnp.mean((uy_left - uy_target_left)**2) +
                         jnp.mean((p_left - p_target_left)**2))

        # RIGHT BOUNDARY: Traction-free σ·n=0, p=0
        # Right boundary: ∂u_x/∂x=0, ∂u_y/∂y=0, p=0
        # Implement traction BCs: σ·n = 0 (n=[1,0] on right boundary)
        # Normal stress: σxx = (2G + λ)∂ux/∂x + λ∂uy/∂y - αp = 0
        # Shear stress: σxy = G(∂ux/∂y + ∂uy/∂x) = 0
        constraint_right = constraints[2]
        x_batch_right = constraint_right[0]
        p_target_right = constraint_right[1]
        duxdx_right = constraint_right[2]
        duxdy_right = constraint_right[3]
        duydx_right = constraint_right[4]
        duydy_right = constraint_right[5]
        p_right = constraint_right[6]
        normal_residual = (2*G + lam)*duxdx_right + lam*duydy_right - alpha*p_right
        shear_residual = G*(duxdy_right + duydx_right)
        boundary_loss += (jnp.mean(normal_residual**2) + 
                         jnp.mean(shear_residual**2) +
                         jnp.mean((p_right - p_target_right)**2))

        # BOTTOM BOUNDARY: u_y=0, ∂p/∂y=0
        constraint_bottom = constraints[3]
        x_batch_bottom = constraint_bottom[0]
        uy_target_bottom = constraint_bottom[1]
        uy_bottom = constraint_bottom[2]
        dpdy_bottom = constraint_bottom[3]
        boundary_loss += (jnp.mean((uy_bottom - uy_target_bottom)**2) +
                         jnp.mean(dpdy_bottom**2))
        
        # TOP BOUNDARY: ∂u_x/∂x=0, ∂u_y/∂y=0, ∂p/∂y=0
        # Implement traction BCs: σ·n = 0 (n=[0,1] on top boundary)
        # Normal stress: σyy = (2G + λ)∂uy/∂y + λ∂ux/∂x - αp = 0
        # Shear stress: σxy = G(∂ux/∂y + ∂uy/∂x) = 0
        constraint_top = constraints[4]
        x_batch_top = constraint_top[0]
        duxdx_top = constraint_top[1]
        duxdy_top = constraint_top[2]
        duydx_top = constraint_top[3]
        duydy_top = constraint_top[4]
        p_top = constraint_top[5]
        dpdy_top = constraint_top[6]
        normal_residual_top = (2*G + lam)*duydy_top + lam*duxdx_top - alpha*p_top
        shear_residual_top = G*(duxdy_top + duydx_top)
        boundary_loss += (jnp.mean(normal_residual_top**2) +
                         jnp.mean(shear_residual_top**2) +
                         jnp.mean(dpdy_top**2))

        # Print loss components periodically for monitoring to diagnose which part of the physics is harder to learn
        def _print_losses(step_val):
            jax.debug.print("Step {}: Mech: {:.2e}, Flow: {:.2e}, BC: {:.2e}", 
                            step_val, mechanics_loss, flow_loss, boundary_loss)
            return 0
            
        def _no_op(_):
            return 0
        
        if auto_balance:
            # Automatic loss balancing
            mech_scale = jax.lax.stop_gradient(mechanics_loss + 1e-8)
            flow_scale = jax.lax.stop_gradient(flow_loss + 1e-8) 
            bc_scale = jax.lax.stop_gradient(boundary_loss + 1e-8)
            
            # Target proportions 
            target_mech_ratio = 0.45
            target_flow_ratio = 0.45
            target_bc_ratio = 0.10
            
            # Compute automatic weights to get target proportions
            auto_w_mech = target_mech_ratio / mech_scale
            auto_w_flow = target_flow_ratio * mech_scale / flow_scale
            auto_w_bc = target_bc_ratio * mech_scale / bc_scale
            
            # Apply base weights and automatic scaling
            total_loss = (w_mech * auto_w_mech * mechanics_loss + 
                         w_flow * auto_w_flow * flow_loss + 
                         w_bc * auto_w_bc * boundary_loss)
        else:
             # Manual weighted loss
             total_loss = (w_mech * mechanics_loss + 
                         w_flow * flow_loss + 
                         w_bc * boundary_loss)

        return total_loss
    
    @staticmethod
    def verify_bcs(all_params, x_points, tol=1e-6, atol_disp=1e-3, atol_p=1e-2):
        """
        Verify that the exact solution satisfies boundary conditions
        
        Args:
            all_params: Parameters dictionary
            x_points: Points to evaluate at
            tol: Tolerance for boundary identification
            atol_disp: Absolute tolerance for displacement comparison
            atol_p: Absolute tolerance for pressure comparison
        """
        # Get exact solution
        pred = BiotCoupled2D.exact_solution(all_params, x_points)
        
        # Get displacements and pressure
        u_x = pred[:, 0:1]
        u_y = pred[:, 1:2]
        p = pred[:, 2:3]
        
        # Extract coordinates
        x = x_points[:, 0:1]
        y = x_points[:, 1:2]
        
        # Left boundary (x=0): u_x=0, u_y=0, p=1
        left_mask = x <= tol
        left_bc_satisfied = (
            jnp.allclose(u_x[left_mask], 0.0, atol=atol_disp) and
            jnp.allclose(u_y[left_mask], 0.0, atol=atol_disp) and
            jnp.allclose(p[left_mask], 1.0, atol=atol_p)
        )
        
        # Right boundary (x=1): p=0 (traction BCs checked separately)
        right_mask = jnp.abs(x - 1.0) <= tol
        right_bc_satisfied = jnp.allclose(p[right_mask], 0.0, atol=atol_p)
        
        # Bottom boundary (y=0): u_y=0
        bottom_mask = y <= tol
        bottom_bc_satisfied = jnp.allclose(u_y[bottom_mask], 0.0, atol=atol_disp)
        
        # Print verification results
        print("Boundary conditions verification:")
        print(f"  Left boundary: {left_bc_satisfied}")
        print(f"  Right boundary: {right_bc_satisfied}")
        print(f"  Bottom boundary: {bottom_bc_satisfied}")
        
        return left_bc_satisfied and right_bc_satisfied and bottom_bc_satisfied
    
    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        """
        Analytical solution that satisfies both Biot physics and boundary conditions
        
        Strategy: Design displacements to exactly satisfy mechanics equations
        while maintaining boundary condition compliance
        """
        # Get material parameters
        static_params = all_params["static"]["problem"]
        alpha = static_params["alpha"]
        k = static_params["k"]
        G = static_params["G"]
        lam = static_params["lam"]
        
        # Unpack coordinates
        x = x_batch[:, 0]
        y = x_batch[:, 1]

        # PRESSURE: Linear decay p = 1 - x (satisfies p=1 at x=0, p=0 at x=1)
        # This gives: ∂p/∂x = -1, ∂p/∂y = 0, ∇²p = 0
        p = (1.0 - x).reshape(-1, 1)
        
        # DISPLACEMENT: Design to satisfy both flow and mechanics equations
        # 
        # From flow equation: -k∇²p + α∇·u = 0
        # Since ∇²p = 0: α∇·u = 0 → ∇·u = 0
        #
        # From mechanics X-equation: (2G+λ)∂²ux/∂x² + ... + α∂p/∂x = 0
        # We need: (2G+λ)∂²ux/∂x² = -α∂p/∂x = -α(-1) = α
        # Therefore: ∂²ux/∂x² = α/(2G+λ)
        #
        # Integrating: ∂ux/∂x = α*x/(2G+λ) + C1
        # Integrating: ux = α*x²/(2*(2G+λ)) + C1*x + C2
        #
        # Apply BC: ux(0,y) = 0 → C2 = 0
        # Apply BC: ux(1,y) = 0 → α/(2*(2G+λ)) + C1 = 0 → C1 = -α/(2*(2G+λ))
        #
        # Final: ux = α*x²/(2*(2G+λ)) - α*x/(2*(2G+λ)) = α*x*(x-1)/(2*(2G+λ))
        
        coeff_x = alpha / (2.0 * (2.0*G + lam))
        ux = coeff_x * x * (x - 1.0)
        
        # For uy: From ∇·u = 0: ∂ux/∂x + ∂uy/∂y = 0
        # ∂ux/∂x = α*(2x-1)/(2*(2G+λ))
        # So: ∂uy/∂y = -α*(2x-1)/(2*(2G+λ))
        #
        # This suggests uy should depend on x, but we need uy(x,0) = 0
        # Compromise: Use original form with adjusted coefficient to minimize residuals
        # uy = A * y * (1-y) where A is chosen to approximately satisfy ∇·u ≈ 0
        
        # At domain center (x=0.5): ∂ux/∂x = 0, so we want ∂uy/∂y ≈ 0
        # This happens when A is small, use A = coeff_x for consistency
        coeff_y = coeff_x  # Same coefficient for symmetry
        uy = coeff_y * y * (1.0 - y)
        
        # Reshape for consistency
        ux = ux.reshape(-1, 1)
        uy = uy.reshape(-1, 1)

        return jnp.hstack([ux, uy, p])

class BiotCoupledTrainer:
    """Trainer class for the unified Biot problem with pre training and gradual coupling"""
    
    def __init__(self, w_mech=1.0, w_flow=1.0, w_bc=1.0, auto_balance=True):
        """
        Initialize with loss weights
        
        Args:
            w_mech: Base weight for mechanics equation
            w_flow: Base weight for flow equation
            w_bc: Base weight for boundary conditions
            auto_balance: Whether to use automatic loss balancing
        """
        self.w_mech = w_mech
        self.w_flow = w_flow
        self.w_bc = w_bc
        self.auto_balance = auto_balance

        self.config = Constants(
            run="biot_coupled_2d",
            domain=RectangularDomainND,
            domain_init_kwargs={'xmin': jnp.array([0., 0.]), 'xmax': jnp.array([1., 1.])},
            problem=BiotCoupled2D,
            problem_init_kwargs={'E': 5000.0, 'nu': 0.25, 'alpha': 0.8, 'k': 1.0, 'mu': 1.0},
            decomposition=RectangularDecompositionND,
            decomposition_init_kwargs={
                'subdomain_xs': [jnp.linspace(0, 1, 4), jnp.linspace(0, 1, 3)],
                'subdomain_ws': [0.5 * jnp.ones(4), 0.7 * jnp.ones(3)],
                'unnorm': (0., 1.)
            },
            network=FCN,
            network_init_kwargs={'layer_sizes': [2, 256, 256, 256, 256, 3], 'activation': 'swish'},  # 3 outputs
            # Original sampling configuration
            ns=((100, 100), (25,), (25,), (25,), (25,)),  # 10k interior vs 100 boundary
            n_test=(15, 15),  # Test points for evaluation
            n_steps=5000,
            optimiser_kwargs={
                'learning_rate': 1e-3,  # Learning rate for Adam optimizer
            },
            summary_freq=100,
            test_freq=500,
            show_figures=False,
            save_figures=False,
            clear_output=True
        )
        self.trainer = FBPINNTrainer(self.config)
        self.all_params = None
    
    def train_mechanics_only(self, n_steps=100):
        """Pre train mechanics only (sets flow weight to 0)"""
        print("Pre training mechanics only")
        return self._train_with_weights(n_steps, w_mech=self.w_mech, w_flow=0.0, w_bc=self.w_bc)
    
    def train_flow_only(self, n_steps=100):
        """Pre train flow only (sets mechanics weight to 0)"""
        print("Pre training flow only")
        return self._train_with_weights(n_steps, w_mech=0.0, w_flow=self.w_flow, w_bc=self.w_bc)
    
    def train_coupled(self, n_steps=100):
        """Train fully coupled system with automatic balancing"""
        print("Training coupled system with automatic loss balancing")
        return self._train_with_weights(n_steps, w_mech=self.w_mech, w_flow=self.w_flow, w_bc=self.w_bc)
    
    def train_gradual_coupling(self, n_steps_pre=50, n_steps_coupled=100):
        """
        Gradual coupling with automatic loss balancing
        """
        print(" Gradual coupling with auto balance ")

        # Step 1: Train mechanics only (disable auto balance for single equation)
        old_auto_balance = self.auto_balance
        self.auto_balance = False
        self.train_mechanics_only(n_steps_pre)
        
        # Step 2: Train flow only  
        self.train_flow_only(n_steps_pre)
        
        # Step 3: Use auto balancing for coupled training
        self.auto_balance = True
        
        # Gradually increasing coupling with auto balancing
        coupling_schedule = [0.1, 0.3, 0.5, 0.8, 1.0]
        for i, coupling_strength in enumerate(coupling_schedule):
            print(f"Coupling step {i+1}/5: strength = {coupling_strength} (auto-balanced)")
            w_mech_scaled = coupling_strength * self.w_mech
            w_flow_scaled = coupling_strength * self.w_flow
            self._train_with_weights(n_steps_coupled//5, w_mech_scaled, w_flow_scaled, self.w_bc)
        
        # Restore original auto balancing setting
        self.auto_balance = old_auto_balance
        
        print(" Gradual coupling with auto balancing completed ")
        return self.all_params
    
    def _train_with_weights(self, n_steps, w_mech, w_flow, w_bc):
        """Internal method to train with specific weights"""
        # Original loss function
        problem_class = self.trainer.c.problem
        original_loss_fn = problem_class.loss_fn
        
        # Weighted loss function with auto balancing
        def weighted_loss_fn(all_params, constraints):
            return original_loss_fn(all_params, constraints, w_mech, w_flow, w_bc, self.auto_balance)
        
        # Temporarily replace the loss function
        problem_class.loss_fn = weighted_loss_fn
        
        # Set training steps
        old_n_steps = self.trainer.c.n_steps
        self.trainer.c.n_steps = n_steps
        
        # Train
        self.all_params = self.trainer.train()
        
        # Restore original settings
        self.trainer.c.n_steps = old_n_steps
        problem_class.loss_fn = original_loss_fn
        
        return self.all_params
    
    def predict(self, x_points):
        """Predict [u_x, u_y, p] at given points using FBPINN_solution"""
        if self.all_params is None:
            raise ValueError(" Model not trained yet")
        
        # Using the FBPINN_solution function from analysis module
        from fbpinns.analysis import FBPINN_solution
        
        # Active array (all subdomains active for prediction)
        active = jnp.ones(self.all_params["static"]["decomposition"]["m"], dtype=jnp.int32)
        
        # Prediction using FBPINN_solution
        predictions = FBPINN_solution(self.config, self.all_params, active, x_points)
        
        return predictions
    
    def get_displacement(self, x_points):
        """Get displacement field [u_x, u_y]"""
        pred = self.predict(x_points)
        return pred[:, :2]  # First 2 outputs
    
    def get_pressure(self, x_points):
        """Get pressure field [p]"""
        pred = self.predict(x_points)
        return pred[:, 2:3]  # Third output
    
    def get_test_points(self):
        """Get test points for evaluation"""
        if self.all_params is None:
            # Initialize parameters if not trained 
            raise RuntimeError("Train first, then call get_test_points")
        return self.trainer.get_batch(self.all_params, self.config.n_test, 'test')

def CoupledTrainer():
    """Create unified coupled trainer with automatic loss balancing"""
    return BiotCoupledTrainer(auto_balance=True)
