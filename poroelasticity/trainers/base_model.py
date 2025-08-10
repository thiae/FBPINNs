import numpy as np
import jax.numpy as jnp
import jax
import optax
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from fbpinns.domains import RectangularDomainND
from fbpinns.problems import Problem
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants
from fbpinns.trainers import FBPINNTrainer

class BiotCoupled2D(Problem):
    """
    Unified 2D Biot poroelasticity problem with transient flow and geomechanics.

    This class implements a homogeneous base model for coupled geomechanics and
    fluid flow in two spatial dimensions and time.  The network takes three
    inputs (x, y, t) and returns three outputs (u_x, u_y, p).  Hard Dirichlet
    boundary conditions are imposed for pressure on the left (injection) and
    right (far-field) boundaries and for displacement on the left and bottom
    boundaries.  Initial conditions are satisfied via a smooth ramp in time.

    The loss function enforces the quasi‑static momentum balance and the
    transient mass balance with a homogeneous permeability.  Storage effects
    enter via the Biot modulus M.
    """

    @staticmethod
    def init_params(E=5000.0, nu=0.25, alpha=0.8, k=1.0, mu=1.0, M=100.0):
        """Initialize material parameters for both mechanics and flow.

        Parameters
        ----------
        E : float
            Young's modulus (Pa).
        nu : float
            Poisson's ratio (dimensionless).
        alpha : float
            Biot coefficient (dimensionless).
        k : float
            Intrinsic permeability (mD or nondimensionalised).
        mu : float
            Fluid viscosity (Pa·s or nondimensionalised).
        M : float
            Biot modulus governing storage (Pa).  The specific storage is 1/M.

        Returns
        -------
        tuple
            A tuple of (static_params, trainable_params).  For this problem
            there are no trainable material parameters."""

        # Convert all parameters to JAX arrays for consistency and automatic
        # differentiation.  Using float32 avoids unnecessary promotions.
        E = jnp.array(E, dtype=jnp.float32)
        nu = jnp.array(nu, dtype=jnp.float32)
        alpha = jnp.array(alpha, dtype=jnp.float32)
        k = jnp.array(k, dtype=jnp.float32)
        mu = jnp.array(mu, dtype=jnp.float32)
        M = jnp.array(M, dtype=jnp.float32)

        # Elastic constants derived from E and nu
        G = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        static_params = {
            # dims = (number of outputs, number of inputs)
            # outputs: (u_x, u_y, p); inputs: (x, y, t)
            "dims": (3, 3),  
            "E": E,
            "nu": nu,
            "G": G,
            "lam": lam,
            "k": k,
            "mu": mu,
            "alpha": alpha,
            "M": M
        }
        trainable_params = {}
        return static_params, trainable_params

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        """
        Apply hard constraints to satisfy boundary and initial conditions.

        The raw network output u(x, y, t) = [u_x, u_y, p] is transformed into
        physically meaningful fields that exactly satisfy Dirichlet boundary
        conditions and initial conditions.  A time dependent envelope ensures
        that the solution vanishes at t=0.

        Hard constraints enforced:

        - u_x(0, y, t) = 0 (fixed left boundary)
        - u_y(x, 0, t) = 0 (fixed bottom boundary)
        - u(x, y, 0) = 0 (initially undeformed)
        - p(0, y, t) = p_L(y, t) (injection boundary, time ramped)
        - p(1, y, t) = 0 (far field boundary)
        - p(x, y, 0) = p0(y) (initial pressure)

        Parameters
        ----------
        all_params : dict
            Dictionary containing all model parameters.
        x_batch : array_like, shape (N, 3)
            Coordinates (x, y, t) at which the network is evaluated.
        u : array_like, shape (N, 3)
            Raw network output before applying boundary conditions.

        Returns
        -------
        array_like, shape (N, 3)
            Constrained network output satisfying boundary and initial
            conditions.
        """

        x = x_batch[:, 0:1]  
        y = x_batch[:, 1:2]  
        t = x_batch[:, 2:3]  

        ux_raw = u[:, 0:1]
        uy_raw = u[:, 1:2]
        p_raw = u[:, 2:3]

        # Smooth ramp function: S(0) = 0 and S(t) → 1 as t → 1.  This prevents
        # discontinuities at t=0 and enforces the initial condition u=0.
        def S(t):
          return 1.0 - jnp.exp(-5.0*jnp.clip(t, 0.0, 1.0))

        # Initial pressure profile p0(y).  For excess pressure formulation,
        # choose p0 = 0.  For hydrostatic baseline, set p0 = c0 * (1 - y).
        p0 = jnp.zeros_like(t)

        # Left boundary pressure p_L(y, t).  A Gaussian centred at injection_center
        # with width sigma, modulated by the ramp S(t).

        injection_center, sigma = 0.4, 0.08
        gauss_y = jnp.exp(-((y - injection_center)**2) / (2.0 * sigma**2))
        pL = p0 + S(t) * gauss_y
        # Right boundary pressure p_R(y, t) = 0 (far‑field)
        pR = p0 * 0.0  

        # Hard Dirichlet conditions for displacement.  Multiplying by x and y
        # enforces u_x(0, ·, ·) = 0 and u_y(·, 0, ·) = 0, and multiplying by
        # S(t) ensures u = 0 at t=0.
        ux = S(t) * x * ux_raw           
        uy = S(t) * x * y * uy_raw       

        # Pressure ansatz combining initial condition, boundary conditions,
        # and learnable interior variation.  The factor x(1-x) ensures that
        # the interior term vanishes on x=0 and x=1 so it cannot violate
        # Dirichlet conditions.  Multiplication by S(t) enforces p = p0 at t=0.
        p = (
            p0 
            + (1.0 - x) * (pL - p0) 
            + x * (pR - p0) 
            + x * (1.0 - x) * S(t) * p_raw
        )

        return jnp.concatenate([ux, uy, p], axis=-1)

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        """
        Sample interior and boundary points for the residual and specify
        required derivatives.

        This method is used by FBPINNs to generate collocation points and
        determine which derivatives of the network output are needed for the
        physics loss.  Interior points are sampled in (x, y, t).  Boundary
        points are still sampled but the boundary conditions are enforced
        exactly, so their derivatives are not used in the loss.
        """

        dom = all_params["static"].setdefault("domain", {})
        d = all_params["static"]["problem"]["dims"][1]
        dom.setdefault("xmin", jnp.zeros((d,), dtype=jnp.float32))
        dom.setdefault("xmax", jnp.ones((d,),  dtype=jnp.float32))
        dom.setdefault("xd", d)

        bs0 = batch_shapes[0]
        if sampler == "grid" and len(bs0) == 1 and d > 1:
            batch_shape_phys = (bs0[0],) * d
        else:
            batch_shape_phys = bs0

        # Sample interior collocation points in (x, y, t)
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shape_phys)

        # Derivatives needed for mechanics and flow equations.
        # Ordering here must match the unpacking in loss_fn.
        required_ujs_phys = (
            (0, (0,)),   # ∂ux/∂x
            (0, (0,0)),  # ∂²ux/∂x²
            (0, (1,1)),  # ∂²ux/∂y²
            (0, (0,1)),  # ∂²ux/(∂x∂y)
            (1, (1,)),   # ∂uy/∂y
            (1, (0,0)),  # ∂²uy/∂x²
            (1, (1,1)),  # ∂²uy/∂y²
            (1, (0,1)),  # ∂²uy/(∂x∂y)
            (2, (0,)),   # ∂p/∂x
            (2, (1,)),   # ∂p/∂y
            (2, (0,0)),  # ∂²p/∂x²
            (2, (1,1)),  # ∂²p/∂y²
            (2, (2,)),   # ∂p/∂t
            (0, (0,2)),  # ∂/∂t(∂ux/∂x)
            (1, (1,2)),  # ∂/∂t(∂uy/∂y)
        )

        # Sample boundary points.  Hard BCs mean we do not need derivatives
        # there, but sampling can be useful for diagnostics.
        boundary_batch_shapes = batch_shapes[1:5]
        try:
            x_batches_boundaries = domain.sample_boundaries(all_params, key, sampler, boundary_batch_shapes)
        except Exception:
            zeros = jnp.zeros((0, d), dtype=jnp.float32)
            x_batches_boundaries = [zeros, zeros, zeros, zeros]

        required_ujs_boundary = ((0, ()), (1, ()), (2, ()))

        return [
            [x_batch_phys, required_ujs_phys],
            [x_batches_boundaries[0], required_ujs_boundary],
            [x_batches_boundaries[1], required_ujs_boundary],
            [x_batches_boundaries[2], required_ujs_boundary],
            [x_batches_boundaries[3], required_ujs_boundary],
        ]

    @staticmethod
    def loss_fn(all_params, constraints):
    
        """
        Compute the mean squared residuals of the Biot equations.

        Mechanics residual:   ∇·σ = 0,  σ = 2G ε + λ tr(ε) I − α p I
        Flow residual:        (1/M) p_t + α ∂t(div u) − ∇·((k/μ) ∇p) = 0

        Homogeneous permeability (∇k = 0) is assumed in this base model.
        """

        prob = all_params["static"]["problem"]
        E, G, lam = prob["E"], prob["G"], prob["lam"]
        alpha, k, mu, M = prob["alpha"], prob["k"], prob["mu"], prob["M"]

        # Unpack derivatives according to the ordering in required_ujs_phys.
        (
        duxdx, d2uxdx2, d2uxdy2, d2uxdxdy,
        duydy, d2uydx2, d2uydy2, d2uydxdy,
        dpdx, dpdy, d2pdx2, d2pdy2,
        p_t, duxdx_t, duydy_t
        ) = constraints[0][1:16]

        # Divergence and its time derivative
        div_u   = duxdx + duydy
        div_u_t = duxdx_t + duydy_t
        lap_p   = d2pdx2 + d2pdy2

        # Momentum equilibrium residuals: ∇·σ=0 with σ = 2G ε + λ tr(ε) I − αp I  (keep the minus!)
        equilibrium_x = (
            (2 * G + lam) * d2uxdx2 
            + lam * d2uydxdy 
            + G * d2uxdy2 
            + G * d2uydxdy 
            - alpha * dpdx
            )
        equilibrium_y = (
            G * d2uxdxdy 
            + G * d2uydx2 
            + lam * d2uxdxdy 
            + (2 * G + lam) * d2uydy2 
            - alpha * dpdy)

       # Mass balance residual: (1/M) p_t + α ∂t(div u) − ∇·((k/μ)∇p) = 0  (homog. k => ∇k=0)
        darcy_div = (k / mu) * lap_p
        flow_residual = (p_t / M) + alpha * div_u_t - darcy_div

        # Characteristic scales to balance contributions
        L = 1.0
        p_scale = 1.0
        mech_scale = jnp.maximum(E / (L ** 2), 1e-6)
        flow_scale = jnp.maximum((k / mu) * (p_scale / (L ** 2)), 1e-8)

        mechanics_loss = jnp.mean((equilibrium_x / mech_scale) ** 2 + (equilibrium_y / mech_scale) ** 2)
        flow_loss = jnp.mean((flow_residual / flow_scale) ** 2)
        return mechanics_loss + flow_loss


    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        """
        There is no closed form solution for this problem; return None.
        """
        return None


class BiotCoupledTrainer:
    """
    Training wrapper for the BiotCoupled2D problem with hard boundary conditions.

    This class configures the domain, decomposition, neural network, sampling,
    and training hyperparameters.  It exposes convenience methods for training,
    prediction, boundary verification, and plotting.
    """

    def __init__(self):
        # Define configuration for FBPINN.  We select a rectangular domain in
        # (x, y, t) ∈ [0,1]×[0,1]×[0,1] with overlapping subdomains for FBPINN.

        self.config = Constants(
            run="biot_coupled_2d",
            domain=RectangularDomainND,
            domain_init_kwargs={
                'xmin': jnp.array([0.0, 0.0, 0.0]), 
                'xmax': jnp.array([1.0, 1.0, 1.0])
                },
            problem=BiotCoupled2D,
            problem_init_kwargs={
                'E': 5000.0, 
                'nu': 0.25, 
                'alpha': 0.8, 
                'k': 1.0, 
                'mu': 1.0, 
                'M': 100.0
                },
            decomposition=RectangularDecompositionND,
            decomposition_init_kwargs={
                # Subdomain centers along x, y, and t.  More subdomains in x,y than t
                'subdomain_xs': [
                    jnp.linspace(0, 1, 3), 
                    jnp.linspace(0, 1, 3), 
                    jnp.linspace(0, 1, 3)
                    ],
                # Subdomain widths.  Overlap is controlled by these values.
                'subdomain_ws': [
                    0.8 * jnp.ones(3),
                    0.8 * jnp.ones(3), 
                    0.8 * jnp.ones(2)
                    ],
                'unnorm': (0., 1.)
            },
            network=FCN,
            network_init_kwargs={
                # Architecture: 3 inputs (x,y,t), hidden layers, 3 outputs (u_x,u_y,p)
                'layer_sizes': [3, 64, 64, 64, 3], 
                'activation': 'tanh'
                },
            # Sampling shapes: interior in (x,y,t), no boundary sampling for BCs
            ns=((80, 80, 16), (0,), (0,), (0,), (0,)), 
            n_test=(20, 20, 5),
            n_steps=5000,  
            optimiser_kwargs={'learning_rate': 1e-3},
            summary_freq=100,
            test_freq=250,
            show_figures=False,
            save_figures=False,
            clear_output=True
        )
        self.trainer = FBPINNTrainer(self.config)
        self.all_params = None

    def train(self, n_steps=None):
        """
        Train the PINN.  If n_steps is provided, override the configuration.
        """
        if n_steps is not None:
            self.config.n_steps = n_steps
            self.trainer.c.n_steps = n_steps
        print("Training with hard boundary constraints.")
        self.all_params = self.trainer.train()
        return self.all_params

    def predict(self, x_points, t=1.0):
        """
        Evaluate the trained network at specified points.  If x_points has
        shape (N,2), append a column of constant time t.
        """
        if self.all_params is None:
            raise ValueError("Model not trained yet")
        x_points = jnp.array(x_points)
        if x_points.shape[1] == 2: 
            tcol = jnp.full((x_points.shape[0], 1), float(t))
            x_points = jnp.concatenate([x_points, tcol], axis=1)
        from fbpinns.analysis import FBPINN_solution
        active = jnp.ones(self.all_params["static"]["decomposition"]["m"], dtype=jnp.int32)
        return FBPINN_solution(self.config, self.all_params, active, x_points)

    def verify_bcs(self, n_points=100, t=1.0):

        """
        Evaluate the maximum violation of hard boundary conditions at time t.
        """
  
        print("\n" + "="*60)
        print(f"Boundary Conditon Verification at t={t}")
        print("="*60)

        y_test = jnp.linspace(0, 1, n_points)
        tcol = jnp.full((n_points, 1), float(t))

        left_points = jnp.column_stack([jnp.zeros(n_points), y_test, tcol.squeeze()])
        right_points = jnp.column_stack([jnp.ones(n_points), y_test, tcol.squeeze()])
        bottom_points = jnp.column_stack([jnp.linspace(0, 1, n_points), jnp.zeros(n_points), tcol.squeeze()])

        left_pred = self.predict(left_points)
        right_pred = self.predict(right_points)
        bottom_pred = self.predict(bottom_points)

        # Reconstruct the target left boundary pressure used in constraining_fn
        injection_center, sigma = 0.4, 0.08
        S = 1.0 - jnp.exp(-5.0*float(t))
        pL = S * jnp.exp(-((y_test - injection_center) ** 2) / (2 * sigma ** 2))

        print(f"Left boundary (x=0): ux=0, uy=0, p=pL(y,t)")
        print(f"  max|ux| = {jnp.max(jnp.abs(left_pred[:, 0])):.2e}")
        print(f"  max|uy| = {jnp.max(jnp.abs(left_pred[:, 1])):.2e}")
        print(f"  max|p - pL| = {jnp.max(jnp.abs(left_pred[:, 2] - pL)):.2e}")
        print(f"Right boundary (x=1): p=0")
        print(f"  max|p| = {jnp.max(jnp.abs(right_pred[:, 2])):.2e}")
        print(f"Bottom boundary (y=0): uy=0")
        print(f"  max|uy| = {jnp.max(jnp.abs(bottom_pred[:, 1])):.2e}")

        all_violations = [
            jnp.max(jnp.abs(left_pred[:, 0])),
            jnp.max(jnp.abs(left_pred[:, 1])),
            jnp.max(jnp.abs(left_pred[:, 2] - pL)),
            jnp.max(jnp.abs(right_pred[:, 2])),
            jnp.max(jnp.abs(bottom_pred[:, 1])),
        ]
        max_violation = max(all_violations)
        if max_violation < 1e-6:
            print("\nStatus: PERFECT - All BCs satisfied to machine precision")
        elif max_violation < 1e-3:
            print("\nStatus: EXCELLENT - All BCs satisfied within tolerance")
        elif max_violation < 1e-2:
            print("\nStatus: GOOD - Minor BC violations")
        else:
            print(f"\nStatus: CHECK - Max violation: {max_violation:.3e}")
        return max_violation < 1e-2

    def compute_physics_metrics(self, n_points=50, method='autodiff'):
        """
        Unified physics validation metrics with option for different derivative methods.

        Args:
            n_points: Number of points along each dimension for evaluation grid
            method: Method for computing derivatives: 'autodiff' (more accurate) or
                   'finite_diff' (faster, better for larger grids)

        Returns:
            Dictionary containing all computed metrics including relative errors
        """
        if self.all_params is None:
            raise ValueError("Model not trained yet")

        print(f"Physics Validation Metrics (using {method})")

        # Create evaluation grid
        if method == 'finite_diff':
            # Avoid boundaries for finite difference method to reduce edge effects
            x = jnp.linspace(0.1, 0.9, n_points)
            y = jnp.linspace(0.1, 0.9, n_points)
        else:
            # Full domain for autodiff method
            x = jnp.linspace(0, 1, n_points)
            y = jnp.linspace(0, 1, n_points)

        X, Y = jnp.meshgrid(x, y)
        points = jnp.column_stack([X.flatten(), Y.flatten()])

        # Get material parameters
        G = self.all_params["static"]["problem"]["G"]
        lam = self.all_params["static"]["problem"]["lam"]
        alpha = self.all_params["static"]["problem"]["alpha"]
        k = self.all_params["static"]["problem"]["k"]
        E = self.all_params["static"]["problem"]["E"]

        # Predict at all points
        u_pred = self.predict(points)
        ux = u_pred[:, 0]
        uy = u_pred[:, 1]
        p = u_pred[:, 2]

        # Compute characteristic scales for normalization (used for relative errors)
        L = 1.0  # Domain size
        p_scale = 1.0  # Maximum pressure (BC value)
        u_scale = jnp.maximum(jnp.max(jnp.abs(ux)), jnp.max(jnp.abs(uy)))
        if u_scale < 1e-10:
            u_scale = p_scale * L / E  # Theoretical scale

        elastic_force_scale = E * u_scale / L**2
        pressure_gradient_scale = k * p_scale / L**2

        # Calculate derivatives based on selected method
        if method == 'autodiff':
            # Use JAX autodiff (more accurate but potentially slower)
            # Helper function to create derivative functions
            def create_derivative_fn(component, order):
                """Create a function that computes derivatives of a component"""
                if component == 'ux':
                    idx = 0
                elif component == 'uy':
                    idx = 1
                else:  # pressure
                    idx = 2

                def predict_component(xy):
                    return self.predict(xy.reshape(1, -1))[0, idx]

                if order == 'dx':
                    return jax.grad(predict_component, argnums=0)
                elif order == 'dy':
                    return jax.grad(predict_component, argnums=1)
                elif order == 'dx2':
                    return jax.grad(jax.grad(predict_component, argnums=0), argnums=0)
                elif order == 'dy2':
                    return jax.grad(jax.grad(predict_component, argnums=1), argnums=1)
                elif order == 'dxdy':
                    return jax.grad(jax.grad(predict_component, argnums=0), argnums=1)

            # Vectorize the derivative functions
            d_ux_dx_fn = jax.vmap(create_derivative_fn('ux', 'dx'))
            d_ux_dy_fn = jax.vmap(create_derivative_fn('ux', 'dy'))
            d2_ux_dx2_fn = jax.vmap(create_derivative_fn('ux', 'dx2'))
            d2_ux_dy2_fn = jax.vmap(create_derivative_fn('ux', 'dy2'))
            d2_ux_dxdy_fn = jax.vmap(create_derivative_fn('ux', 'dxdy'))

            d_uy_dx_fn = jax.vmap(create_derivative_fn('uy', 'dx'))
            d_uy_dy_fn = jax.vmap(create_derivative_fn('uy', 'dy'))
            d2_uy_dx2_fn = jax.vmap(create_derivative_fn('uy', 'dx2'))
            d2_uy_dy2_fn = jax.vmap(create_derivative_fn('uy', 'dy2'))
            d2_uy_dxdy_fn = jax.vmap(create_derivative_fn('uy', 'dxdy'))

            d_p_dx_fn = jax.vmap(create_derivative_fn('p', 'dx'))
            d_p_dy_fn = jax.vmap(create_derivative_fn('p', 'dy'))
            d2_p_dx2_fn = jax.vmap(create_derivative_fn('p', 'dx2'))
            d2_p_dy2_fn = jax.vmap(create_derivative_fn('p', 'dy2'))

            # Sample points for derivative evaluation (use subset for efficiency)
            sample_indices = jax.random.randint(
                jax.random.PRNGKey(0),
                shape=(min(1000, n_points*n_points),),
                minval=0,
                maxval=n_points*n_points
            )
            sample_points = points[sample_indices]

            # Compute derivatives
            print("Computing derivatives using JAX autodiff...")

            d_ux_dx = d_ux_dx_fn(sample_points)
            d_ux_dy = d_ux_dy_fn(sample_points)
            d2_ux_dx2 = d2_ux_dx2_fn(sample_points)
            d2_ux_dy2 = d2_ux_dy2_fn(sample_points)
            d2_ux_dxdy = d2_ux_dxdy_fn(sample_points)

            d_uy_dx = d_uy_dx_fn(sample_points)
            d_uy_dy = d_uy_dy_fn(sample_points)
            d2_uy_dx2 = d2_uy_dx2_fn(sample_points)
            d2_uy_dy2 = d2_uy_dy2_fn(sample_points)
            d2_uy_dxdy = d2_uy_dxdy_fn(sample_points)

            d_p_dx = d_p_dx_fn(sample_points)
            d_p_dy = d_p_dy_fn(sample_points)
            d2_p_dx2 = d2_p_dx2_fn(sample_points)
            d2_p_dy2 = d2_p_dy2_fn(sample_points)

            # Compute physics residuals
            # Divergence of displacement
            div_u = d_ux_dx + d_uy_dy

            # Mechanics equations
            equilibrium_x = ((2*G + lam)*d2_ux_dx2 + lam*d2_uy_dxdy +
                           G*d2_ux_dy2 + G*d2_uy_dxdy + alpha*d_p_dx)

            equilibrium_y = (G*d2_ux_dxdy + G*d2_uy_dx2 +
                           lam*d2_ux_dxdy + (2*G + lam)*d2_uy_dy2 + alpha*d_p_dy)

            # Flow equation
            laplacian_p = d2_p_dx2 + d2_p_dy2
            flow_residual = -k * laplacian_p + alpha * div_u

        else:  # 'finite_diff'
            # Use finite differences (faster but less accurate)
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

            # Compute physics residuals
            div_u = dux_dx + duy_dy

            # Mechanics equations residuals
            equilibrium_x = ((2*G + lam)*d2ux_dx2 + lam*d2uy_dxdy +
                             G*d2ux_dy2 + G*d2uy_dxdy + alpha*dp_dx)

            equilibrium_y = (G*d2ux_dxdy + G*d2uy_dx2 +
                             lam*d2ux_dxdy + (2*G + lam)*d2uy_dy2 + alpha*dp_dy)

            # Flow equation residual
            laplacian_p = d2p_dx2 + d2p_dy2
            flow_residual = -k * laplacian_p + alpha * div_u

        # Compute absolute error metrics
        mechanics_x_rms = jnp.sqrt(jnp.mean(equilibrium_x**2))
        mechanics_y_rms = jnp.sqrt(jnp.mean(equilibrium_y**2))
        mechanics_rms = jnp.sqrt((mechanics_x_rms**2 + mechanics_y_rms**2) / 2)
        flow_rms = jnp.sqrt(jnp.mean(flow_residual**2))

        # Compute L-infinity norms
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
        # Normalize mechanics residual by elastic force scale
        mechanics_x_rel = mechanics_x_rms / elastic_force_scale
        mechanics_y_rel = mechanics_y_rms / elastic_force_scale
        mechanics_rel = mechanics_rms / elastic_force_scale

        # Normalize flow residual by pressure gradient scale
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

        # Print detailed results in a table
        print("\n┌─────────────────┬──────────────┬──────────────┬──────────────┐")
        print("  │ Equation        │ RMS Absolute │ RMS Relative │ L-inf Norm   │")
        print("  ├─────────────────┼──────────────┼──────────────┼──────────────┤")
        print(f" │ Mechanics (x)   │ {mechanics_x_rms:.2e} │ {mechanics_x_rel:.2e} │ {mechanics_x_linf:.2e} │")
        print(f" │ Mechanics (y)   │ {mechanics_y_rms:.2e} │ {mechanics_y_rel:.2e} │ {mechanics_y_linf:.2e} │")
        print(f" │ Mechanics (avg) │ {mechanics_rms:.2e} │ {mechanics_rel:.2e} │ {mechanics_linf:.2e} │")
        print(f" │ Flow            │ {flow_rms:.2e} │ {flow_rel:.2e} │ {flow_linf:.2e} │")
        print("  └─────────────────┴──────────────┴──────────────┴──────────────┘")

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

        # Store residuals for spatial distribution plotting if using finite differences
        # (this enables plotting with plot_residual_spatial_distribution method)
        if method == 'finite_diff':
            self.last_residuals = {
                'x': X,
                'y': Y,
                'mechanics_x': equilibrium_x.reshape(n_points, n_points),
                'mechanics_y': equilibrium_y.reshape(n_points, n_points),
                'flow': flow_residual.reshape(n_points, n_points),
                'div_u': div_u.reshape(n_points, n_points)
            }
        elif hasattr(self, 'last_residuals'):
            # If we're using autodiff but have previously stored residuals, let the user know
            print("\nNote: Residual spatial data is not updated when using autodiff method.")
            print("Previous spatial data will be used if plotting residual distribution.")

        print("="*60)

        # Return comprehensive metrics as dictionary
        return {
            # Absolute metrics
            "mechanics_x_rms": float(mechanics_x_rms),
            "mechanics_y_rms": float(mechanics_y_rms),
            "mechanics_rms": float(mechanics_rms),
            "flow_rms": float(flow_rms),
            "mechanics_x_l2": float(mechanics_x_l2) if method == 'autodiff' else None,
            "mechanics_y_l2": float(mechanics_y_l2) if method == 'autodiff' else None,
            "mechanics_l2": float(mechanics_l2) if method == 'autodiff' else None,
            "flow_l2": float(flow_l2) if method == 'autodiff' else None,
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
            "method": method
        }

    def check_bc_satisfaction(self):
        """Quick BC check for convergence history."""
        # Sample boundary points
        n = 20
        y_test = jnp.linspace(0, 1, n)
        x_test = jnp.linspace(0, 1, n)

        # Left boundary
        left_points = jnp.column_stack([jnp.zeros(n), y_test])
        left_pred = self.predict(left_points)

        # Right boundary
        right_points = jnp.column_stack([jnp.ones(n), y_test])
        right_pred = self.predict(right_points)

        # Bottom boundary
        bottom_points = jnp.column_stack([x_test, jnp.zeros(n)])
        bottom_pred = self.predict(bottom_points)

        violations = [
            jnp.max(jnp.abs(left_pred[:, 0])),  # ux at left
            jnp.max(jnp.abs(left_pred[:, 1])),  # uy at left
            jnp.max(jnp.abs(left_pred[:, 2] - 1.0)),  # p at left
            jnp.max(jnp.abs(right_pred[:, 2])),  # p at right
            jnp.max(jnp.abs(bottom_pred[:, 1]))  # uy at bottom
        ]

        return {'max_violation': float(max(violations)), 'all_violations': violations}

    def plot_solution(self, n_points=50, t=1.0, depth_style=True):
        x = jnp.linspace(0, 1, n_points); y = jnp.linspace(0, 1, n_points)
        X, Y = jnp.meshgrid(x, y)
        XYT = jnp.column_stack([X.flatten(), Y.flatten(), jnp.full((n_points*n_points,), float(t))])

        pred = self.predict(XYT)
        UX = pred[:,0].reshape(n_points, n_points)
        UY = pred[:,1].reshape(n_points, n_points)
        P  = pred[:,2].reshape(n_points, n_points)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        c1 = axes[0].contourf(X, Y, UX*1000.0, levels=20, cmap='RdBu')
        axes[0].set_title('Horizontal disp $u_x$ [mm]'); axes[0].set_xlabel('x'); axes[0].set_ylabel('y'); plt.colorbar(c1, ax=axes[0])

        c2 = axes[1].contourf(X, Y, UY*1000.0, levels=20, cmap='RdBu')
        axes[1].set_title('Uplift $u_y$ [mm] (positive up)'); axes[1].set_xlabel('x'); axes[1].set_ylabel('y'); plt.colorbar(c2, ax=axes[1])

        c3 = axes[2].contourf(X, Y, P, levels=20, cmap='plasma')
        axes[2].set_title('Pressure p [nondim or MPa]'); axes[2].set_xlabel('x'); axes[2].set_ylabel('y'); plt.colorbar(c3, ax=axes[2])

        if depth_style:
            for ax in axes: ax.invert_yaxis()
            axes[0].set_ylabel('Depth y (down)')
            axes[1].set_ylabel('Depth y (down)')
            axes[2].set_ylabel('Depth y (down)')

        plt.suptitle(f'Solution at t={t:.2f}', fontsize=12, fontweight='bold'); plt.tight_layout(); plt.show()
        return fig, axes


    def track_convergence_history(self, checkpoint_steps=[100, 500, 1000, 1500, 2000],
                                save_checkpoints=True, checkpoint_dir="checkpoints"):
        """
        Track how physics satisfaction improves during training.

        Args:
            checkpoint_steps: List of training steps at which to evaluate
            save_checkpoints: Whether to save model at each checkpoint
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary containing convergence history
        """
        print("\n" + "="*60)
        print("CONVERGENCE HISTORY TRACKING")
        print("="*60)

        history = {
            'steps': [],
            'loss': [],
            'mechanics_rms': [],
            'flow_rms': [],
            'mechanics_rel': [],
            'flow_rel': [],
            'total_rel': [],
            'bc_violation': []
        }

        # Create checkpoint directory if needed
        if save_checkpoints:
            os.makedirs(checkpoint_dir, exist_ok=True)

        previous_steps = 0

        for step in checkpoint_steps:
            # Train for the incremental steps
            steps_to_train = step - previous_steps
            if steps_to_train > 0:
                print(f"\nTraining to step {step}...")
                self.train(n_steps=steps_to_train)
                previous_steps = step

            # Compute metrics using finite differences for spatial plotting
            metrics = self.compute_physics_metrics(n_points=30, method='finite_diff')

            # Check BC violations
            bc_metrics = self.check_bc_satisfaction()

            # Store history
            history['steps'].append(step)
            history['mechanics_rms'].append(metrics['mechanics_rms'])
            history['flow_rms'].append(metrics['flow_rms'])
            history['mechanics_rel'].append(metrics['mechanics_rel'])
            history['flow_rel'].append(metrics['flow_rel'])
            history['total_rel'].append(metrics['total_rel'])
            history['bc_violation'].append(bc_metrics['max_violation'])

            # Get current loss (approximate from trainer)
            # Note: This is a simplified approach
            current_loss = metrics['mechanics_rms'] + metrics['flow_rms']
            history['loss'].append(current_loss)

            # Save checkpoint if requested
            if save_checkpoints:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{step}.pkl")
                self.save_checkpoint(checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")

            # Print progress
            print(f"\nStep {step} Summary:")
            print(f"  Relative Errors - Mechanics: {metrics['mechanics_rel']:.2e}, Flow: {metrics['flow_rel']:.2e}")
            print(f"  BC Violation: {bc_metrics['max_violation']:.2e}")
            print(f"  Status: {metrics['status']}")

        # Plot convergence history
        self.plot_convergence_history(history)

        print("\n" + "="*60)

        return history

    def plot_convergence_history(self, history):
        """Plot the convergence history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        steps = history['steps']

        # Absolute RMS errors
        axes[0, 0].semilogy(steps, history['mechanics_rms'], 'b-o', label='Mechanics')
        axes[0, 0].semilogy(steps, history['flow_rms'], 'r-s', label='Flow')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('RMS Residual')
        axes[0, 0].set_title('Absolute RMS Errors')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Relative errors
        axes[0, 1].semilogy(steps, history['mechanics_rel'], 'b-o', label='Mechanics')
        axes[0, 1].semilogy(steps, history['flow_rel'], 'r-s', label='Flow')
        axes[0, 1].semilogy(steps, history['total_rel'], 'g-^', label='Total')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Relative Error')
        axes[0, 1].set_title('Relative Errors')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # BC violations
        axes[1, 0].semilogy(steps, history['bc_violation'], 'k-d')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Max BC Violation')
        axes[1, 0].set_title('Boundary Condition Satisfaction')
        axes[1, 0].grid(True, alpha=0.3)

        # Combined progress
        axes[1, 1].semilogy(steps, history['total_rel'], 'g-', linewidth=2, label='Physics Error')
        axes[1, 1].semilogy(steps, history['bc_violation'], 'k--', linewidth=2, label='BC Violation')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].set_title('Overall Convergence')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Add horizontal lines for reference
        for ax in axes.flat:
            ax.axhline(y=1e-3, color='g', linestyle=':', alpha=0.5, label='Excellent')
            ax.axhline(y=1e-2, color='y', linestyle=':', alpha=0.5, label='Good')
            ax.axhline(y=1e-1, color='r', linestyle=':', alpha=0.5, label='Acceptable')

        plt.suptitle('Training Convergence History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return fig, axes

    def visualize_domain_decomposition(self, save_path=None):
        """
        Visualize the FBPINN domain decomposition with subdomain boundaries and overlaps.
        Can be called before or after training.

        Args:
          save_path: Optional path to save the figure 

        Returns:
          fig, axes: Matplotlib figure and axes objects
        """
        print("\n" + "="*60)
        print("FBPINN DOMAIN DECOMPOSITION")
        print("="*60)

        # Get subdomain parameters from config (available before training)
        subdomain_xs = self.config.decomposition_init_kwargs['subdomain_xs']
        subdomain_ws = self.config.decomposition_init_kwargs['subdomain_ws']

        # Calculate total number of subdomains
        m = len(subdomain_xs[0]) * len(subdomain_xs[1])

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left plot: Subdomain boundaries
        ax1 = axes[0]

        # Get x and y divisions
        x_divs = subdomain_xs[0]
        y_divs = subdomain_xs[1]
        wx = subdomain_ws[0][0]  # Assuming uniform weights
        wy = subdomain_ws[1][0]

        # Plot the main domain
        ax1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))

        # Plot subdomain centers and boundaries
        colors = plt.cm.Set3(np.linspace(0, 1, m))
        subdomain_idx = 0

        for i, x_center in enumerate(x_divs):
            for j, y_center in enumerate(y_divs):
                # Calculate subdomain boundaries with overlap
                x_min = max(0, float(x_center - wx/2))
                x_max = min(1, float(x_center + wx/2))
                y_min = max(0, float(y_center - wy/2))
                y_max = min(1, float(y_center + wy/2))

                # Plot subdomain
                rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                    fill=True, facecolor=colors[subdomain_idx],
                                    alpha=0.3, edgecolor=colors[subdomain_idx],
                                    linewidth=1.5)
                ax1.add_patch(rect)

                # Add subdomain number
                ax1.text(float(x_center), float(y_center), f'{subdomain_idx+1}',
                        ha='center', va='center', fontsize=12, fontweight='bold')

                # Mark center point
                ax1.plot(float(x_center), float(y_center), 'ko', markersize=6)

                subdomain_idx += 1

        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'FBPINN Decomposition ({m} subdomains)')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # Right plot: Overlap visualization
        ax2 = axes[1]

        # Create overlap intensity map
        overlap_map = np.zeros((100, 100))
        x_grid = np.linspace(0, 1, 100)
        y_grid = np.linspace(0, 1, 100)

        for i, x_center in enumerate(x_divs):
            for j, y_center in enumerate(y_divs):
                x_min = max(0, float(x_center - wx/2))
                x_max = min(1, float(x_center + wx/2))
                y_min = max(0, float(y_center - wy/2))
                y_max = min(1, float(y_center + wy/2))

                # Find grid points in this subdomain
                x_mask = (x_grid >= x_min) & (x_grid <= x_max)
                y_mask = (y_grid >= y_min) & (y_grid <= y_max)

                for xi in np.where(x_mask)[0]:
                    for yi in np.where(y_mask)[0]:
                        overlap_map[yi, xi] += 1

        im = ax2.imshow(overlap_map, extent=[0, 1, 0, 1], origin='lower',
                        cmap='YlOrRd', aspect='equal')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Subdomain Overlap (Number of overlapping subdomains)')
        plt.colorbar(im, ax=ax2, label='Number of subdomains')

        # Add grid lines at subdomain boundaries
        for x in x_divs:
            ax2.axvline(float(x), color='black', linestyle='--', alpha=0.3)
        for y in y_divs:
            ax2.axhline(float(y), color='black', linestyle='--', alpha=0.3)

        # Print decomposition info
        print(f"Total subdomains: {m}")
        print(f"X-direction splits: {len(x_divs)} at positions: {[float(x) for x in x_divs]}")
        print(f"Y-direction splits: {len(y_divs)} at positions: {[float(y) for y in y_divs]}")
        print(f"Subdomain width (x): {float(wx):.2f}")
        print(f"Subdomain width (y): {float(wy):.2f}")
        print(f"Overlap parameter: {float(wx - 1/len(x_divs)):.2f} in x, {float(wy - 1/len(y_divs)):.2f} in y")

        # Calculate overlap statistics
        unique_overlaps = np.unique(overlap_map)
        print(f"\nOverlap statistics:")
        for overlap_count in unique_overlaps:
            coverage = np.sum(overlap_map == overlap_count) / overlap_map.size * 100
            print(f"  {int(overlap_count)} subdomain(s): {coverage:.1f}% of domain")

        plt.tight_layout()

        # Save the figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nDomain decomposition figure saved to: {save_path}")

        plt.show()

        print("="*60)

        return fig, axes

    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'all_params': self.all_params,
            'config': self.config
        }
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Checkpoint saved to {filepath}")

#  convenient functions
def FixedTrainer():
    """Create a trainer with CORRECT hard BC enforcement"""
    return BiotCoupledTrainer()

def CoupledTrainer():
    """Compatibility alias"""
    return BiotCoupledTrainer()

def ResearchTrainer():
    """Compatibility alias"""
    return BiotCoupledTrainer()