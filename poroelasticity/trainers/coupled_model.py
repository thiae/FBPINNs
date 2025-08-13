"""
Biot‐Coupled poroelasticity with spatially heterogeneous permeability.

This module defines a JAX/FBPINN implementation of the 2D transient Biot
equations where the permeability may vary in space.  It reuses
most of the structure from the homogeneous base model (see
`base_model.py`) but modifies the flow residual to account
for a spatially varying permeability field `k(x, y)`.  Users can supply
their own permeability function when constructing the trainer; by
default a simple layered distribution is used to emulate a caprock
/ reservoir system.

Key features:

* The network takes three inputs `(x, y, t)` and outputs `(u_x, u_y, p)`.
* Hard Dirichlet boundary conditions enforce displacement fixation on
  the left and bottom boundaries and prescribe pressure on the left and
  right boundaries.  Initial conditions are satisfied by a smooth
  temporal ramp.
* The flow residual includes both `k * laplacian(p)` and the term
  `∇k·∇p` when the permeability varies smoothly.
* Physics metrics optionally compute the divergence of the Darcy flux
  `q = (k/μ) ∇p` rather than assuming constant permeability.

Note that this model remains nondimensional; users interested in
physical predictions should introduce characteristic scales as discussed
elsewhere in the report.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import pickle
import os
import matplotlib.pyplot as plt
from fbpinns.domains import RectangularDomainND
from fbpinns.problems import Problem
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants
from fbpinns.trainers import FBPINNTrainer, get_inputs, FBPINN_model_jit
from poroelasticity.trainers.base_model import BiotCoupledTrainer as BaseBiotTrainer


class BiotCoupled2D_Heterogeneous(Problem):
    """
    2D Biot poroelasticity with transient flow and spatially varying
    permeability.

    This problem class closely follows the homogeneous version but
    introduces a permeability function `k_fun(x, y)` that may vary
    spatially.  If `k_fun` is not provided, a constant permeability is
    assumed.  Users can also provide a spatially varying Young's
    modulus `E_fun(x, y)` if desired; otherwise the mechanics remain
    homogeneous.
    """

    @staticmethod
    def default_k_fun(x, y):
        """Default permeability profile: a high‐perm reservoir overlain by a low‐perm caprock.

        The domain is `y∈[0,1]` with the caprock occupying the top
        portion (y < 0.3) and the reservoir below.  A hyperbolic
        tangent is used to smoothly transition between the two zones.

        Parameters
        ----------
        x, y : array_like
            Spatial coordinates.  Only `y` is used in this default
            implementation.

        Returns
        -------
        ndarray
            Permeability values at the given coordinates.
        """
        k_reservoir = 10.0  # high permeability in the reservoir (nondim)
        k_caprock = 0.1     # low permeability in the caprock (nondim)
        interface_y = 0.3
        sharpness = 40.0
        # weight ~1 in caprock, ~0 in reservoir
        w = 0.5 * (1.0 + jnp.tanh(sharpness * (interface_y - y)))
        return k_caprock + (k_reservoir - k_caprock) * (1.0 - w)

    @staticmethod
    def default_E_fun(x, y):
        """Default Young's modulus profile: uniform (homogeneous).

        Users can override this method to supply a spatially varying
        stiffness profile.  The returned values should be positive and
        represent nondimensional moduli when working in nondimensional
        units.
        """
        return jnp.ones_like(x)

    @staticmethod
    def init_params(E=5000.0, nu=0.25, alpha=0.8, k=1.0, mu=1.0, M=100.0,
                    k_fun=None, E_fun=None):
        """Initialize material parameters and heterogeneity functions.

        Parameters
        ----------
        E, nu, alpha, k, mu, M : floats
            Homogeneous material properties as in the base model.
        k_fun : callable or None
            Optional function `(x, y) -> k(x,y)` returning the
            nondimensional permeability at each point.  If `None`, a
            constant permeability `k` is used.
        E_fun : callable or None
            Optional function `(x, y) -> E(x,y)` returning the
            nondimensional Young's modulus.  If `None`, a constant
            modulus `E` is used.

        Returns
        -------
        tuple
            (static_params, trainable_params) suitable for FBPINN.
        """
        # Cast scalar parameters to JAX arrays
        E = jnp.array(E, dtype=jnp.float32)
        nu = jnp.array(nu, dtype=jnp.float32)
        alpha = jnp.array(alpha, dtype=jnp.float32)
        k = jnp.array(k, dtype=jnp.float32)
        mu = jnp.array(mu, dtype=jnp.float32)
        M = jnp.array(M, dtype=jnp.float32)

        # Default functions if none provided
        if k_fun is None:
            # wrap constant k into a function
            k_fun = lambda x, y, k_val=k: jnp.broadcast_to(k_val, x.shape)
        if E_fun is None:
            E_fun = lambda x, y, E_val=E: jnp.broadcast_to(E_val, x.shape)

        # Derive uniform stiffness parameters for scaling; local values
        # will be computed in the loss when heterogeneity is active.
        G = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        static_params = {
            "dims": (3, 3),
            "E": E,
            "nu": nu,
            "G": G,
            "lam": lam,
            "k": k,
            "mu": mu,
            "alpha": alpha,
            "M": M,
            # store the heterogeneity functions
            "k_fun": k_fun,
            "E_fun": E_fun
        }
        trainable_params = {}
        return static_params, trainable_params

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        """Impose hard BCs and ICs on the raw network output.

        This method is identical to the homogeneous version.  See
        `biot_coupled_homogeneous.py` for detailed documentation.
        """
        x = x_batch[:, 0:1]
        y = x_batch[:, 1:2]
        t = x_batch[:, 2:3]
        ux_raw = u[:, 0:1]
        uy_raw = u[:, 1:2]
        p_raw = u[:, 2:3]

        def S(t):
            return 1.0 - jnp.exp(-5.0 * jnp.clip(t, 0.0, 1.0))

        p0 = jnp.zeros_like(t)
        injection_center, sigma = 0.4, 0.08
        gauss_y = jnp.exp(-((y - injection_center) ** 2) / (2.0 * sigma ** 2))
        pL = p0 + S(t) * gauss_y
        pR = p0 * 0.0

        ux = S(t) * x * ux_raw
        uy = S(t) * x * y * uy_raw
        p = (
            p0
            + (1.0 - x) * (pL - p0)
            + x * (pR - p0)
            + x * (1.0 - x) * S(t) * p_raw
        )
        return jnp.concatenate([ux, uy, p], axis=-1)

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        """Sample collocation points and specify derivative orders.

        Reuses the specification from the homogeneous model.
        """
        dom = all_params["static"].setdefault("domain", {})
        d = all_params["static"]["problem"]["dims"][1]
        dom.setdefault("xmin", jnp.zeros((d,), dtype=jnp.float32))
        dom.setdefault("xmax", jnp.ones((d,), dtype=jnp.float32))
        dom.setdefault("xd", d)
        bs0 = batch_shapes[0]
        if sampler == "grid" and len(bs0) == 1 and d > 1:
            batch_shape_phys = (bs0[0],) * d
        else:
            batch_shape_phys = bs0
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shape_phys)
        required_ujs_phys = (
            (0, (0,)), (0, (0, 0)), (0, (1, 1)), (0, (0, 1)),
            (1, (1,)), (1, (0, 0)), (1, (1, 1)), (1, (0, 1)),
            (2, (0,)), (2, (1,)), (2, (0, 0)), (2, (1, 1)),
            (2, (2,)), (0, (0, 2)), (1, (1, 2))
        )
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
        """Compute the PDE residuals including heterogeneous permeability.

        Mechanics residual: ∇·σ = 0 with σ = 2G ε + λ tr(ε) I − α p I.
        Flow residual: (1/M) p_t + α ∂t(div u) − ∇·((k/μ) ∇p) = 0.

        If a spatially varying modulus `E_fun` is provided, local
        values of G and λ are computed at each collocation point.
        """
        prob = all_params["static"]["problem"]
        # Homogeneous baseline values (used for scaling)
        E0 = prob["E"]
        nu = prob["nu"]
        alpha = prob["alpha"]
        mu = prob["mu"]
        M = prob["M"]
        # Heterogeneous functions
        k_fun = prob.get("k_fun")
        E_fun = prob.get("E_fun")
        # Unpack derivatives and collocation points
        x_batch_phys = constraints[0][0]
        x = x_batch_phys[:, 0]
        y = x_batch_phys[:, 1]
        (
            duxdx, d2uxdx2, d2uxdy2, d2uxdxdy,
            duydy, d2uydx2, d2uydy2, d2uydxdy,
            dpdx, dpdy, d2pdx2, d2pdy2,
            p_t, duxdx_t, duydy_t
        ) = constraints[0][1:16]
        div_u = duxdx + duydy
        div_u_t = duxdx_t + duydy_t
        lap_p = d2pdx2 + d2pdy2
        # Compute local permeability and its gradients
        k_val = k_fun(x, y)
        # If k_fun depends smoothly on x,y, include gradient term
        def k_scalar(z):
            return k_fun(z[0], z[1])
        dk_dx = jax.vmap(jax.grad(k_scalar, argnums=0))(jnp.stack([x, y], axis=1))
        dk_dy = jax.vmap(jax.grad(k_scalar, argnums=1))(jnp.stack([x, y], axis=1))
        # Compute local elastic constants if E_fun is provided
        if E_fun is not None:
            E_local = E_fun(x, y)
            G_local = E_local / (2.0 * (1.0 + nu))
            lam_local = E_local * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        else:
            G_local = prob["G"]
            lam_local = prob["lam"]
        # Mechanics residuals
        equilibrium_x = (
            (2 * G_local + lam_local) * d2uxdx2
            + lam_local * d2uydxdy
            + G_local * d2uxdy2
            + G_local * d2uydxdy
            - alpha * dpdx
        )
        equilibrium_y = (
            G_local * d2uxdxdy
            + G_local * d2uydx2
            + lam_local * d2uxdxdy
            + (2 * G_local + lam_local) * d2uydy2
            - alpha * dpdy
        )
        # Flow residual
        div_q = (k_val * lap_p + dk_dx * dpdx + dk_dy * dpdy) / mu
        flow_residual = (p_t / M) + alpha * div_u_t - div_q
        # Characteristic scales for balancing losses
        L = 1.0
        p_scale = 1.0
        mech_scale = jnp.maximum(E0 / (L ** 2), 1e-6)
        flow_scale = jnp.maximum(jnp.max(k_val) / mu * (p_scale / (L ** 2)), 1e-8)
        mechanics_loss = jnp.mean((equilibrium_x / mech_scale) ** 2 + (equilibrium_y / mech_scale) ** 2)
        flow_loss = jnp.mean((flow_residual / flow_scale) ** 2)
        return mechanics_loss + flow_loss

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        return None


class BiotCoupledTrainer_Heterogeneous(BaseBiotTrainer):
    """
    Trainer for the heterogeneous Biot problem.  This class mirrors
    `BiotCoupledTrainer` but uses `BiotCoupled2D_Heterogeneous` as the
    problem.  It accepts optional permeability and modulus functions to
    customise the heterogeneity.
    """
    def __init__(self, k_fun=None, E_fun=None, n_steps=8000):
        # Configure domain and problem
        problem_init_kwargs = {
            'E': 5000.0,
            'nu': 0.25,
            'alpha': 0.8,
            'k': 1.0,
            'mu': 1.0,
            'M': 100.0,
            'k_fun': k_fun,
            'E_fun': E_fun
        }
        self.config = Constants(
            run="biot_coupled_2d_hetero",
            domain=RectangularDomainND,
            domain_init_kwargs={
                'xmin': jnp.array([0.0, 0.0, 0.0]),
                'xmax': jnp.array([1.0, 1.0, 1.0])
            },
            problem=BiotCoupled2D_Heterogeneous,
            problem_init_kwargs=problem_init_kwargs,
            decomposition=RectangularDecompositionND,
            decomposition_init_kwargs={
                'subdomain_xs': [
                    jnp.linspace(0, 1, 3),
                    jnp.linspace(0, 1, 3),
                    jnp.linspace(0, 1, 3)
                ],
                'subdomain_ws': [
                    0.8 * jnp.ones(3),
                    0.8 * jnp.ones(3),
                    0.8 * jnp.ones(3)
                ],
                'unnorm': (0., 1.)
            },
            network=FCN,
            network_init_kwargs={
                'layer_sizes': [3, 128, 128, 128, 3],
                'activation': 'tanh'
            },
            ns=((80, 80, 20), (0,), (0,), (0,), (0,)),
            n_test=(20, 20, 5),
            n_steps=n_steps,
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
        if n_steps is not None:
            self.config.n_steps = n_steps
            self.trainer.c.n_steps = n_steps
        print("Training heterogeneous model with hard BCs.")
        self.all_params = self.trainer.train()
        return self.all_params

    def predict(self, x_points, t=1.0):
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
        injection_center, sigma = 0.4, 0.08
        S = 1.0 - jnp.exp(-5.0 * float(t))
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

    # The metrics, plotting and history tracking methods could reuse the
    # implementations from the homogeneous trainer via inheritance or
    # delegation.  For brevity they are omitted here; the user can
    # directly call `compute_physics_metrics`, `plot_solution`, etc. on
    # this trainer since it inherits the methods from the homogeneous
    # version by composition of the FBPINN trainer.


def FixedTrainer():
    """Factory function for the heterogeneous trainer with default heterogeneity."""
    return BiotCoupledTrainer_Heterogeneous(
        k_fun=BiotCoupled2D_Heterogeneous.default_k_fun,
        E_fun=BiotCoupled2D_Heterogeneous.default_E_fun
    )