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
    Unified 2D Biot Poroelasticity Problem
    Solves both mechanics (u_x, u_y) and flow (p) in one problem class
    
    THE FIX: Properly implements hard BC enforcement through the solution method
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
        
        # Calculate the derived parameters
        G = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

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
        d = all_params["static"]["problem"]["dims"][1]
        dom.setdefault("xmin", jnp.zeros((d,), dtype=jnp.float32))
        dom.setdefault("xmax", jnp.ones((d,),  dtype=jnp.float32))
        dom.setdefault("xd", d)

        bs0 = batch_shapes[0]
        if sampler == "grid" and len(bs0) == 1 and d > 1:
            batch_shape_phys = (bs0[0],) * d
        else:
            batch_shape_phys = bs0

        # Sample interior points
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shape_phys)

        # Required derivatives for physics equations
        required_ujs_phys = (
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
            # p derivatives
            (2, (0,)),
            (2, (1,)), 
            (2, (0,0)), 
            (2, (1,1))
        )
        
        # For hard BC enforcement, we still need boundary points but with minimal constraints
        boundary_batch_shapes = batch_shapes[1:5]
        x_batches_boundaries = domain.sample_boundaries(all_params, key, sampler, boundary_batch_shapes)

        # Simplified boundary constraints - just sample the points
        x_batch_left = x_batches_boundaries[0]
        x_batch_right = x_batches_boundaries[1]
        x_batch_bottom = x_batches_boundaries[2]
        x_batch_top = x_batches_boundaries[3]

        return [
            # Physics constraints only
            [x_batch_phys, required_ujs_phys],
            # Boundary points (minimal constraints for hard BC)
            [x_batch_left],
            [x_batch_right],
            [x_batch_bottom],
            [x_batch_top]
        ]

    @staticmethod
    def loss_fn(all_params, constraints):
        """
        Loss function focusing on physics with hard BC enforcement
        """
        # Get material parameters
        G = all_params["static"]["problem"]["G"]
        lam = all_params["static"]["problem"]["lam"]
        alpha = all_params["static"]["problem"]["alpha"]
        k = all_params["static"]["problem"]["k"]

        # Physics constraints
        x_batch_phys = constraints[0][0]
        duxdx, d2uxdx2, d2uxdy2, d2uxdxdy, duydy, d2uydx2, d2uydy2, d2uydxdy, dpdx, dpdy, d2pdx2, d2pdy2 = constraints[0][1:13]
        
        # Compute divergence of displacement
        div_u = duxdx + duydy
        
        # Mechanics equations
        equilibrium_x = ((2*G + lam)*d2uxdx2 + lam*d2uydxdy + 
                        G*d2uxdy2 + G*d2uydxdy + alpha*dpdx)
        
        equilibrium_y = (G*d2uxdxdy + G*d2uydx2 + 
                        lam*d2uxdxdy + (2*G + lam)*d2uydy2 + alpha*dpdy)
        
        mechanics_loss = jnp.mean(equilibrium_x**2) + jnp.mean(equilibrium_y**2)
        
        # Flow equation
        laplacian_p = d2pdx2 + d2pdy2
        flow_residual = -k * laplacian_p + alpha * div_u
        flow_loss = jnp.mean(flow_residual**2)

        # With hard BCs, we don't need strong BC penalties
        total_loss = mechanics_loss + flow_loss
        
        return total_loss

    @staticmethod
    def solution(all_params, x_batch, batch_shape=None):
        """
        CRITICAL FIX: This is the method that FBPINNs actually calls!
        Transform network output to satisfy boundary conditions exactly.
        """
        # Get network function from all_params
        net_fn = all_params["nn"]
        
        # Get raw network output
        u_raw = net_fn(all_params, x_batch)
        
        x = x_batch[:, 0:1]  # x coordinates
        y = x_batch[:, 1:2]  # y coordinates
        
        # Raw outputs
        ux_raw = u_raw[:, 0:1]
        uy_raw = u_raw[:, 1:2]
        p_raw = u_raw[:, 2:3]
        
        # HARD BC ENFORCEMENT
        # Pressure: p(0,y)=1, p(1,y)=0
        # Use distance functions for smooth enforcement
        p = (1.0 - x) + p_raw * x * (1.0 - x) * 4.0  # Enhanced interior variation
        
        # X-displacement: ux(0,y)=0
        ux = ux_raw * x
        
        # Y-displacement: uy(x,0)=0, uy(0,y)=0
        uy = uy_raw * x * y
        
        return jnp.concatenate([ux, uy, p], axis=-1)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        """No exact solution - returns None to indicate physics-only training"""
        return None


class BiotCoupledTrainer:
    """
    Fixed trainer with proper hard BC enforcement
    """
    
    def __init__(self):
        """Initialize with optimal settings for hard BC training"""
        
        self.config = Constants(
            run="biot_coupled_2d",
            domain=RectangularDomainND,
            domain_init_kwargs={'xmin': jnp.array([0., 0.]), 'xmax': jnp.array([1., 1.])},
            problem=BiotCoupled2D,
            problem_init_kwargs={'E': 5000.0, 'nu': 0.25, 'alpha': 0.8, 'k': 1.0, 'mu': 1.0},
            decomposition=RectangularDecompositionND,
            decomposition_init_kwargs={
                'subdomain_xs': [jnp.linspace(0, 1, 3), jnp.linspace(0, 1, 3)],  # Simplified
                'subdomain_ws': [0.5 * jnp.ones(3), 0.5 * jnp.ones(3)],
                'unnorm': (0., 1.)
            },
            network=FCN,
            network_init_kwargs={'layer_sizes': [2, 32, 32, 32, 3], 'activation': 'tanh'},  # Smaller network
            ns=((40, 40), (50,), (50,), (50,), (50,)),  # Balanced sampling
            n_test=(20, 20),
            n_steps=1000,  # Moderate training
            optimiser_kwargs={
                'learning_rate': 1e-3,  # Higher learning rate for faster convergence
            },
            summary_freq=100,
            test_freq=250,
            show_figures=False,
            save_figures=False,
            clear_output=True
        )
        self.trainer = FBPINNTrainer(self.config)
        self.all_params = None
    
    def train(self, n_steps=None):
        """Train with hard BC enforcement"""
        if n_steps is not None:
            self.config.n_steps = n_steps
            self.trainer.c.n_steps = n_steps
        
        print("Training with mathematically enforced boundary conditions")
        print("BCs are guaranteed to be satisfied!")
        
        self.all_params = self.trainer.train()
        return self.all_params
    
    def predict(self, x_points):
        """Predict with hard BC enforcement"""
        if self.all_params is None:
            raise ValueError("Model not trained yet")
        
        # Use the solution method which includes hard BC enforcement
        return BiotCoupled2D.solution(self.all_params, x_points)
    
    def get_displacement(self, x_points):
        """Get displacement field [u_x, u_y]"""
        pred = self.predict(x_points)
        return pred[:, :2]
    
    def get_pressure(self, x_points):
        """Get pressure field [p]"""
        pred = self.predict(x_points)
        return pred[:, 2:3]
    
    def verify_bcs(self, n_points=100):
        """Verify that boundary conditions are satisfied"""
        # Check left boundary (x=0)
        y_test = jnp.linspace(0, 1, n_points)
        left_points = jnp.column_stack([jnp.zeros(n_points), y_test])
        left_pred = self.predict(left_points)
        
        # Check right boundary (x=1)
        right_points = jnp.column_stack([jnp.ones(n_points), y_test])
        right_pred = self.predict(right_points)
        
        # Check bottom boundary (y=0)
        x_test = jnp.linspace(0, 1, n_points)
        bottom_points = jnp.column_stack([x_test, jnp.zeros(n_points)])
        bottom_pred = self.predict(bottom_points)
        
        print("\nBoundary Condition Verification:")
        print("=" * 50)
        print(f"Left boundary (x=0):")
        print(f"  ux = 0: max violation = {jnp.max(jnp.abs(left_pred[:, 0])):.6e}")
        print(f"  uy = 0: max violation = {jnp.max(jnp.abs(left_pred[:, 1])):.6e}")
        print(f"  p = 1:  max violation = {jnp.max(jnp.abs(left_pred[:, 2] - 1.0)):.6e}")
        
        print(f"\nRight boundary (x=1):")
        print(f"  p = 0:  max violation = {jnp.max(jnp.abs(right_pred[:, 2])):.6e}")
        
        print(f"\nBottom boundary (y=0):")
        print(f"  uy = 0: max violation = {jnp.max(jnp.abs(bottom_pred[:, 1])):.6e}")
        
        # Check if all violations are small
        all_violations = [
            jnp.max(jnp.abs(left_pred[:, 0])),
            jnp.max(jnp.abs(left_pred[:, 1])),
            jnp.max(jnp.abs(left_pred[:, 2] - 1.0)),
            jnp.max(jnp.abs(right_pred[:, 2])),
            jnp.max(jnp.abs(bottom_pred[:, 1]))
        ]
        
        max_violation = max(all_violations)
        if max_violation < 1e-6:
            print("\nStatus: EXCELLENT - All BCs satisfied to machine precision")
        elif max_violation < 1e-3:
            print("\nStatus: GOOD - All BCs satisfied within tolerance")
        elif max_violation < 1e-2:
            print("\nStatus: ACCEPTABLE - Minor BC violations")
        else:
            print(f"\nStatus: WARNING - Significant BC violations (max: {max_violation:.3e})")
        
        return max_violation < 1e-2
    
    def plot_solution(self, n_points=50):
        """Plot all solution fields"""
        x = jnp.linspace(0, 1, n_points)
        y = jnp.linspace(0, 1, n_points)
        X, Y = jnp.meshgrid(x, y)
        points = jnp.column_stack([X.flatten(), Y.flatten()])
        
        pred = self.predict(points)
        
        UX = pred[:, 0].reshape(n_points, n_points)
        UY = pred[:, 1].reshape(n_points, n_points)
        P = pred[:, 2].reshape(n_points, n_points)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # X-displacement
        c1 = axes[0].contourf(X, Y, UX, levels=20, cmap='RdBu')
        axes[0].set_title('X-Displacement')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(c1, ax=axes[0])
        
        # Y-displacement
        c2 = axes[1].contourf(X, Y, UY, levels=20, cmap='RdBu')
        axes[1].set_title('Y-Displacement')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(c2, ax=axes[1])
        
        # Pressure
        c3 = axes[2].contourf(X, Y, P, levels=20, cmap='plasma')
        axes[2].set_title('Pressure')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(c3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
        
        # Print field statistics
        print("\nField Statistics:")
        print(f"ux: min={jnp.min(UX):.4f}, max={jnp.max(UX):.4f}")
        print(f"uy: min={jnp.min(UY):.4f}, max={jnp.max(UY):.4f}")
        print(f"p:  min={jnp.min(P):.4f}, max={jnp.max(P):.4f}")
        
        return fig, axes

# Create convenient functions
def FixedTrainer():
    """Create a trainer with working hard BC enforcement"""
    return BiotCoupledTrainer()

def CoupledTrainer():
    """Compatibility alias"""
    return BiotCoupledTrainer()

def ResearchTrainer():
    """Compatibility alias"""
    return BiotCoupledTrainer()