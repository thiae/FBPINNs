# MECHANICS

import numpy as np
import jax.numpy as jnp
from fbpinns.domains import RectangularDomainND
from fbpinns.problems import Problem
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants
from fbpinns.trainers import FBPINNTrainer

class BiotMechanics2D(Problem):
    """
    2D Biot Mechanics (displacements u_x, u_y)
    Governing PDE: ∇·σ' + α∇p = 0
    """

    @staticmethod
    def init_params(E=5000.0, nu=0.25, alpha=0.8):
        G = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        static_params = {
            "dims": (2, 2),
            "E": E,
            "nu": nu,
            "G": G,
            "lam": lam,
            "alpha": alpha
        }
        trainable_params = {}
        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0, ()), (0, (0,)), (0, (1,)), (0, (0,0)), (0, (1,1)), (0, (0,1)),
            (1, ()), (1, (0,)), (1, (1,)), (1, (0,0)), (1, (1,1)), (1, (0,1))
        )

        # Boundaries
        boundary_batch_shapes = ((25,), (25,), (25,), (25,))
        x_batches_boundaries = domain.sample_boundaries(all_params, key, sampler, boundary_batch_shapes)

        x_batch_left = x_batches_boundaries[0]
        ux_target_left = jnp.zeros((x_batch_left.shape[0], 1))
        uy_target_left = jnp.zeros((x_batch_left.shape[0], 1))
        required_ujs_ux_left = ((0, ()),)
        required_ujs_uy_left = ((1, ()),)

        x_batch_right = x_batches_boundaries[1]
        required_ujs_right = ((0, (0,)), (1, (1,)))

        x_batch_bottom = x_batches_boundaries[2]
        uy_target_bottom = jnp.zeros((x_batch_bottom.shape[0], 1))
        required_ujs_bottom = ((1, ()),)

        x_batch_top = x_batches_boundaries[3]
        required_ujs_top = ((0, (0,)), (1, (1,)))

        return [
            [x_batch_phys, required_ujs_phys],
            [x_batch_left, ux_target_left, required_ujs_ux_left],
            [x_batch_left, uy_target_left, required_ujs_uy_left],
            [x_batch_right, required_ujs_right],
            [x_batch_bottom, uy_target_bottom, required_ujs_bottom],
            [x_batch_top, required_ujs_top]
        ]

    @staticmethod
    def loss_fn(all_params, constraints, external_pressure=None):
        """
        Fixed loss function that properly uses external_pressure for coupling
        """
        G = all_params["static"]["problem"]["G"]
        lam = all_params["static"]["problem"]["lam"]
        alpha = all_params["static"]["problem"]["alpha"]

        (x_batch_phys, ux, duxdx, duxdy, d2uxdx2, d2uxdy2, d2uxdxdy,
         uy, duydx, duydy, d2uydx2, d2uydy2, d2uydxdy) = constraints[0]

        if external_pressure is not None:
            # Use external pressure from flow solver
            # Ensure external_pressure has correct shape
            if external_pressure.shape[0] != x_batch_phys.shape[0]:
                n_points = x_batch_phys.shape[0]
                if external_pressure.shape[0] > n_points:
                    external_pressure = external_pressure[:n_points]
                else:
                    # Repeat the last values
                    repeats = n_points - external_pressure.shape[0]
                    last_val = external_pressure[-1:]
                    external_pressure = jnp.concatenate([external_pressure,
                                                       jnp.repeat(last_val, repeats, axis=0)])
            
            # Compute pressure gradients using finite differences on the pressure field
            # Reshape to 2D grid for gradient computation
            n_grid = int(jnp.sqrt(x_batch_phys.shape[0]))
            if n_grid * n_grid == x_batch_phys.shape[0]:
                p_grid = external_pressure.reshape((n_grid, n_grid))
                dpdx_grid, dpdy_grid = jnp.gradient(p_grid)
                dpdx = dpdx_grid.flatten().reshape(-1, 1)
                dpdy = dpdy_grid.flatten().reshape(-1, 1)
            else:
                # Fallback: use simple finite differences
                dpdx = jnp.gradient(external_pressure.flatten())
                dpdy = jnp.zeros_like(dpdx)
                dpdx = dpdx.reshape(-1, 1) 
                dpdy = dpdy.reshape(-1, 1)
        else:
            # No coupling: set pressure gradients to zero
            dpdx = jnp.zeros((x_batch_phys.shape[0], 1))
            dpdy = jnp.zeros((x_batch_phys.shape[0], 1))

        # Mechanics equilibrium equations with coupling term
        # X-momentum: ∇·σ'_x + α∂p/∂x = 0
        equilibrium_x = ((2*G + lam)*d2uxdx2 + G*d2uxdy2 + 
                        (G + lam)*d2uydxdy + alpha*dpdx.flatten())
        
        # Y-momentum: ∇·σ'_y + α∂p/∂y = 0  
        equilibrium_y = (G*d2uxdxdy + (G + lam)*d2uydx2 + 
                        (2*G + lam)*d2uydy2 + alpha*dpdy.flatten())
        
        physics_loss = jnp.mean(equilibrium_x**2) + jnp.mean(equilibrium_y**2)

        # Boundary losses
        x_batch_left_ux, ux_target_left, ux_pred_left = constraints[1]
        left_ux_loss = jnp.mean((ux_pred_left - ux_target_left)**2)

        x_batch_left_uy, uy_target_left, uy_pred_left = constraints[2]
        left_uy_loss = jnp.mean((uy_pred_left - uy_target_left)**2)

        x_batch_right, dux_dx_right, duy_dy_right = constraints[3]
        right_loss = jnp.mean(dux_dx_right**2) + jnp.mean(duy_dy_right**2)

        x_batch_bottom, uy_target_bottom, uy_pred_bottom = constraints[4]
        bottom_loss = jnp.mean((uy_pred_bottom - uy_target_bottom)**2)

        x_batch_top, dux_dx_top, duy_dy_top = constraints[5]
        top_loss = jnp.mean(dux_dx_top**2) + jnp.mean(duy_dy_top**2)

        boundary_loss = left_ux_loss + left_uy_loss + right_loss + bottom_loss + top_loss
        
        # Weight boundary conditions for stability
        total_loss = physics_loss + 100.0 * boundary_loss
        return total_loss

class CoupledMechanicsTrainer:
    """Wrapper class to handle coupling for mechanics trainer"""
    
    def __init__(self):
        self.config = Constants(
            run="biot_mechanics_2d",
            domain=RectangularDomainND,
            domain_init_kwargs={'xmin': np.array([0., 0.]), 'xmax': np.array([1., 1.])},
            problem=BiotMechanics2D,
            problem_init_kwargs={'E': 5000.0, 'nu': 0.25, 'alpha': 0.8},
            decomposition=RectangularDecompositionND,
            decomposition_init_kwargs={
                'subdomain_xs': [np.linspace(0,1,4), np.linspace(0,1,3)],
                'subdomain_ws': [0.5*np.ones(4), 0.7*np.ones(3)],
                'unnorm': (0.,1.)
            },
            network=FCN,
            network_init_kwargs={'layer_sizes': [2, 128, 128, 2], 'activation': 'tanh'},
            ns=((100,100), (25,), (25,), (25,), (25,), (25,)),
            n_test=(15,15),
            n_steps=1,
            optimiser_kwargs={'learning_rate': 1e-5},
            summary_freq=100,
            test_freq=500,
            show_figures=False,
            save_figures=False,
            clear_output=True
        )
        self.trainer = FBPINNTrainer(self.config)
        self.all_params = None
        self.external_pressure = None  # Store coupling term
    
    def set_coupling_term(self, external_pressure):
        """Set the external pressure for coupling"""
        self.external_pressure = external_pressure
    
    def train_step(self, n_steps=1):
        """Train with current coupling term"""
        # Temporarily modify the loss function to include coupling
        original_loss_fn = self.trainer.problem.loss_fn
        
        def coupled_loss_fn(all_params, constraints):
            return original_loss_fn(all_params, constraints, self.external_div_u)  # for flow
            # OR: return original_loss_fn(all_params, constraints, self.external_pressure)  # for mechanics
        
        # Replace loss function temporarily
        self.trainer.problem.loss_fn = coupled_loss_fn
        
        # Set number of steps and train
        old_n_steps = self.config.n_steps
        self.config.n_steps = n_steps
        
        # Train and get parameters
        self.all_params = self.trainer.train(self.all_params)
        
        # Restore original settings
        self.config.n_steps = old_n_steps
        self.trainer.problem.loss_fn = original_loss_fn
        
        return self.all_params
    
    def predict(self, x_points):
        """Predict displacement at given points"""
        return self.trainer.predict(self.all_params, x_points)
    
    def get_divergence(self, x_points):
        """Compute divergence of displacement field"""
        u_pred = self.predict(x_points)
        
        # Compute divergence using finite differences
        n_grid = int(jnp.sqrt(x_points.shape[0]))
        if n_grid * n_grid == x_points.shape[0]:
            ux_grid = u_pred[:, 0].reshape((n_grid, n_grid))
            uy_grid = u_pred[:, 1].reshape((n_grid, n_grid))
            
            dux_dx, _ = jnp.gradient(ux_grid)
            _, duy_dy = jnp.gradient(uy_grid)
            
            div_u = dux_dx + duy_dy
            return div_u.flatten().reshape(-1, 1)
        else:
            # Fallback: return zeros
            return jnp.zeros((x_points.shape[0], 1))
    
    def get_test_points(self):
        """Get test points for evaluation"""
        return self.trainer.get_batch(self.all_params, self.config.n_test, 'test')

def MechanicsTrainer():
    """Original interface for compatibility"""
    mech_trainer = CoupledMechanicsTrainer()
    return mech_trainer, mech_trainer.all_params, mech_trainer.config