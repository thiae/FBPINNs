# FLUID FLOW

import numpy as np
import jax.numpy as jnp
from fbpinns.domains import RectangularDomainND
from fbpinns.problems import Problem
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants
from fbpinns.trainers import FBPINNTrainer

class BiotFlow2D(Problem):
    """
    2D Biot Flow (pressure p)
    Governing PDE: -∇·(k∇p) + α∇·u = 0
    """

    @staticmethod
    def init_params(k=1.0, mu=1.0, alpha=0.8):
        static_params = {
            "dims": (1, 2),  # 1 output (p), 2 inputs (x,y)
            "k": k,
            "mu": mu,
            "alpha": alpha
        }
        trainable_params = {}
        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0, ()), (0, (0,0)), (0, (1,1))
        )

        boundary_batch_shapes = ((25,), (25,))
        x_batches_boundaries = domain.sample_boundaries(all_params, key, sampler, boundary_batch_shapes)

        # Left boundary: p=1
        x_batch_left = x_batches_boundaries[0]
        p_target_left = jnp.ones((x_batch_left.shape[0], 1))
        required_ujs_left = ((0, ()),)

        # Right boundary: p=0
        x_batch_right = x_batches_boundaries[1]
        p_target_right = jnp.zeros((x_batch_right.shape[0], 1))
        required_ujs_right = ((0, ()),)

        return [
            [x_batch_phys, required_ujs_phys],
            [x_batch_left, p_target_left, required_ujs_left],
            [x_batch_right, p_target_right, required_ujs_right]
        ]

    @staticmethod
    def loss_fn(all_params, constraints, external_div_u=None):
        """
        Fixed loss function that properly uses external_div_u for coupling
        """
        k = all_params["static"]["problem"]["k"]
        alpha = all_params["static"]["problem"]["alpha"]

        (x_batch_phys, p, d2pdx2, d2pdy2) = constraints[0]
        
        # Flow equation: -k∇²p + α∇·u = 0
        # Rearranged: k∇²p = α∇·u
        laplacian_p = d2pdx2 + d2pdy2
        
        if external_div_u is not None:
            # Proper coupling: flow residual includes divergence of displacement
            # Ensure external_div_u has correct shape
            if external_div_u.shape[0] != x_batch_phys.shape[0]:
                # If shapes don't match, interpolate or pad
                n_points = x_batch_phys.shape[0]
                if external_div_u.shape[0] > n_points:
                    external_div_u = external_div_u[:n_points]
                else:
                    # Repeat the last values
                    repeats = n_points - external_div_u.shape[0]
                    last_val = external_div_u[-1:] 
                    external_div_u = jnp.concatenate([external_div_u, 
                                                    jnp.repeat(last_val, repeats, axis=0)])
            
            flow_residual = -k * laplacian_p + alpha * external_div_u.flatten()
        else:
            # Uncoupled case: just solve Laplace equation
            flow_residual = -k * laplacian_p
            
        physics_loss = jnp.mean(flow_residual**2)

        # Boundary losses
        x_batch_left, p_target_left, p_pred_left = constraints[1]
        left_bc_loss = jnp.mean((p_pred_left - p_target_left)**2)

        x_batch_right, p_target_right, p_pred_right = constraints[2]
        right_bc_loss = jnp.mean((p_pred_right - p_target_right)**2)

        # Weight boundary conditions heavily for stability
        total_loss = physics_loss + 1000.0 * (left_bc_loss + right_bc_loss)
        return total_loss

class CoupledFlowTrainer:
    """Wrapper class to handle coupling for flow trainer"""
    
    def __init__(self):
        self.config = Constants(
            run="biot_flow_2d",
            domain=RectangularDomainND,
            domain_init_kwargs={'xmin': np.array([0., 0.]), 'xmax': np.array([1., 1.])},
            problem=BiotFlow2D,
            problem_init_kwargs={'k': 1.0, 'mu': 1.0, 'alpha': 0.8},
            decomposition=RectangularDecompositionND,
            decomposition_init_kwargs={
                'subdomain_xs': [np.linspace(0, 1, 4), np.linspace(0, 1, 3)],
                'subdomain_ws': [0.5 * np.ones(4), 0.7 * np.ones(3)],
                'unnorm': (0., 1.)
            },
            network=FCN,
            network_init_kwargs={'layer_sizes': [2, 128, 128, 1], 'activation': 'tanh'},
            ns=((100, 100), (25,), (25,)),
            n_test=(15, 15),
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
        self.external_div_u = None  # Store coupling term
    
    def set_coupling_term(self, external_div_u):
        """Set the external divergence of displacement for coupling"""
        self.external_div_u = external_div_u
    
    def train_step(self, n_steps=1):
        """Train with current coupling term"""
        # Get the problem class and original loss function
        problem_class = self.trainer.c.problem
        original_loss_fn = problem_class.loss_fn
        
        # Create coupled loss function
        if hasattr(self, 'external_div_u'):  # Flow trainer
            def coupled_loss_fn(all_params, constraints):
                return original_loss_fn(all_params, constraints, self.external_div_u)
        else:  # Mechanics trainer
            def coupled_loss_fn(all_params, constraints):
                return original_loss_fn(all_params, constraints, self.external_pressure)
        
        # Temporarily replace the loss function
        problem_class.loss_fn = coupled_loss_fn
        
        # Set number of steps and train
        old_n_steps = self.trainer.c.n_steps
        self.trainer.c.n_steps = n_steps
        
        # Train
        self.all_params = self.trainer.train(self.all_params)
        
        # Restore original settings
        self.trainer.c.n_steps = old_n_steps
        problem_class.loss_fn = original_loss_fn
        
        return self.all_params
    
    def predict(self, x_points):
        """Predict pressure at given points"""
        return self.trainer.predict(self.all_params, x_points)
    
    def get_test_points(self):
        """Get test points for evaluation"""
        return self.trainer.get_batch(self.all_params, self.config.n_test, 'test')

def FlowTrainer():
    """Original interface for compatibility"""
    flow_trainer = CoupledFlowTrainer()
    return flow_trainer, flow_trainer.all_params, flow_trainer.config