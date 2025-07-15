# FLUID FLOW

import numpy as np
import jax
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
    Governing PDE: ∂/∂t[α∇·u + (1/M)p] + ∇·q = f
    For steady state: α∇·u + (1/M)p + ∇·(-k/μ ∇p) = 0
    Simplifying: α∇·u - (k/μ)∇²p + (1/M)p = 0
    """

    @staticmethod
    def init_params(k=1.0, mu=1.0, alpha = 0.8, M = 1000.0):
        static_params = {
            "dims": (1, 2),  # 1 output (p), 2 inputs (x,y)
            "k": k,
            "mu": mu,
            "alpha": alpha,
            "M": M
        }
        trainable_params = {}
        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0, ()), # p
            (0,(0,)), # dp/dx
            (0, (1)), # dp/dy
            (0, (0,0)), # d2p/dx2
            (0, (1,1)) # d2p/dy2
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
    def loss_fn(all_params, constraints, external_div_u):
        k = all_params["static"]["problem"]["k"]
        mu = all_params["static"]["problem"]["mu"]
        alpha = all_params["static"]["problem"]["alpha"]
        M = all_params["static"]["problem"]["M"]

        (x_batch_phys, p, dpdx, dpdy, d2pdx2, d2pdy2) = constraints[0]

        # Biot's flow equation (steady state)
        #α∇·u - (k/μ)∇²p + (1/M)p = 0
        laplacian_p = d2pdx2 + d2pdy2
        flow_residual = alpha * external_div_u -(k/mu) * laplacian_p + (1.0/M) * p
        physics_loss = jnp.mean(flow_residual**2)

        # Boundary losses
        x_batch_left, p_target_left, p_pred_left = constraints[1]
        left_bc_loss = jnp.mean((p_pred_left - p_target_left)**2)

        x_batch_right, p_target_right, p_pred_right = constraints[2]
        right_bc_loss = jnp.mean((p_pred_right - p_target_right)**2)

        # Gradient based adaptive weighting
        sum_of_boundary_losses = left_bc_loss + right_bc_loss
        physics_loss_scale = jax.lax.stop_gradient(physics_loss)
        boundary_loss_scale = jax.lax.stop_gradient(sum_of_boundary_losses)
        boundary_weight = physics_loss_scale/(boundary_loss_scale + 1e-8)
        boundary_weight = jnp.clip(boundary_weight, 1.0, 10000.0)

        coupling_strength = jnp.mean((alpha * external_div_u)**2)

        total_loss = physics_loss + boundary_weight * sum_of_boundary_losses
        return total_loss
    
class FlowTrainerWithGradients(FBPINNTrainer):
    " Extended trainer that can return pressure gradients"

    def predict_with_gradients(self, all_params, x_test):
        # Get network prediction including gradients
        required_ujs = ((0, ()), # p
                        (0, ()), # dp/dx
                        (0, (1))) # dp/dy
        
        u_pred = self.network.u_pred_aug(all_params, x_test, required_ujs)
        pressure = u_pred[0] # p
        dpdx = u_pred[1] # dp/dx
        dpdy = u_pred[2] # dp/dy

        return pressure, dpdx, dpdy


def FlowTrainer():
    config = Constants(
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
        ns=((100, 100), (25,), (25,), (25,),),
        n_test=(15, 15),
        n_steps=1,
        optimiser_kwargs={'learning_rate': 1e-5},
        summary_freq=10,
        test_freq=50,
        show_figures=False,
        save_figures=False,
        clear_output=True
    )

    trainer = FlowTrainerWithGradients(config)
    #trainer = FBPINNTrainer(config)
    all_params = trainer.init_params()
    return trainer, all_params, config