# MECHANICS

import numpy as np
import jax
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
    Governing PDE: ∇·σ' + ρb = 0, where σ' = σ - αpI
    Expanding: ∇·σ - α∇p + ρb = 0
    For no body forces: ∇·σ - α∇p = 0
    """

    @staticmethod
    def init_params(E=5000.0, nu=0.25, alpha=0.8):
        G = E / (2.0 * (1.0 + nu)) # shear modulus
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) # Lame parameter
        K = E / (3.0 * (1.0 - 2.0 * nu)) # Bulk modulus

        static_params = {
            "dims": (2, 2),
            "E": E,
            "nu": nu,
            "G": G,
            "lam": lam,
            "K": K,
            "alpha": alpha
        }
        trainable_params = {}
        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0, ()), 
            (0, (0,)), 
            (0, (1,)), 
            (0, (0,0)), 
            (0, (1,1)), 
            (0, (0,1)),
            (1, ()), 
            (1, (0,)), 
            (1, (1,)), 
            (1, (0,0)), 
            (1, (1,1)), 
            (1, (0,1))
        )

        # Boundaries (similar to elasticity)
        boundary_batch_shapes = ((25,), (25,), (25,), (25,))
        x_batches_boundaries = domain.sample_boundaries(all_params, key, sampler, boundary_batch_shapes)

        # Left Boundary (u = 0)
        x_batch_left = x_batches_boundaries[0]
        ux_target_left = jnp.zeros((x_batch_left.shape[0], 1))
        uy_target_left = jnp.zeros((x_batch_left.shape[0], 1))
        required_ujs_ux_left = ((0, ()),)
        required_ujs_uy_left = ((1, ()),)

        # Right Boundary : traction free (du/dn = 0)
        x_batch_right = x_batches_boundaries[1]
        required_ujs_right = ((0, (0,)), (1, (1,)))

        # Bottom boundary: fixed vertical displacement (uy = 0)
        x_batch_bottom = x_batches_boundaries[2]
        uy_target_bottom = jnp.zeros((x_batch_bottom.shape[0], 1))
        required_ujs_bottom = ((1, ()),)

        # Top boundary : traction_free
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
    def loss_fn(all_params, constraints, external_pressure):
        G = all_params["static"]["problem"]["G"]
        lam = all_params["static"]["problem"]["lam"]
        alpha = all_params["static"]["problem"]["alpha"]

        (x_batch_phys, ux, duxdx, duxdy, d2uxdx2, d2uxdy2, d2uxdxdy,
         uy, duydx, duydy, d2uydx2, d2uydy2, d2uydydx) = constraints[0]
        
        # pressure gradient directly from flow network
        dpdx, dpdy = external_pressure
        if dpdx.shape[0] != x_batch_phys.shape[0]:
            print(f"Warning: Gradient shape mismatch. dpdx: {dpdx.shape}, physics: {x_batch_phys.shape}")

        # linear elasticity + Biot's coupling
        # Stress equilibrium: ∇·σ - α∇p = 0
        # σxx = (λ + 2G)∂ux/∂x + λ∂uy/∂y
        # σyy = (λ + 2G)∂uy/∂y + λ∂ux/∂x  
        # σxy = G(∂ux/∂y + ∂uy/∂x)

        # ∂σxx/∂x + ∂σxy/∂y - α∂p/∂x = 0
        equilibrium_x = ((lam + 2*G) * d2uxdx2 + lam * d2uydydx 
                         + G * (d2uxdy2 + d2uydydx ) - alpha * dpdx)
        
        # ∂σxy/∂x + ∂σyy/∂y - α∂p/∂y = 0  
        equilibrium_y = (G * (d2uydx2 + d2uxdxdy) + 
                        (lam + 2*G) * d2uydy2 + lam * d2uxdxdy - alpha * dpdy)
        
        physics_loss = jnp.mean(equilibrium_x**2) + jnp.mean(equilibrium_y**2)


        # Mechanics residual (includes coupling term)
        #equilibrium_x = (2*G + lam)*d2uxdx2 + G*d2uxdy2 + (G+lam)*d2uydydx + alpha*dpdx
        #equilibrium_y = (2*G + lam)*d2uydy2 + G*d2uydx2 + (G+lam)*d2uxdxdy + alpha*dpdy
        #physics_loss = jnp.mean(equilibrium_x**2) + jnp.mean(equilibrium_y**2)

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

        sum_of_boundary_losses = left_ux_loss + left_uy_loss + right_loss + bottom_loss + top_loss
        physics_loss_scale = jax.lax.stop_gradient(physics_loss)
        boundary_loss_scale = jax.lax.stop_gradient(sum_of_boundary_losses)
        boundary_weight = physics_loss_scale/(boundary_loss_scale + 1e-8)
        boundary_weight = jnp.clip(boundary_weight, 10.0, 50000.0)

        coupling_strength = jnp.mean((alpha * dpdx)**2) + jnp.mean((alpha * dpdy)**2)
        
        total_loss = physics_loss + boundary_weight * sum_of_boundary_losses
        return total_loss

def MechanicsTrainer():
    config = Constants(
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
        summary_freq=10,
        test_freq=50,
        show_figures=False,
        save_figures=False,
        clear_output=True
    )

    trainer = FBPINNTrainer(config)
    all_params = trainer.init_params()
    return trainer, all_params, config