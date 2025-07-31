import os
import sys
import pytest
import jax
import jax.numpy as jnp

# Add the parent directory to Python path to find modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Also add the poroelasticity directory to find trainers
poroelasticity_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, poroelasticity_dir)

from fbpinns.domains import RectangularDomainND
from trainers.biot_trainer_2d import BiotCoupled2D, BiotCoupledTrainer  

@pytest.fixture
def mock_problem():
    # Initialise static parameters and dummy state
    static, _ = BiotCoupled2D.init_params()
    all_params = {"static": {"problem": static}, "step": 0}
    # Create domain and sampler key
    dom = RectangularDomainND()
    # Initialize domain parameters properly
    domain_static, _ = dom.init_params(jnp.array([0., 0.]), jnp.array([1., 1.]))
    all_params["static"]["domain"] = domain_static
    
    key = jax.random.PRNGKey(0)
    # Define batch shapes: 10 interior points, 5 per boundary
    # Note: grid sampler converts (10,) to (10,10) in 2D, so we get 100 points
    shapes = ((10,), (5,), (5,), (5,), (5,))
    # Sample constraints
    constraints = BiotCoupled2D.sample_constraints(
        all_params, dom, key, sampler="grid", batch_shapes=shapes
    )
    return all_params, constraints

def test_sample_constraints_structure(mock_problem):
    all_params, cons = mock_problem
    # Expect 1 interior + 4 BC blocks
    assert len(cons) == 5, "Expected 5 constraint blocks"
    
    # Interior: x_batch + required_ujs (12 derivatives) : framework adds derivatives
    interior_constraint = cons[0]
    assert len(interior_constraint) >= 2, "Interior should have at least x_batch and required_ujs"
    x_phys = interior_constraint[0] 
    assert isinstance(x_phys, jnp.ndarray)
    assert x_phys.shape == (100, 2)  # Grid sampler converts (10,) to (10,10) = 100 points

    # Left BC: x_batch + 3 targets + required_ujs : framework adds derivatives
    left_constraint = cons[1]
    assert len(left_constraint) >= 5, "Left BC should have at least x_batch, 3 targets, and required_ujs"
    assert left_constraint[0].shape == (5, 2), "Left BC x_batch should be (5, 2)"

def test_loss_fn_returns_scalar(mock_problem):
    all_params, cons = mock_problem
    
    # Create mock processed constraints (as the framework would provide them)
    # The framework would normally process required_ujs and add derivatives
    x_batch_phys = cons[0][0]  # 100 x 2 points
    n_points = x_batch_phys.shape[0]
    
    # Create mock derivatives (all zeros for simplicity)
    mock_derivatives = [jnp.zeros((n_points, 1)) for _ in range(12)]
    
    # Create mock boundary constraints with proper structure
    mock_boundary_cons = []
    for i in range(1, 5):  # 4 boundary constraints
        x_batch = cons[i][0]
        n_pts = x_batch.shape[0]
        # Add mock derivatives for each boundary constraint
        mock_boundary_cons.append([x_batch] + [jnp.zeros((n_pts, 1)) for _ in range(10)])
    
    # Create properly structured constraints for loss function
    processed_constraints = [
        [x_batch_phys] + mock_derivatives,  # Interior
        *mock_boundary_cons  # Boundaries
    ]
    
    loss = BiotCoupled2D.loss_fn(all_params, processed_constraints)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == (), "Loss must be a scalar"

def test_mini_training_run():
    trainer = BiotCoupledTrainer()
     # Pre train mechanics only
    trainer.auto_balance = False
    trainer.train_mechanics_only(n_steps=10)
     # Pre train flow only
    trainer.train_flow_only(n_steps=10)
     # Coupled training
    trainer.auto_balance = True
    params = trainer.train_coupled(n_steps=20)
    assert params is not None, "Trainer must return parameters"

def test_exact_solution_bcs():
    static, _ = BiotCoupled2D.init_params()
    all_params = {"static": {"problem": static}, "step": 0}
     # Points on left boundary x=0
    ys = jnp.linspace(0, 1, 20)
    pts = jnp.stack([jnp.zeros_like(ys), ys], axis=1)
    assert BiotCoupled2D.verify_bcs(all_params, pts), "Exact solution must satisfy left BC"
