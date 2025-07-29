import os
import sys
import pytest
import jax
import jax.numpy as jnp
from fbpinns.domains import RectangularDomainND
from poroelasticity.biot_trainer import BiotCoupled2D, CoupledTrainer  

@pytest.fixture
def mock_problem():
    # Initialise static parameters and dummy state
    static, _ = BiotCoupled2D.init_params()
    all_params = {"static": {"problem": static}, "step": 0}
    # Create domain and sampler key
    dom = RectangularDomainND()
    key = jax.random.PRNGKey(0)
    # Define batch shapes: 10 interior points, 5 per boundary
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
     # Interior: x_batch + 12 derivatives
    x_phys, req = cons[0]
    assert isinstance(x_phys, jnp.ndarray)
    assert x_phys.shape == (10, 2)
    assert len(req) == 12, "Interior must request 12 derivatives"
     # Left BC: x + 3 targets + 3 fields
    left = cons[1]
    assert len(left) == 1 + 3 + 3

def test_loss_fn_returns_scalar(mock_problem):
    all_params, cons = mock_problem
    loss = BiotCoupled2D.loss_fn(all_params, cons)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == (), "Loss must be a scalar"

def test_mini_training_run():
    trainer = CoupledTrainer()
     # Pre-train mechanics only
    trainer.auto_balance = False
    trainer.train_mechanics_only(n_steps=10)
     # Pre-train flow only
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
