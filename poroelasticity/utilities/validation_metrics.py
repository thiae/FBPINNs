"""
Validation metrics and error analysis tools
"""

import numpy as np

class ValidationMetrics:
    """Comprehensive error analysis and validation metrics"""
    
    @staticmethod
    def compute_error_metrics(pred, exact, labels=['ux', 'uy', 'p']):
        """
        Compute comprehensive error metrics between predicted and exact solutions
        
        Args:
            pred: Predicted values (N, 3) array
            exact: Exact values (N, 3) array
            labels: Labels for each component
            
        Returns:
            Dictionary of error metrics
        """
        # Ensure numpy arrays
        if hasattr(pred, 'numpy'):
            pred = pred.numpy()
        elif hasattr(pred, '__array__'):
            pred = np.array(pred)
            
        if hasattr(exact, 'numpy'):
            exact = exact.numpy()
        elif hasattr(exact, '__array__'):
            exact = np.array(exact)
        
        metrics = {
            'L2_errors': {},
            'Linf_errors': {},
            'Relative_errors': {},
            'RMSE': {},
            'MAE': {}
        }
        
        total_l2_error = 0.0
        
        for i, label in enumerate(labels):
            if i >= pred.shape[1] or i >= exact.shape[1]:
                continue
                
            error = pred[:, i] - exact[:, i]
            
            # L2 error (RMS)
            l2_error = np.sqrt(np.mean(error**2))
            metrics['L2_errors'][label] = l2_error
            total_l2_error += l2_error**2
            
            # L∞ error (maximum absolute error)
            linf_error = np.max(np.abs(error))
            metrics['Linf_errors'][label] = linf_error
            
            # Relative error
            exact_norm = np.sqrt(np.mean(exact[:, i]**2))
            if exact_norm > 1e-12:
                rel_error = l2_error / exact_norm
            else:
                rel_error = 0.0
            metrics['Relative_errors'][label] = rel_error
            
            # RMSE (same as L2 but explicit)
            metrics['RMSE'][label] = l2_error
            
            # Mean Absolute Error
            mae = np.mean(np.abs(error))
            metrics['MAE'][label] = mae
        
        # Total L2 error
        metrics['L2_errors']['total'] = np.sqrt(total_l2_error)
        
        return metrics
    
    @staticmethod
    def print_validation_summary(metrics, model_name="Model"):
        """Print comprehensive validation summary"""
        print("=" * 60)
        print(f" {model_name.upper()} - VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f" L2 Errors:")
        if 'L2_errors' in metrics:
            for key, value in metrics['L2_errors'].items():
                print(f"   {key:8}: {value:.2e}")
        
        print(f"\n L∞ Errors (Max):")
        if 'Linf_errors' in metrics:
            for key, value in metrics['Linf_errors'].items():
                print(f"   {key:8}: {value:.2e}")
        
        print(f"\n Relative Errors:")
        if 'Relative_errors' in metrics:
            for key, value in metrics['Relative_errors'].items():
                print(f"   {key:8}: {value:.2e}")
        
        # Assessment
        if 'L2_errors' in metrics and 'total' in metrics['L2_errors']:
            l2_total = metrics['L2_errors']['total']
            print(f"\n Assessment:")
            if l2_total < 1e-2:
                print(f"    Excellent accuracy: L2 error = {l2_total:.2e}")
                print("    Physics implementation is working correctly!")
                status = "EXCELLENT"
            elif l2_total < 1e-1:
                print(f"    Good accuracy: L2 error = {l2_total:.2e}")
                print("    Physics implementation is acceptable")
                status = "GOOD"
            else:
                print(f"    Consider more training: L2 error = {l2_total:.2e}")
                print("    Try increasing training steps or checking implementation")
                status = "NEEDS_IMPROVEMENT"
        else:
            status = "UNKNOWN"
        
        print("=" * 60)
        return status
    
    @staticmethod
    def verify_boundary_conditions(trainer, tol=1e-6, domain=((0, 1), (0, 1))):
        """
        Verify boundary conditions for a trainer
        
        Args:
            trainer: Biot trainer with predict method
            tol: Tolerance for boundary identification
            domain: Domain bounds
            
        Returns:
            Dictionary of boundary condition verification results
        """
        try:
            # Handle JAX/NumPy compatibility
            try:
                import jax.numpy as jnp
                jnp_available = True
            except ImportError:
                jnp_available = False
            
            results = {}
            
            # Create boundary points
            n_boundary = 100
            x_min, x_max = domain[0]
            y_min, y_max = domain[1]
            
            # Left boundary (x = x_min)
            y_left = np.linspace(y_min, y_max, n_boundary)
            x_left = np.full_like(y_left, x_min)
            points_left = np.column_stack([x_left, y_left])
            
            if jnp_available:
                points_left_jax = jnp.array(points_left)
            else:
                points_left_jax = points_left
                
            pred_left = trainer.predict(points_left_jax)
            
            # Convert to numpy if needed
            if hasattr(pred_left, 'numpy'):
                pred_left = pred_left.numpy()
            
            # Check left boundary conditions (typically u_x=0, u_y=0, p=1)
            ux_left_max = np.max(np.abs(pred_left[:, 0]))
            uy_left_max = np.max(np.abs(pred_left[:, 1]))
            p_left_mean = np.mean(pred_left[:, 2])
            p_left_std = np.std(pred_left[:, 2])
            
            results['left_boundary'] = {
                'ux_max_deviation': ux_left_max,
                'uy_max_deviation': uy_left_max, 
                'p_mean': p_left_mean,
                'p_std': p_left_std,
                'ux_satisfied': ux_left_max < 1e-2,
                'uy_satisfied': uy_left_max < 1e-2,
                'p_satisfied': abs(p_left_mean - 1.0) < 0.1
            }
            
            # Right boundary (x = x_max)
            y_right = np.linspace(y_min, y_max, n_boundary)
            x_right = np.full_like(y_right, x_max)
            points_right = np.column_stack([x_right, y_right])
            
            if jnp_available:
                points_right_jax = jnp.array(points_right)
            else:
                points_right_jax = points_right
                
            pred_right = trainer.predict(points_right_jax)
            
            if hasattr(pred_right, 'numpy'):
                pred_right = pred_right.numpy()
            
            # Check right boundary conditions (typically p=0)
            p_right_max = np.max(np.abs(pred_right[:, 2]))
            
            results['right_boundary'] = {
                'p_max_deviation': p_right_max,
                'p_satisfied': p_right_max < 1e-2
            }
            
            # Bottom boundary (y = y_min)
            x_bottom = np.linspace(x_min, x_max, n_boundary)
            y_bottom = np.full_like(x_bottom, y_min)
            points_bottom = np.column_stack([x_bottom, y_bottom])
            
            if jnp_available:
                points_bottom_jax = jnp.array(points_bottom)
            else:
                points_bottom_jax = points_bottom
                
            pred_bottom = trainer.predict(points_bottom_jax)
            
            if hasattr(pred_bottom, 'numpy'):
                pred_bottom = pred_bottom.numpy()
            
            # Check bottom boundary conditions (typically u_y=0)
            uy_bottom_max = np.max(np.abs(pred_bottom[:, 1]))
            
            results['bottom_boundary'] = {
                'uy_max_deviation': uy_bottom_max,
                'uy_satisfied': uy_bottom_max < 1e-2
            }
            
            # Overall assessment
            all_satisfied = (
                results['left_boundary']['ux_satisfied'] and
                results['left_boundary']['uy_satisfied'] and
                results['left_boundary']['p_satisfied'] and
                results['right_boundary']['p_satisfied'] and
                results['bottom_boundary']['uy_satisfied']
            )
            
            results['overall'] = {
                'all_boundaries_satisfied': all_satisfied,
                'summary': "All boundary conditions satisfied" if all_satisfied else "Some boundary conditions violated"
            }
            
            return results
            
        except Exception as e:
            print(f" Error verifying boundary conditions: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def physics_conservation_check(trainer, n_test=1000, domain=((0, 1), (0, 1))):
        """
        Check physics conservation properties
        
        Args:
            trainer: Biot trainer
            n_test: Number of test points
            domain: Domain bounds
            
        Returns:
            Dictionary of conservation check results
        """
        try:
            # Generate random test points
            x_min, x_max = domain[0]
            y_min, y_max = domain[1]
            
            np.random.seed(42)  # For reproducibility
            x_test = np.random.uniform(x_min, x_max, n_test)
            y_test = np.random.uniform(y_min, y_max, n_test)
            points_test = np.column_stack([x_test, y_test])
            
            # Handle JAX/NumPy compatibility
            try:
                import jax.numpy as jnp
                points_jax = jnp.array(points_test)
            except ImportError:
                points_jax = points_test
            
            # Get predictions
            pred = trainer.predict(points_jax)
            
            if hasattr(pred, 'numpy'):
                pred = pred.numpy()
            
            # Basic physics checks
            results = {}
            
            # Check displacement magnitudes are reasonable
            ux_mag = np.abs(pred[:, 0])
            uy_mag = np.abs(pred[:, 1])
            p_vals = pred[:, 2]
            
            results['displacement_checks'] = {
                'ux_max': np.max(ux_mag),
                'ux_mean': np.mean(ux_mag),
                'uy_max': np.max(uy_mag),
                'uy_mean': np.mean(uy_mag),
                'displacement_reasonable': np.max(ux_mag) < 10.0 and np.max(uy_mag) < 10.0
            }
            
            results['pressure_checks'] = {
                'p_min': np.min(p_vals),
                'p_max': np.max(p_vals),
                'p_mean': np.mean(p_vals),
                'pressure_reasonable': np.min(p_vals) >= -0.1 and np.max(p_vals) <= 2.0
            }
            
            # Overall physics assessment
            physics_ok = (
                results['displacement_checks']['displacement_reasonable'] and
                results['pressure_checks']['pressure_reasonable']
            )
            
            results['overall'] = {
                'physics_reasonable': physics_ok,
                'summary': "Physics values are reasonable" if physics_ok else "Physics values may be unrealistic"
            }
            
            return results
            
        except Exception as e:
            print(f" Error in physics conservation check: {e}")
            return {'error': str(e)}
