"""
Shared visualization tools for Biot poroelasticity validation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
from datetime import datetime

class BiotVisualizationTools:
    """Shared visualization utilities for all Biot trainers"""
    
    def __init__(self):
        self.style_setup()
    
    def style_setup(self):
        """Set up consistent plotting style"""
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def create_mesh_grid(self, nx=50, ny=50, domain=((0, 1), (0, 1))):
        """Create uniform mesh grid for visualization"""
        x_min, x_max = domain[0]
        y_min, y_max = domain[1]
        
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        
        # Flatten for prediction
        x_flat = X.flatten()
        y_flat = Y.flatten()
        points = np.column_stack([x_flat, y_flat])
        
        return X, Y, points
    
    def plot_solution_fields(self, trainer, save_path=None, title_prefix="", domain=((0, 1), (0, 1))):
        """
        Plot displacement and pressure fields for any Biot trainer
        
        Args:
            trainer: Any Biot trainer with predict() method
            save_path: Optional path to save figure
            title_prefix: Prefix for plot title (e.g., "2D Physics Only")
            domain: Domain bounds for plotting
        """
        # Create mesh
        X, Y, points = self.create_mesh_grid(domain=domain)
        
        try:
            # Handle JAX/NumPy compatibility
            try:
                import jax.numpy as jnp
                points_jax = jnp.array(points)
            except ImportError:
                points_jax = points
            
            # Get predictions
            pred = trainer.predict(points_jax)
            
            # Get exact solution if available
            exact = None
            if hasattr(trainer, 'trainer') and hasattr(trainer.trainer.c, 'problem'):
                try:
                    exact = trainer.trainer.c.problem.exact_solution(trainer.all_params, points_jax)
                except:
                    pass
            
            # Convert to numpy for plotting
            if hasattr(pred, 'numpy'):
                pred = pred.numpy()
            elif hasattr(pred, '__array__'):
                pred = np.array(pred)
            
            if exact is not None:
                if hasattr(exact, 'numpy'):
                    exact = exact.numpy()
                elif hasattr(exact, '__array__'):
                    exact = np.array(exact)
            
            # Reshape for plotting
            ux_pred = pred[:, 0].reshape(X.shape)
            uy_pred = pred[:, 1].reshape(X.shape)
            p_pred = pred[:, 2].reshape(X.shape)
            
            # Create plots
            if exact is not None:
                # Show predicted vs exact
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle(f'{title_prefix} Biot Poroelasticity: Predicted vs Exact', fontsize=16)
                
                ux_exact = exact[:, 0].reshape(X.shape)
                uy_exact = exact[:, 1].reshape(X.shape)
                p_exact = exact[:, 2].reshape(X.shape)
                
                # Predicted results
                im1 = axes[0, 0].contourf(X, Y, ux_pred, levels=20, cmap='RdBu_r')
                axes[0, 0].set_title('$u_x$ (Predicted)')
                plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
                
                im2 = axes[0, 1].contourf(X, Y, uy_pred, levels=20, cmap='RdBu_r')
                axes[0, 1].set_title('$u_y$ (Predicted)')
                plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
                
                im3 = axes[0, 2].contourf(X, Y, p_pred, levels=20, cmap='viridis')
                axes[0, 2].set_title('Pressure $p$ (Predicted)')
                plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
                
                # Exact results
                im4 = axes[1, 0].contourf(X, Y, ux_exact, levels=20, cmap='RdBu_r')
                axes[1, 0].set_title('$u_x$ (Exact)')
                plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
                
                im5 = axes[1, 1].contourf(X, Y, uy_exact, levels=20, cmap='RdBu_r')
                axes[1, 1].set_title('$u_y$ (Exact)')
                plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
                
                im6 = axes[1, 2].contourf(X, Y, p_exact, levels=20, cmap='viridis')
                axes[1, 2].set_title('Pressure $p$ (Exact)')
                plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
                
            else:
                # Show predicted results only
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f'{title_prefix} Biot Poroelasticity Results', fontsize=16)
                
                im1 = axes[0].contourf(X, Y, ux_pred, levels=20, cmap='RdBu_r')
                axes[0].set_title('Horizontal Displacement $u_x$')
                plt.colorbar(im1, ax=axes[0])
                
                im2 = axes[1].contourf(X, Y, uy_pred, levels=20, cmap='RdBu_r')
                axes[1].set_title('Vertical Displacement $u_y$')
                plt.colorbar(im2, ax=axes[1])
                
                im3 = axes[2].contourf(X, Y, p_pred, levels=20, cmap='viridis')
                axes[2].set_title('Pressure $p$')
                plt.colorbar(im3, ax=axes[2])
            
            # Add coordinate labels
            for ax in axes.flat:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f" Solution fields saved to: {save_path}")
            
            plt.show()
            return fig
            
        except Exception as e:
            print(f" Error plotting solution fields: {e}")
            return None
    
    def plot_error_fields(self, trainer, save_path=None, title_prefix="", domain=((0, 1), (0, 1))):
        """Plot error fields between predicted and exact solutions"""
        # Create mesh
        X, Y, points = self.create_mesh_grid(domain=domain)
        
        try:
            # Get predictions and exact solution
            try:
                import jax.numpy as jnp
                points_jax = jnp.array(points)
            except ImportError:
                points_jax = points
                
            pred = trainer.predict(points_jax)
            
            if not hasattr(trainer, 'trainer') or not hasattr(trainer.trainer.c, 'problem'):
                print(" No exact solution available for error computation")
                return None
                
            exact = trainer.trainer.c.problem.exact_solution(trainer.all_params, points_jax)
            
            # Convert to numpy
            if hasattr(pred, 'numpy'):
                pred = pred.numpy()
            if hasattr(exact, 'numpy'):
                exact = exact.numpy()
            
            # Calculate errors
            error_ux = np.abs(pred[:, 0] - exact[:, 0]).reshape(X.shape)
            error_uy = np.abs(pred[:, 1] - exact[:, 1]).reshape(X.shape)
            error_p = np.abs(pred[:, 2] - exact[:, 2]).reshape(X.shape)
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'{title_prefix} Prediction Errors: |Predicted - Exact|', fontsize=16)
            
            # Plot errors
            im1 = axes[0].contourf(X, Y, error_ux, levels=20, cmap='Reds')
            axes[0].set_title(f'|Error $u_x$| (Max: {np.max(error_ux):.2e})')
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].contourf(X, Y, error_uy, levels=20, cmap='Reds')
            axes[1].set_title(f'|Error $u_y$| (Max: {np.max(error_uy):.2e})')
            plt.colorbar(im2, ax=axes[1])
            
            im3 = axes[2].contourf(X, Y, error_p, levels=20, cmap='Reds')
            axes[2].set_title(f'|Error $p$| (Max: {np.max(error_p):.2e})')
            plt.colorbar(im3, ax=axes[2])
            
            for ax in axes:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f" Error fields saved to: {save_path}")
            
            plt.show()
            return fig
            
        except Exception as e:
            print(f" Error plotting error fields: {e}")
            return None

    def save_training_summary(self, trainer, metrics, save_dir, model_name):
        """Save comprehensive training summary"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = save_dir / f"{model_name}_metrics.json"
        
        # Convert numpy types to regular Python types for JSON
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                json_metrics[key] = {k: float(v) for k, v in value.items()}
            else:
                json_metrics[key] = float(value)
        
        # Add metadata
        json_metrics['timestamp'] = datetime.now().isoformat()
        json_metrics['model_type'] = model_name
        
        with open(metrics_file, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"Metrics saved to: {metrics_file}")
        
        # Save text summary
        summary_file = save_dir / f"{model_name}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"BIOT POROELASTICITY VALIDATION REPORT\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("L2 Errors:\n")
            if 'L2_errors' in metrics:
                for key, value in metrics['L2_errors'].items():
                    f.write(f"  {key}: {value:.2e}\n")
            
            f.write("\nL∞ Errors:\n")
            if 'Linf_errors' in metrics:
                for key, value in metrics['Linf_errors'].items():
                    f.write(f"  {key}: {value:.2e}\n")
            
            f.write("\nRelative Errors:\n")
            if 'Relative_errors' in metrics:
                for key, value in metrics['Relative_errors'].items():
                    f.write(f"  {key}: {value:.2e}\n")
        
        print(f"Summary saved to: {summary_file}")
        
        return metrics_file, summary_file
