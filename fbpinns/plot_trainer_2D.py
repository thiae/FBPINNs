# """
# Defines plotting functions for 2D FBPINN / PINN problems

# This module is used by plot_trainer.py (and subsequently trainers.py)
# """

# import matplotlib.pyplot as plt

# from fbpinns.plot_trainer_1D import _plot_setup, _to_numpy

# def _plot_test_im(u_test, xlim, ulim, n_test, it=None):
#     u_test = u_test.reshape(n_test)
#     if it is not None:
#         u_test = u_test[:,:,it]# for 3D
#     plt.imshow(u_test.T,# transpose as jnp.meshgrid uses indexing="ij"
#                origin="lower", extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
#                cmap="viridis", vmin=ulim[0], vmax=ulim[1])
#     plt.colorbar()
#     plt.xlim(xlim[0][0], xlim[1][0])
#     plt.ylim(xlim[0][1], xlim[1][1])
#     plt.gca().set_aspect("equal")

# @_to_numpy
# def plot_2D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test):

#     xlim, ulim = _plot_setup(x_batch_test, u_exact)
#     xlim0 = x_batch_test.min(0), x_batch_test.max(0)

#     f = plt.figure(figsize=(8,10))

#     # plot domain + x_batch
#     plt.subplot(3,2,1)
#     plt.title(f"[{i}] Domain decomposition")
#     plt.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
#     decomposition.plot(all_params, active=active, create_fig=False)
#     plt.xlim(xlim[0][0], xlim[1][0])
#     plt.ylim(xlim[0][1], xlim[1][1])
#     plt.gca().set_aspect("equal")

#     # plot full solutions
#     plt.subplot(3,2,2)
#     plt.title(f"[{i}] Difference")
#     _plot_test_im(u_exact - u_test, xlim0, ulim, n_test)

#     plt.subplot(3,2,3)
#     plt.title(f"[{i}] Full solution")
#     _plot_test_im(u_test, xlim0, ulim, n_test)

#     plt.subplot(3,2,4)
#     plt.title(f"[{i}] Ground truth")
#     _plot_test_im(u_exact, xlim0, ulim, n_test)

#     # plot raw hist
#     plt.subplot(3,2,5)
#     plt.title(f"[{i}] Raw solutions")
#     plt.hist(us_raw_test.flatten(), bins=100, label=f"{us_raw_test.min():.1f}, {us_raw_test.max():.1f}")
#     plt.legend(loc=1)
#     plt.xlim(-5,5)

#     plt.tight_layout()

#     return (("test",f),)

# @_to_numpy
# def plot_2D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):

#     xlim, ulim = _plot_setup(x_batch_test, u_exact)
#     xlim0 = x_batch.min(0), x_batch.max(0)

#     f = plt.figure(figsize=(8,10))

#     # plot x_batch
#     plt.subplot(3,2,1)
#     plt.title(f"[{i}] Training points")
#     plt.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
#     plt.xlim(xlim[0][0], xlim[1][0])
#     plt.ylim(xlim[0][1], xlim[1][1])
#     plt.gca().set_aspect("equal")

#     # plot full solution
#     plt.subplot(3,2,2)
#     plt.title(f"[{i}] Difference")
#     _plot_test_im(u_exact - u_test, xlim0, ulim, n_test)

#     plt.subplot(3,2,3)
#     plt.title(f"[{i}] Full solution")
#     _plot_test_im(u_test, xlim0, ulim, n_test)

#     plt.subplot(3,2,4)
#     plt.title(f"[{i}] Ground truth")
#     _plot_test_im(u_exact, xlim0, ulim, n_test)

#     # plot raw hist
#     plt.subplot(3,2,5)
#     plt.title(f"[{i}] Raw solution")
#     plt.hist(u_raw_test.flatten(), bins=100, label=f"{u_raw_test.min():.1f}, {u_raw_test.max():.1f}")
#     plt.legend(loc=1)
#     plt.xlim(-5,5)

#     plt.tight_layout()

#     return (("test",f),)









"""
Defines plotting functions for 2D FBPINN / PINN problems

This module is used by plot_trainer.py (and subsequently trainers.py)

FIXED VERSION: Now supports multi-output PINNs (e.g., ux, uy for geomechanics)
"""

import matplotlib.pyplot as plt
import numpy as np

from fbpinns.plot_trainer_1D import _plot_setup, _to_numpy

def _plot_test_im(u_test, xlim, ulim, n_test, it=None, output_idx=0):
    """
    Plot test image with support for multi-output PINNs
    
    Args:
        u_test: Test data - can be (n_points,) or (n_points, n_outputs)
        xlim: x limits
        ulim: u limits  
        n_test: Test grid shape (nx, ny)
        it: Time index (for 3D problems)
        output_idx: Which output component to plot (0 for ux, 1 for uy, etc.)
    """
    
    # Handle multi-output case
    if u_test.ndim == 2 and u_test.shape[1] > 1:
        # Multi-output: select the specified component
        if output_idx < u_test.shape[1]:
            u_plot = u_test[:, output_idx]
        else:
            # Default to first component if index out of bounds
            u_plot = u_test[:, 0]
            print(f"Warning: output_idx {output_idx} out of bounds, using component 0")
    else:
        # Single output or already selected component
        u_plot = u_test.flatten()
    
    # Reshape to grid
    try:
        u_plot = u_plot.reshape(n_test)
        if it is not None:
            u_plot = u_plot[:,:,it]  # for 3D
            
        plt.imshow(u_plot.T,  # transpose as jnp.meshgrid uses indexing="ij"
                   origin="lower", extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
                   cmap="viridis", vmin=ulim[0], vmax=ulim[1])
        plt.colorbar()
        plt.xlim(xlim[0][0], xlim[1][0])
        plt.ylim(xlim[0][1], xlim[1][1])
        plt.gca().set_aspect("equal")
        
    except ValueError as e:
        print(f"Warning: Could not reshape data for plotting. Shape: {u_test.shape}, Expected: {n_test}")
        print(f"Error: {e}")
        # Create a placeholder plot
        plt.text(0.5, 0.5, f"Plot Error\nData shape: {u_test.shape}\nExpected: {n_test}", 
                 transform=plt.gca().transAxes, ha='center', va='center')

def _get_component_limits(u_data, component_idx=0):
    """Get limits for a specific component of multi-output data"""
    if u_data.ndim == 2 and u_data.shape[1] > 1:
        if component_idx < u_data.shape[1]:
            component_data = u_data[:, component_idx]
        else:
            component_data = u_data[:, 0]
    else:
        component_data = u_data.flatten()
    
    return component_data.min(), component_data.max()

@_to_numpy
def plot_2D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test):

    xlim, _ = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch_test.min(0), x_batch_test.max(0)
    
    # Determine number of output components
    if u_exact is not None and u_exact.ndim == 2 and u_exact.shape[1] > 1:
        n_outputs = u_exact.shape[1]
        output_names = [f"u{i}" for i in range(n_outputs)]
        if n_outputs == 2:
            output_names = ["ux", "uy"]  # Common for geomechanics
        elif n_outputs == 3:
            output_names = ["ux", "uy", "uz"]
    elif u_test.ndim == 2 and u_test.shape[1] > 1:
        # No exact solution - infer from test data
        n_outputs = u_test.shape[1]
        output_names = [f"u{i}" for i in range(n_outputs)]
        if n_outputs == 2:
            output_names = ["ux", "uy"]
        elif n_outputs == 3:
            output_names = ["ux", "uy", "p"]  # Common for your poroelasticity problem
    else:
        n_outputs = 1
        output_names = ["u"]

    # Create figure with enough subplots for all components
    fig_height = max(10, 3 * n_outputs + 4)  # Scale figure height with number of outputs
    f = plt.figure(figsize=(12, fig_height))
    
    plot_idx = 1
    
    # Plot domain + x_batch (always first)
    plt.subplot(n_outputs + 2, 3, plot_idx)
    plt.title(f"[{i}] Domain decomposition")
    plt.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
    decomposition.plot(all_params, active=active, create_fig=False)
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")
    plot_idx += 1
    
    # Plot raw histogram (second plot)
    plt.subplot(n_outputs + 2, 3, plot_idx)
    plt.title(f"[{i}] Raw solutions")
    plt.hist(us_raw_test.flatten(), bins=100, label=f"{us_raw_test.min():.1f}, {us_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5, 5)
    plot_idx += 1
    
    # Skip third position to start new row
    plot_idx += 1
    
    # Plot each component
    for comp_idx in range(n_outputs):
        comp_name = output_names[comp_idx]
        
        # Get limits for this component
        ulim_test = _get_component_limits(u_test, comp_idx)
        if u_exact is not None:
            ulim_exact = _get_component_limits(u_exact, comp_idx)
            ulim = (min(ulim_exact[0], ulim_test[0]), max(ulim_exact[1], ulim_test[1]))
        else:
            ulim = ulim_test
        
        # FBPINN solution
        plt.subplot(n_outputs + 2, 3, plot_idx)
        plt.title(f"[{i}] {comp_name} - FBPINN")
        _plot_test_im(u_test, xlim0, ulim, n_test, output_idx=comp_idx)
        plot_idx += 1
        
        if u_exact is not None:
            # Ground truth (only if exact solution exists)
            plt.subplot(n_outputs + 2, 3, plot_idx)
            plt.title(f"[{i}] {comp_name} - Exact")
            _plot_test_im(u_exact, xlim0, ulim, n_test, output_idx=comp_idx)
            plot_idx += 1
            
            # Difference (only if exact solution exists)
            plt.subplot(n_outputs + 2, 3, plot_idx)
            plt.title(f"[{i}] {comp_name} - Error")
            if u_exact.ndim == 2 and u_test.ndim == 2:
                diff = u_exact[:, comp_idx:comp_idx+1] - u_test[:, comp_idx:comp_idx+1]
            else:
                diff = u_exact - u_test
            diff_lim = np.abs(diff).max()
            _plot_test_im(diff, xlim0, (-diff_lim, diff_lim), n_test, output_idx=0)
            plot_idx += 1
        else:
            # Physics-only training - skip exact solution and error plots
            plt.subplot(n_outputs + 2, 3, plot_idx)
            plt.title(f"[{i}] {comp_name} - Physics Only")
            plt.text(0.5, 0.5, "No exact solution\n(Physics-only training)", 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plot_idx += 2  # Skip both exact and error plots

    plt.tight_layout()
    return (("test", f),)

@_to_numpy
def plot_2D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):

    xlim, _ = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch.min(0), x_batch.max(0)
    
    # Determine number of output components
    if u_exact is not None and u_exact.ndim == 2 and u_exact.shape[1] > 1:
        n_outputs = u_exact.shape[1]
        output_names = [f"u{i}" for i in range(n_outputs)]
        if n_outputs == 2:
            output_names = ["ux", "uy"]
        elif n_outputs == 3:
            output_names = ["ux", "uy", "uz"]
    elif u_test.ndim == 2 and u_test.shape[1] > 1:
        # No exact solution - infer from test data
        n_outputs = u_test.shape[1]
        output_names = [f"u{i}" for i in range(n_outputs)]
        if n_outputs == 2:
            output_names = ["ux", "uy"]
        elif n_outputs == 3:
            output_names = ["ux", "uy", "p"]  # Common for your poroelasticity problem
    else:
        n_outputs = 1
        output_names = ["u"]

    # Create figure
    fig_height = max(10, 3 * n_outputs + 4)
    f = plt.figure(figsize=(12, fig_height))
    
    plot_idx = 1
    
    # Plot training points
    plt.subplot(n_outputs + 2, 3, plot_idx)
    plt.title(f"[{i}] Training points")
    plt.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")
    plot_idx += 1
    
    # Plot raw histogram
    plt.subplot(n_outputs + 2, 3, plot_idx)
    plt.title(f"[{i}] Raw solution")
    plt.hist(u_raw_test.flatten(), bins=100, label=f"{u_raw_test.min():.1f}, {u_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5, 5)
    plot_idx += 1
    
    # Skip third position
    plot_idx += 1
    
    # Plot each component
    for comp_idx in range(n_outputs):
        comp_name = output_names[comp_idx]
        
        # Get limits for this component
        ulim_test = _get_component_limits(u_test, comp_idx)
        if u_exact is not None:
            ulim_exact = _get_component_limits(u_exact, comp_idx)
            ulim = (min(ulim_exact[0], ulim_test[0]), max(ulim_exact[1], ulim_test[1]))
        else:
            ulim = ulim_test
        
        # PINN solution
        plt.subplot(n_outputs + 2, 3, plot_idx)
        plt.title(f"[{i}] {comp_name} - PINN")
        _plot_test_im(u_test, xlim0, ulim, n_test, output_idx=comp_idx)
        plot_idx += 1
        
        if u_exact is not None:
            # Ground truth (only if exact solution exists)
            plt.subplot(n_outputs + 2, 3, plot_idx)
            plt.title(f"[{i}] {comp_name} - Exact")
            _plot_test_im(u_exact, xlim0, ulim, n_test, output_idx=comp_idx)
            plot_idx += 1
            
            # Difference (only if exact solution exists)
            plt.subplot(n_outputs + 2, 3, plot_idx)
            plt.title(f"[{i}] {comp_name} - Error")
            if u_exact.ndim == 2 and u_test.ndim == 2:
                diff = u_exact[:, comp_idx:comp_idx+1] - u_test[:, comp_idx:comp_idx+1]
            else:
                diff = u_exact - u_test
            diff_lim = np.abs(diff).max()
            _plot_test_im(diff, xlim0, (-diff_lim, diff_lim), n_test, output_idx=0)
            plot_idx += 1
        else:
            # Physics-only training - skip exact solution and error plots
            plt.subplot(n_outputs + 2, 3, plot_idx)
            plt.title(f"[{i}] {comp_name} - Physics Only")
            plt.text(0.5, 0.5, "No exact solution\n(Physics-only training)", 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plot_idx += 2  # Skip both exact and error plots

    plt.tight_layout()
    return (("test", f),)





