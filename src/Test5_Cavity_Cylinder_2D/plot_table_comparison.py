import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.ticker as ticker
import os
import scipy.interpolate as interp
import jax.numpy as jnp
import sys
sys.path.insert(0, '../')

from cavity_cylinder_spikan import CavityCylinder_SF_KAN_Separable, create_interior_mask
from cavity_cylinder_pikan import CavityCylinder_SF_KAN

# Configure global font settings
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
})

# Define data files for all k values
data_files_by_k = {
    3: {
        'pikan': './data/cavity_cylinder_pikan_Re_100.0_nx100_ny100_epochs200000_k3.npy',
        'spikan': {
            100: './data/cavity_cylinder_spikan_Re_100.0_nx100_ny100_epochs200000_r10_k3_silu.npy',
            400: './data/cavity_cylinder_spikan_Re_100.0_nx400_ny400_epochs200000_r10_k3_silu.npy',
            800: './data/cavity_cylinder_spikan_Re_100.0_nx800_ny800_epochs200000_r10_k3_silu.npy'
        }
    },
    4: {
        'pikan': './data/cavity_cylinder_pikan_Re_100.0_nx100_ny100_epochs200000_k4.npy',
        'spikan': {
            100: './data/cavity_cylinder_spikan_Re_100.0_nx100_ny100_epochs200000_r10_k4_silu.npy',
            400: './data/cavity_cylinder_spikan_Re_100.0_nx400_ny400_epochs200000_r10_k4_silu.npy',
            800: './data/cavity_cylinder_spikan_Re_100.0_nx800_ny800_epochs200000_r10_k4_silu.npy'
        }
    },
    5: {
        'pikan': './data/cavity_cylinder_pikan_Re_100.0_nx100_ny100_epochs200000_k5.npy',
        'spikan': {
            100: './data/cavity_cylinder_spikan_Re_100.0_nx100_ny100_epochs200000_r10_k5_silu.npy',
            400: './data/cavity_cylinder_spikan_Re_100.0_nx400_ny400_epochs200000_r10_k5_silu.npy',
            800: './data/cavity_cylinder_spikan_Re_100.0_nx800_ny800_epochs200000_r10_k5_silu.npy'
        }
    }
}

# Define data files for different activation functions (k=5, nx=800)
activation_files = {
    'silu': './data/cavity_cylinder_spikan_Re_100.0_nx800_ny800_epochs200000_r10_k5_silu.npy',
    'tanh': './data/cavity_cylinder_spikan_Re_100.0_nx800_ny800_epochs200000_r10_k5_tanh.npy',
    'relu': './data/cavity_cylinder_spikan_Re_100.0_nx800_ny800_epochs200000_r10_k5_relu.npy',
    'sine': './data/cavity_cylinder_spikan_Re_100.0_nx800_ny800_epochs200000_r10_k5_sine.npy'
}

# Load OpenFOAM reference data
openfoam_file = './data/reference_data/openfoam_cavity_cylinder_Re100.npz'

# Check if reference file exists
if not os.path.exists(openfoam_file):
    print(f"Error: OpenFOAM reference file not found: {openfoam_file}")
    exit(1)

print("Loading reference data...")

# Load reference data
ref_data = np.load(openfoam_file, allow_pickle=True)
X_ref = ref_data['X_mesh']
Y_ref = ref_data['Y_mesh']
U_ref = ref_data['U']
V_ref = ref_data['V']
P_ref = ref_data['P']

# Extract cylinder parameters from reference or set defaults
Re = 100.0
cylinder_center = (0.5, 0.5)
cylinder_radius = 0.2
L_range = (0.0, 1.0)
H_range = (0.0, 1.0)

# Function to load and evaluate model predictions on high-res grid
def load_and_evaluate_model(data_file, model_type='spikan', nx_eval=200, ny_eval=200):
    """Load model and evaluate on a uniform grid."""
    if not os.path.exists(data_file):
        print(f"Warning: Data file not found: {data_file}")
        return None, None, None, None, None
        
    data = np.load(data_file, allow_pickle=True).item()
    params = data['parameters']
    
    # Create evaluation grid
    x_eval = jnp.linspace(L_range[0], L_range[1], nx_eval)
    y_eval = jnp.linspace(H_range[0], H_range[1], ny_eval)
    
    if model_type == 'spikan':
        # Load SPIKAN model
        model = CavityCylinder_SF_KAN_Separable(
            layer_dims=params['layer_dims'],
            init_lr=1e-3,
            k=params['k'],
            r=params['r'],
            Re=params['Re'],
            cylinder_center=params['cylinder_center'],
            cylinder_radius=params['cylinder_radius']
        )
        model.variables_x = data['model']['variables_x']
        model.variables_y = data['model']['variables_y']
        
        # Predict
        uvp_pred = model.predict(x_eval, y_eval)
        
    else:  # pikan
        # Load PIKAN model
        model = CavityCylinder_SF_KAN(
            layer_dims=params['layer_dims'],
            init_lr=1e-3,
            k=params['k'],
            Re=params['Re'],
            cylinder_center=params['cylinder_center'],
            cylinder_radius=params['cylinder_radius']
        )
        model.variables = data['model']['variables']
        
        # Create grid for PIKAN evaluation
        X_mesh, Y_mesh = jnp.meshgrid(x_eval, y_eval)
        xy_eval = jnp.column_stack((X_mesh.ravel(), Y_mesh.ravel()))
        
        # Predict
        uvp_pred_flat, _ = model.forward_pass(model.variables, xy_eval)
        uvp_pred = uvp_pred_flat.reshape(ny_eval, nx_eval, 3)
        uvp_pred = uvp_pred.transpose(1, 0, 2)  # Make it (nx, ny, 3)
    
    # Extract fields
    U = np.array(uvp_pred[:, :, 0]).T
    V = np.array(uvp_pred[:, :, 1]).T
    P = np.array(uvp_pred[:, :, 2]).T
    
    # Apply mask
    mask = create_interior_mask(x_eval, y_eval, cylinder_center, cylinder_radius)
    U[~mask] = np.nan
    V[~mask] = np.nan
    P[~mask] = np.nan
    
    # Also return grid info and timing
    X_mesh, Y_mesh = jnp.meshgrid(x_eval, y_eval)
    X_mesh = np.array(X_mesh)
    Y_mesh = np.array(Y_mesh)
    
    # Get timing info if available
    ms_per_iter = data['training'].get('ms_per_iter', None)
    
    return U, V, P, X_mesh, Y_mesh, ms_per_iter

# Helper function to set velocity to zero inside cylinder
def zero_inside_cylinder(values, positions, center_pos, radius):
    """Set values to zero for points inside the cylinder."""
    values = values.copy()
    dist_from_center = np.abs(positions - center_pos)
    inside_cylinder = dist_from_center <= radius
    values[inside_cylinder] = 0.0
    return values

# Define colors and line styles
colors = {
    'ref': 'black',
    'pikan': 'red',
    100: 'blue',
    400: 'green',
    800: 'purple'
}

linestyles = {
    'ref': '-',
    'pikan': '--',
    100: '-.',
    400: '-.',
    800: '-.'
}

def create_centerline_plot_zoomed():
    """Create zoomed centerline comparison plot with all k values."""
    
    print("\nCreating zoomed centerline comparison plot...")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Load reference data once
    centerline_y = 0.5
    centerline_x = 0.5
    
    # Horizontal centerline (y=0.5) - v velocity (LEFT PLOT, x=0 to 0.3)
    centerline_idx_ref = np.argmin(np.abs(Y_ref[:, 0] - centerline_y))
    x_ref = X_ref[centerline_idx_ref, :]
    v_ref = V_ref[centerline_idx_ref, :].copy()
    
    # Set velocity to zero inside cylinder
    v_ref = zero_inside_cylinder(v_ref, x_ref, cylinder_center[0], cylinder_radius)
    
    # Filter for x=0 to 0.3
    mask_x_ref = (x_ref >= 0) & (x_ref <= 0.3) & ~np.isnan(v_ref)
    ax1.plot(x_ref[mask_x_ref], v_ref[mask_x_ref], 
             color='black', linestyle='-', linewidth=2.5,
             label='Reference (FVM)')
    
    # Vertical centerline (x=0.5) - u velocity (RIGHT PLOT, y=0.7 to 1.0)
    centerline_idx_ref = np.argmin(np.abs(X_ref[0, :] - centerline_x))
    y_ref = Y_ref[:, centerline_idx_ref]
    u_ref = U_ref[:, centerline_idx_ref].copy()
    
    # Set velocity to zero inside cylinder
    u_ref = zero_inside_cylinder(u_ref, y_ref, cylinder_center[1], cylinder_radius)
    
    # Filter for y=0.7 to 1.0
    mask_y_ref = (y_ref >= 0.7) & (y_ref <= 1.0) & ~np.isnan(u_ref)
    ax2.plot(u_ref[mask_y_ref], y_ref[mask_y_ref], 
             color='black', linestyle='-', linewidth=2.5,
             label='Reference (FVM)')
    
    # Define colors for methods and markers for k values
    method_colors = {'pikan': 'blue', 'spikan': 'green'}
    method_linestyles = {'pikan': '--', 'spikan': '-.'}
    k_markers = {3: 'o', 4: 's', 5: '^'}  # circle for k=3, square for k=4, triangle for k=5
    
    # Store line handles and labels for custom legend ordering
    lines_ax1 = []
    labels_ax1 = []
    lines_ax2 = []
    labels_ax2 = []
    
    # Add reference line to lists (already plotted above)
    lines_ax1.append(ax1.lines[-1])
    labels_ax1.append('Reference (FVM)')
    lines_ax2.append(ax2.lines[-1])
    labels_ax2.append('Reference (FVM)')
    
    # First pass: Plot all PIKAN data
    for k in [3, 4, 5]:
        pikan_file = data_files_by_k[k]['pikan']
        if os.path.exists(pikan_file):
            U_pikan, V_pikan, P_pikan, X_pikan, Y_pikan, _ = load_and_evaluate_model(pikan_file, model_type='pikan')
            
            if U_pikan is not None:
                # Horizontal centerline
                centerline_idx_pikan = np.argmin(np.abs(Y_pikan[:, 0] - centerline_y))
                x_pikan = X_pikan[centerline_idx_pikan, :]
                v_pikan = V_pikan[centerline_idx_pikan, :].copy()
                v_pikan = zero_inside_cylinder(v_pikan, x_pikan, cylinder_center[0], cylinder_radius)
                
                mask_x_pikan = (x_pikan >= 0) & (x_pikan <= 0.3) & ~np.isnan(v_pikan)
                line1, = ax1.plot(x_pikan[mask_x_pikan], v_pikan[mask_x_pikan], 
                         color=method_colors['pikan'], linestyle=method_linestyles['pikan'], 
                         marker=k_markers[k], markevery=5, markersize=6,
                         linewidth=2, label=rf'PIKAN, $k={k}$, $n_{{\mathrm{{cp}}}}=100^2$')
                lines_ax1.append(line1)
                labels_ax1.append(rf'PIKAN, $k={k}$, $n_{{\mathrm{{cp}}}}=100^2$')
                
                # Vertical centerline
                centerline_idx_pikan = np.argmin(np.abs(X_pikan[0, :] - centerline_x))
                y_pikan = Y_pikan[:, centerline_idx_pikan]
                u_pikan = U_pikan[:, centerline_idx_pikan].copy()
                u_pikan = zero_inside_cylinder(u_pikan, y_pikan, cylinder_center[1], cylinder_radius)
                
                mask_y_pikan = (y_pikan >= 0.7) & (y_pikan <= 1.0) & ~np.isnan(u_pikan)
                line2, = ax2.plot(u_pikan[mask_y_pikan], y_pikan[mask_y_pikan], 
                         color=method_colors['pikan'], linestyle=method_linestyles['pikan'],
                         marker=k_markers[k], markevery=5, markersize=6,
                         linewidth=2, label=rf'PIKAN, $k={k}$, $n_{{\mathrm{{cp}}}}=100^2$')
                lines_ax2.append(line2)
                labels_ax2.append(rf'PIKAN, $k={k}$, $n_{{\mathrm{{cp}}}}=100^2$')
    
    # Second pass: Plot all SPIKAN data
    for k in [3, 4, 5]:
        spikan_file = data_files_by_k[k]['spikan'][100]
        if os.path.exists(spikan_file):
            U_spikan, V_spikan, P_spikan, X_spikan, Y_spikan, _ = load_and_evaluate_model(spikan_file, model_type='spikan')
            
            if U_spikan is not None:
                # Horizontal centerline
                centerline_idx_spikan = np.argmin(np.abs(Y_spikan[:, 0] - centerline_y))
                x_spikan = X_spikan[centerline_idx_spikan, :]
                v_spikan = V_spikan[centerline_idx_spikan, :].copy()
                v_spikan = zero_inside_cylinder(v_spikan, x_spikan, cylinder_center[0], cylinder_radius)
                
                mask_x_spikan = (x_spikan >= 0) & (x_spikan <= 0.3) & ~np.isnan(v_spikan)
                line1, = ax1.plot(x_spikan[mask_x_spikan], v_spikan[mask_x_spikan], 
                         color=method_colors['spikan'], linestyle=method_linestyles['spikan'],
                         marker=k_markers[k], markevery=5, markersize=6,
                         linewidth=2, label=rf'SPIKAN, $k={k}$, $n_{{\mathrm{{cp}}}}=100^2$')
                lines_ax1.append(line1)
                labels_ax1.append(rf'SPIKAN, $k={k}$, $n_{{\mathrm{{cp}}}}=100^2$')
                
                # Vertical centerline
                centerline_idx_spikan = np.argmin(np.abs(X_spikan[0, :] - centerline_x))
                y_spikan = Y_spikan[:, centerline_idx_spikan]
                u_spikan = U_spikan[:, centerline_idx_spikan].copy()
                u_spikan = zero_inside_cylinder(u_spikan, y_spikan, cylinder_center[1], cylinder_radius)
                
                mask_y_spikan = (y_spikan >= 0.7) & (y_spikan <= 1.0) & ~np.isnan(u_spikan)
                line2, = ax2.plot(u_spikan[mask_y_spikan], y_spikan[mask_y_spikan], 
                         color=method_colors['spikan'], linestyle=method_linestyles['spikan'],
                         marker=k_markers[k], markevery=5, markersize=6,
                         linewidth=2, label=rf'SPIKAN, $k={k}$, $n_{{\mathrm{{cp}}}}=100^2$')
                lines_ax2.append(line2)
                labels_ax2.append(rf'SPIKAN, $k={k}$, $n_{{\mathrm{{cp}}}}=100^2$')
    
    # Configure left plot (horizontal centerline)
    ax1.set_xlabel(r'$x/L$')
    ax1.set_ylabel(r'$v/U_0$')
    ax1.set_title('v on Horizontal Centerline')
    ax1.grid(True, alpha=0.3)
    ax1.legend(lines_ax1, labels_ax1, loc='best')
    ax1.set_xlim(0, 0.3)
    
    # Configure right plot (vertical centerline)
    ax2.set_xlabel(r'$u/U_0$')
    ax2.set_ylabel(r'$y/L$')
    ax2.set_title('u on Vertical Centerline')
    ax2.grid(True, alpha=0.3)
    ax2.legend(lines_ax2, labels_ax2, loc='best')
    ax2.set_ylim(0.7, 1.0)
    
    # Format axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    plt.tight_layout()
    
    # Save figure
    save_path = './results/centerline_comparison_zoomed.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved zoomed centerline plot to: {save_path}")

def create_activation_comparison_plot():
    """Create centerline comparison plot for different activation functions."""
    
    print("\nCreating activation function comparison plot...")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Load reference data once
    centerline_y = 0.5
    centerline_x = 0.5
    
    # Horizontal centerline (y=0.5) - v velocity (LEFT PLOT, x=0 to 0.3)
    centerline_idx_ref = np.argmin(np.abs(Y_ref[:, 0] - centerline_y))
    x_ref = X_ref[centerline_idx_ref, :]
    v_ref = V_ref[centerline_idx_ref, :].copy()
    
    # Set velocity to zero inside cylinder
    v_ref = zero_inside_cylinder(v_ref, x_ref, cylinder_center[0], cylinder_radius)
    
    # Filter for x=0 to 0.3
    mask_x_ref = (x_ref >= 0) & (x_ref <= 0.3) & ~np.isnan(v_ref)
    ax1.plot(x_ref[mask_x_ref], v_ref[mask_x_ref], 
             color='black', linestyle='-', linewidth=2.5,
             label='Reference (FVM)')
    
    # Vertical centerline (x=0.5) - u velocity (RIGHT PLOT, y=0.7 to 1.0)
    centerline_idx_ref = np.argmin(np.abs(X_ref[0, :] - centerline_x))
    y_ref = Y_ref[:, centerline_idx_ref]
    u_ref = U_ref[:, centerline_idx_ref].copy()
    
    # Set velocity to zero inside cylinder
    u_ref = zero_inside_cylinder(u_ref, y_ref, cylinder_center[1], cylinder_radius)
    
    # Filter for y=0.7 to 1.0
    mask_y_ref = (y_ref >= 0.7) & (y_ref <= 1.0) & ~np.isnan(u_ref)
    ax2.plot(u_ref[mask_y_ref], y_ref[mask_y_ref], 
             color='black', linestyle='-', linewidth=2.5,
             label='Reference (FVM)')
    
    # Define colors and markers for different activation functions
    activation_colors = {'silu': 'blue', 'tanh': 'green', 'relu': 'red', 'sine': 'orange'}
    activation_markers = {'silu': 'o', 'tanh': 's', 'relu': '^', 'sine': 'D'}
    activation_linestyles = {'silu': '-', 'tanh': '--', 'relu': '-.', 'sine': ':'}
    
    # Process all activation functions
    for activation in ['silu', 'tanh', 'relu']:  # 'sine' commented out
        activation_file = activation_files[activation]
        if os.path.exists(activation_file):
            U_spikan, V_spikan, P_spikan, X_spikan, Y_spikan, _ = load_and_evaluate_model(activation_file, model_type='spikan')
            
            if U_spikan is not None:
                # Horizontal centerline
                centerline_idx_spikan = np.argmin(np.abs(Y_spikan[:, 0] - centerline_y))
                x_spikan = X_spikan[centerline_idx_spikan, :]
                v_spikan = V_spikan[centerline_idx_spikan, :].copy()
                v_spikan = zero_inside_cylinder(v_spikan, x_spikan, cylinder_center[0], cylinder_radius)
                
                mask_x_spikan = (x_spikan >= 0) & (x_spikan <= 0.3) & ~np.isnan(v_spikan)
                ax1.plot(x_spikan[mask_x_spikan], v_spikan[mask_x_spikan], 
                         color=activation_colors[activation], 
                         linestyle=activation_linestyles[activation],
                         marker=activation_markers[activation], markevery=5, markersize=6,
                         linewidth=2, label=rf'SPIKAN, $k=5$, $n_{{\mathrm{{cp}}}}=800^2$, {activation}')
                
                # Vertical centerline
                centerline_idx_spikan = np.argmin(np.abs(X_spikan[0, :] - centerline_x))
                y_spikan = Y_spikan[:, centerline_idx_spikan]
                u_spikan = U_spikan[:, centerline_idx_spikan].copy()
                u_spikan = zero_inside_cylinder(u_spikan, y_spikan, cylinder_center[1], cylinder_radius)
                
                mask_y_spikan = (y_spikan >= 0.7) & (y_spikan <= 1.0) & ~np.isnan(u_spikan)
                ax2.plot(u_spikan[mask_y_spikan], y_spikan[mask_y_spikan], 
                         color=activation_colors[activation],
                         linestyle=activation_linestyles[activation],
                         marker=activation_markers[activation], markevery=5, markersize=6,
                         linewidth=2, label=rf'SPIKAN, $k=5$, $n_{{\mathrm{{cp}}}}=800^2$, {activation}')
    
    # Configure left plot (horizontal centerline)
    ax1.set_xlabel(r'$x/L$')
    ax1.set_ylabel(r'$v/U_0$')
    ax1.set_title('v on Horizontal Centerline')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xlim(0, 0.3)
    
    # Configure right plot (vertical centerline)
    ax2.set_xlabel(r'$u/U_0$')
    ax2.set_ylabel(r'$y/L$')
    ax2.set_title('u on Vertical Centerline')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_ylim(0.7, 1.0)
    
    # Format axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    plt.tight_layout()
    
    # Save figure
    save_path = './results/centerline_comparison_activations.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved activation comparison plot to: {save_path}")

# Create output directory
os.makedirs('./results', exist_ok=True)

# Create the zoomed centerline comparison plot
create_centerline_plot_zoomed()

print("\nCenterline plots generated successfully!")

# Create contour comparison plots for best cases
print("\nGenerating contour comparison plots for best cases...")

def create_contour_comparison(method_name, U_pred, V_pred, P_pred, X_pred, Y_pred, save_path):
    """Create contour comparison plot with FVM reference, predictions, and absolute differences."""
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    
    # Normalize pressure for both predicted and reference
    P_pred_mean = np.nanmean(P_pred)
    P_pred_norm = P_pred - P_pred_mean
    
    P_ref_mean = np.nanmean(P_ref)
    P_ref_norm = P_ref - P_ref_mean
    
    # Interpolate predictions to reference grid for difference calculation
    from scipy.interpolate import RegularGridInterpolator
    
    # Get unique coordinates
    x_pred = X_pred[0, :]
    y_pred = Y_pred[:, 0]
    
    # Interpolate U
    interp_U = RegularGridInterpolator((y_pred, x_pred), U_pred, 
                                       method='linear', bounds_error=False, fill_value=np.nan)
    points = np.column_stack((Y_ref.ravel(), X_ref.ravel()))
    U_interp = interp_U(points).reshape(U_ref.shape)
    
    # Interpolate V
    interp_V = RegularGridInterpolator((y_pred, x_pred), V_pred, 
                                       method='linear', bounds_error=False, fill_value=np.nan)
    V_interp = interp_V(points).reshape(V_ref.shape)
    
    # Interpolate P
    interp_P = RegularGridInterpolator((y_pred, x_pred), P_pred_norm, 
                                       method='linear', bounds_error=False, fill_value=np.nan)
    P_norm_interp = interp_P(points).reshape(P_ref.shape)
    
    # Find data limits for consistent color scales across all plots
    u_min = min(np.nanmin(U_ref), np.nanmin(U_pred))
    u_max = max(np.nanmax(U_ref), np.nanmax(U_pred))
    v_min = min(np.nanmin(V_ref), np.nanmin(V_pred))
    v_max = max(np.nanmax(V_ref), np.nanmax(V_pred))
    p_min = min(np.nanmin(P_ref_norm), np.nanmin(P_pred_norm))
    p_max = max(np.nanmax(P_ref_norm), np.nanmax(P_pred_norm))
    
    # Define consistent color levels
    u_levels = np.linspace(u_min, u_max, 50)
    v_levels = np.linspace(v_min, v_max, 50)
    p_levels = np.linspace(p_min, p_max, 50)
    
    # Row 1: FVM Reference
    # u velocity
    im00 = axs[0, 0].contourf(X_ref, Y_ref, U_ref, levels=u_levels, cmap='RdBu_r')
    axs[0, 0].set_title(r'Reference $u$ (FVM)')
    cbar00 = plt.colorbar(im00, ax=axs[0, 0], fraction=0.046, pad=0.04)
    cbar00.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # v velocity
    im01 = axs[0, 1].contourf(X_ref, Y_ref, V_ref, levels=v_levels, cmap='RdBu_r')
    axs[0, 1].set_title(r'Reference $v$ (FVM)')
    cbar01 = plt.colorbar(im01, ax=axs[0, 1], fraction=0.046, pad=0.04)
    cbar01.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # pressure (normalized)
    im02 = axs[0, 2].contourf(X_ref, Y_ref, P_ref_norm, levels=p_levels, cmap='RdBu_r')
    axs[0, 2].set_title(r'Reference $p$ (FVM)')
    cbar02 = plt.colorbar(im02, ax=axs[0, 2], fraction=0.046, pad=0.04)
    cbar02.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # Row 2: Predictions
    # u velocity
    im10 = axs[1, 0].contourf(X_pred, Y_pred, U_pred, levels=u_levels, cmap='RdBu_r')
    axs[1, 0].set_title(rf'Predicted $u$ ({method_name})')
    cbar10 = plt.colorbar(im10, ax=axs[1, 0], fraction=0.046, pad=0.04)
    cbar10.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # v velocity
    im11 = axs[1, 1].contourf(X_pred, Y_pred, V_pred, levels=v_levels, cmap='RdBu_r')
    axs[1, 1].set_title(rf'Predicted $v$ ({method_name})')
    cbar11 = plt.colorbar(im11, ax=axs[1, 1], fraction=0.046, pad=0.04)
    cbar11.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # pressure (normalized)
    im12 = axs[1, 2].contourf(X_pred, Y_pred, P_pred_norm, levels=p_levels, cmap='RdBu_r')
    axs[1, 2].set_title(rf'Predicted $p$ ({method_name})')
    cbar12 = plt.colorbar(im12, ax=axs[1, 2], fraction=0.046, pad=0.04)
    cbar12.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # Row 3: Absolute Differences
    # Calculate absolute differences
    diff_U = np.abs(U_interp - U_ref)
    diff_V = np.abs(V_interp - V_ref)
    diff_P = np.abs(P_norm_interp - P_ref_norm)
    
    # u difference
    im20 = axs[2, 0].contourf(X_ref, Y_ref, diff_U, levels=50, cmap='RdBu_r')
    axs[2, 0].set_title(r'Absolute Error in $u$')
    cbar20 = plt.colorbar(im20, ax=axs[2, 0], fraction=0.046, pad=0.04)
    cbar20.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # v difference
    im21 = axs[2, 1].contourf(X_ref, Y_ref, diff_V, levels=50, cmap='RdBu_r')
    axs[2, 1].set_title(r'Absolute Error in $v$')
    cbar21 = plt.colorbar(im21, ax=axs[2, 1], fraction=0.046, pad=0.04)
    cbar21.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # pressure difference
    im22 = axs[2, 2].contourf(X_ref, Y_ref, diff_P, levels=50, cmap='RdBu_r')
    axs[2, 2].set_title(r'Absolute Error in $p$')
    cbar22 = plt.colorbar(im22, ax=axs[2, 2], fraction=0.046, pad=0.04)
    cbar22.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # Add cylinder to all plots
    for ax in axs.flat:
        circle = Circle(cylinder_center, cylinder_radius, fill=True, 
                       edgecolor='black', facecolor='white', linewidth=2)
        ax.add_artist(circle)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('x/L')
        ax.set_ylabel('y/L')
        # Format axis ticks to 2 decimal places
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved contour comparison to: {save_path}")

# Best PIKAN case: k=5, nx=100
best_pikan_file = data_files_by_k[5]['pikan']
if os.path.exists(best_pikan_file):
    print("  Creating contour comparison for best PIKAN (k=5, nx=100)...")
    U_pikan, V_pikan, P_pikan, X_pikan, Y_pikan, _ = load_and_evaluate_model(best_pikan_file, model_type='pikan')
    if U_pikan is not None:
        create_contour_comparison('PIKAN', U_pikan, V_pikan, P_pikan, X_pikan, Y_pikan, 
                                './results/contour_comparison_pikan_best.png')

# Best SPIKAN case: k=3, nx=800, silu
best_spikan_file = data_files_by_k[3]['spikan'][800]
if os.path.exists(best_spikan_file):
    print("  Creating contour comparison for best SPIKAN (k=3, nx=800, silu)...")
    U_spikan, V_spikan, P_spikan, X_spikan, Y_spikan, _ = load_and_evaluate_model(best_spikan_file, model_type='spikan')
    if U_spikan is not None:
        create_contour_comparison('SPIKAN', U_spikan, V_spikan, P_spikan, X_spikan, Y_spikan, 
                                './results/contour_comparison_spikan_best.png')

print("\nAll plots generated successfully!")

# Create streamline comparison plots
print("\nGenerating streamline comparison plots...")

def create_streamline_comparison(U_pikan, V_pikan, X_pikan, Y_pikan,
                               U_spikan, V_spikan, X_spikan, Y_spikan,
                               save_path):
    """Create streamline comparison plot for reference, PIKAN, and SPIKAN."""
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Reference streamlines
    # Replace NaN with 0 for streamplot
    U_stream_ref = np.where(np.isnan(U_ref), 0, U_ref)
    V_stream_ref = np.where(np.isnan(V_ref), 0, V_ref)
    axs[0].streamplot(X_ref, Y_ref, U_stream_ref, V_stream_ref, density=2, color='k', linewidth=1)
    axs[0].set_title('Reference (FVM)', fontsize=16)
    
    # PIKAN streamlines
    if U_pikan is not None:
        U_stream_pikan = np.where(np.isnan(U_pikan), 0, U_pikan)
        V_stream_pikan = np.where(np.isnan(V_pikan), 0, V_pikan)
        axs[1].streamplot(X_pikan, Y_pikan, U_stream_pikan, V_stream_pikan, density=2, color='k', linewidth=1)
        axs[1].set_title('PIKAN', fontsize=16)
    
    # SPIKAN streamlines
    if U_spikan is not None:
        U_stream_spikan = np.where(np.isnan(U_spikan), 0, U_spikan)
        V_stream_spikan = np.where(np.isnan(V_spikan), 0, V_spikan)
        axs[2].streamplot(X_spikan, Y_spikan, U_stream_spikan, V_stream_spikan, density=2, color='k', linewidth=1)
        axs[2].set_title('SPIKAN', fontsize=16)
    
    # Add cylinders and format all subplots
    for ax in axs:
        circle = Circle(cylinder_center, cylinder_radius, fill=True, 
                       edgecolor='black', facecolor='white', linewidth=2)
        ax.add_artist(circle)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('x/L')
        ax.set_ylabel('y/L')
        # Format axis ticks to 2 decimal places
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved streamline comparison to: {save_path}")

# Create streamline plots for best cases
# Best PIKAN case: k=5, nx=100
best_pikan_file = data_files_by_k[5]['pikan']
if os.path.exists(best_pikan_file):
    print("  Creating streamline comparison for best PIKAN (k=5, nx=100)...")
    U_pikan, V_pikan, P_pikan, X_pikan, Y_pikan, _ = load_and_evaluate_model(best_pikan_file, model_type='pikan')
    
    # Best SPIKAN case: k=3, nx=800
    best_spikan_file = data_files_by_k[3]['spikan'][800]
    if os.path.exists(best_spikan_file):
        U_spikan, V_spikan, P_spikan, X_spikan, Y_spikan, _ = load_and_evaluate_model(best_spikan_file, model_type='spikan')
        
        if U_pikan is not None and U_spikan is not None:
            create_streamline_comparison(U_pikan, V_pikan, X_pikan, Y_pikan,
                                       U_spikan, V_spikan, X_spikan, Y_spikan,
                                       './results/streamline_comparison_best.png')

# Calculate and display L2 errors for all cases
print("\n" + "="*80)
print("L2 RELATIVE ERRORS SUMMARY")
print("="*80)

def calculate_l2_error_on_grid(U_pred, V_pred, P_pred, X_pred, Y_pred, U_ref, V_ref, P_ref, X_ref, Y_ref):
    """Calculate L2 error by interpolating to reference grid."""
    from scipy.interpolate import RegularGridInterpolator
    
    # Get unique coordinates
    x_pred = X_pred[0, :]
    y_pred = Y_pred[:, 0]
    x_ref_unique = X_ref[0, :]
    y_ref_unique = Y_ref[:, 0]
    
    # Interpolate U
    interp_U = RegularGridInterpolator((y_pred, x_pred), U_pred, 
                                       method='linear', bounds_error=False, fill_value=np.nan)
    points = np.column_stack((Y_ref.ravel(), X_ref.ravel()))
    U_interp = interp_U(points).reshape(U_ref.shape)
    
    # Interpolate V
    interp_V = RegularGridInterpolator((y_pred, x_pred), V_pred, 
                                       method='linear', bounds_error=False, fill_value=np.nan)
    V_interp = interp_V(points).reshape(V_ref.shape)
    
    # Interpolate P
    interp_P = RegularGridInterpolator((y_pred, x_pred), P_pred, 
                                       method='linear', bounds_error=False, fill_value=np.nan)
    P_interp = interp_P(points).reshape(P_ref.shape)
    
    # Calculate errors where all are valid
    valid = ~(np.isnan(U_interp) | np.isnan(V_interp) | np.isnan(P_interp) | 
              np.isnan(U_ref) | np.isnan(V_ref) | np.isnan(P_ref))
    
    U_diff = U_interp[valid] - U_ref[valid]
    V_diff = V_interp[valid] - V_ref[valid]
    P_diff = P_interp[valid] - P_ref[valid]
    
    l2_error_u = 100 * np.sqrt(np.sum(U_diff**2)) / np.sqrt(np.sum(U_ref[valid]**2))
    l2_error_v = 100 * np.sqrt(np.sum(V_diff**2)) / np.sqrt(np.sum(V_ref[valid]**2))
    l2_error_p = 100 * np.sqrt(np.sum(P_diff**2)) / np.sqrt(np.sum(P_ref[valid]**2))
    
    return l2_error_u, l2_error_v, l2_error_p

for k in [3, 4, 5]:
    print(f"\nk = {k}:")
    print("-" * 60)
    
    pikan_file = data_files_by_k[k]['pikan']
    spikan_files = data_files_by_k[k]['spikan']
    
    # PIKAN error
    if os.path.exists(pikan_file):
        U_pikan, V_pikan, P_pikan, X_pikan, Y_pikan, _ = load_and_evaluate_model(pikan_file, model_type='pikan')
        if U_pikan is not None:
            err_u, err_v, err_p = calculate_l2_error_on_grid(U_pikan, V_pikan, P_pikan, X_pikan, Y_pikan, 
                                                            U_ref, V_ref, P_ref, X_ref, Y_ref)
            print(f"  PIKAN (k={k}, nx=100): u={err_u:.2f}%, v={err_v:.2f}%, p={err_p:.2f}%")
    
    # SPIKAN errors
    for nx in [100, 400, 800]:
        if os.path.exists(spikan_files[nx]):
            U, V, P, X, Y, _ = load_and_evaluate_model(spikan_files[nx], model_type='spikan')
            if U is not None:
                err_u, err_v, err_p = calculate_l2_error_on_grid(U, V, P, X, Y, U_ref, V_ref, P_ref, X_ref, Y_ref)
                print(f"  SPIKAN (k={k}, nx={nx}): u={err_u:.2f}%, v={err_v:.2f}%, p={err_p:.2f}%")

# Add L2 errors for different activation functions
print("\nActivation function comparison (SPIKAN k=5, nx=800):")
print("-" * 60)
for activation in ['silu', 'tanh', 'relu', 'sine']:
    activation_file = activation_files[activation]
    if os.path.exists(activation_file):
        U, V, P, X, Y, _ = load_and_evaluate_model(activation_file, model_type='spikan')
        if U is not None:
            err_u, err_v, err_p = calculate_l2_error_on_grid(U, V, P, X, Y, U_ref, V_ref, P_ref, X_ref, Y_ref)
            print(f"  SPIKAN (k=5, nx=800, {activation}): u={err_u:.2f}%, v={err_v:.2f}%, p={err_p:.2f}%")

print("\nAll plots including streamlines generated successfully!")