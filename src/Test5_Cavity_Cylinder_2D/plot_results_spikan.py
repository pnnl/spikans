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

# Configure global font settings
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

# Specify the data file to load
data_file = './data/cavity_cylinder_spikan_Re_100.0_nx200_ny200_epochs100000_r10_k5.npy'

# Load OpenFOAM reference data
openfoam_file = './data/reference_data/openfoam_cavity_cylinder_Re100.npz'

if not os.path.exists(data_file):
    print(f"Error: Data file not found: {data_file}")
    exit(1)

if not os.path.exists(openfoam_file):
    print(f"Error: OpenFOAM reference file not found: {openfoam_file}")
    print("Please run vtk_to_npz.py first to generate the reference data.")
    exit(1)

print(f"Loading SPIKAN data from: {data_file}")
print(f"Loading OpenFOAM reference from: {openfoam_file}")

# Load data
data = np.load(data_file, allow_pickle=True).item()
ref_data = np.load(openfoam_file, allow_pickle=True)

# Extract parameters
params = data['parameters']
Re = params['Re']
cylinder_center = params['cylinder_center']
cylinder_radius = params['cylinder_radius']
L_range = params['L_range']
H_range = params['H_range']
r = params['r']
k = params['k']
layer_dims = params['layer_dims']

# Extract training info
loss_history = data['training']['loss_history']

# Recreate model and load saved weights
model = CavityCylinder_SF_KAN_Separable(
    layer_dims=layer_dims,
    init_lr=1e-3,  # Not used for inference
    k=k,
    r=r,
    Re=Re,
    cylinder_center=cylinder_center,
    cylinder_radius=cylinder_radius
)
model.variables_x = data['model']['variables_x']
model.variables_y = data['model']['variables_y']

# Create evaluation grid
nx_plot, ny_plot = 200, 200
x_plot = jnp.linspace(L_range[0], L_range[1], nx_plot)
y_plot = jnp.linspace(H_range[0], H_range[1], ny_plot)
X_mesh, Y_mesh = jnp.meshgrid(x_plot, y_plot)

# Make predictions
print("Generating predictions on evaluation grid...")
uvp_pred = model.predict(x_plot, y_plot)

# Extract fields and ensure they are numpy arrays
U = np.array(uvp_pred[:, :, 0]).T
V = np.array(uvp_pred[:, :, 1]).T
P = np.array(uvp_pred[:, :, 2]).T
vmag = np.sqrt(U**2 + V**2)

# Convert JAX arrays to numpy for matplotlib
X_mesh = np.array(X_mesh)
Y_mesh = np.array(Y_mesh)

# Create mask for plotting
plot_mask = create_interior_mask(x_plot, y_plot, cylinder_center, cylinder_radius)
U[~plot_mask] = np.nan
V[~plot_mask] = np.nan
P[~plot_mask] = np.nan
vmag[~plot_mask] = np.nan

# Extract OpenFOAM reference data
X_ref = ref_data['X_mesh']
Y_ref = ref_data['Y_mesh']
U_ref = ref_data['U']
V_ref = ref_data['V']
P_ref = ref_data['P']
UMAG_ref = ref_data['UMAG']

# Helper functions
def interpolate_to_ref_grid(field, X_source, Y_source, X_target, Y_target):
    """Interpolate field from source to target grid."""
    # Get unique coordinates
    x_source = X_source[0, :]
    y_source = Y_source[:, 0]
    x_target = X_target[0, :]
    y_target = Y_target[:, 0]
    
    # Create interpolator
    interpolator = interp.RegularGridInterpolator(
        (y_source, x_source),
        field,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Create points for interpolation
    XX, YY = np.meshgrid(x_target, y_target)
    points = np.column_stack((YY.ravel(), XX.ravel()))
    
    # Interpolate
    interpolated = interpolator(points).reshape(len(y_target), len(x_target))
    
    return interpolated

def calculate_l2_error(pred, ref, mask=None):
    """Calculate L2 relative error."""
    if mask is not None:
        pred = pred[mask]
        ref = ref[mask]
    
    # Remove NaN values
    valid = ~(np.isnan(pred) | np.isnan(ref))
    pred = pred[valid]
    ref = ref[valid]
    
    if len(ref) == 0:
        return np.nan
    
    l2_diff = np.sqrt(np.sum((pred - ref)**2))
    l2_ref = np.sqrt(np.sum(ref**2))
    
    return 100 * (l2_diff / l2_ref) if l2_ref != 0 else np.inf

# Create figure with subplots for field comparison
fig, axs = plt.subplots(3, 4, figsize=(20, 15))

# Get data limits for consistent color scales (include both reference and SPIKAN data)
u_min = min(np.nanmin(U_ref), np.nanmin(U))
u_max = max(np.nanmax(U_ref), np.nanmax(U))
v_min = min(np.nanmin(V_ref), np.nanmin(V))
v_max = max(np.nanmax(V_ref), np.nanmax(V))
umag_min = min(np.nanmin(UMAG_ref), np.nanmin(vmag))
umag_max = max(np.nanmax(UMAG_ref), np.nanmax(vmag))
P_ref_norm = (P_ref - np.nanmean(P_ref)) / np.nanmax(np.abs(P_ref - np.nanmean(P_ref)))
P_norm = (P - np.nanmean(P)) / np.nanmax(np.abs(P - np.nanmean(P)))
p_norm_min = min(np.nanmin(P_ref_norm), np.nanmin(P_norm))
p_norm_max = max(np.nanmax(P_ref_norm), np.nanmax(P_norm))

# Create common contour levels for each field
u_levels = np.linspace(u_min, u_max, 50)
v_levels = np.linspace(v_min, v_max, 50)
umag_levels = np.linspace(umag_min, umag_max, 50)
p_levels = np.linspace(p_norm_min, p_norm_max, 50)

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

# velocity magnitude
im02 = axs[0, 2].contourf(X_ref, Y_ref, UMAG_ref, levels=umag_levels, cmap='RdBu_r')
axs[0, 2].set_title(r'Reference $|\mathbf{u}|$ (FVM)')
cbar02 = plt.colorbar(im02, ax=axs[0, 2], fraction=0.046, pad=0.04)
cbar02.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# pressure (normalized)
im03 = axs[0, 3].contourf(X_ref, Y_ref, P_ref_norm, levels=p_levels, cmap='RdBu_r')
axs[0, 3].set_title(r'Reference $p$ (FVM)')
cbar03 = plt.colorbar(im03, ax=axs[0, 3], fraction=0.046, pad=0.04)
cbar03.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# Row 2: SPIKAN Predictions (using same levels as reference)
# u velocity
im10 = axs[1, 0].contourf(X_mesh, Y_mesh, U, levels=u_levels, cmap='RdBu_r')
axs[1, 0].set_title(r'Predicted $u$ (SPIKAN)')
cbar10 = plt.colorbar(im10, ax=axs[1, 0], fraction=0.046, pad=0.04)
cbar10.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# v velocity
im11 = axs[1, 1].contourf(X_mesh, Y_mesh, V, levels=v_levels, cmap='RdBu_r')
axs[1, 1].set_title(r'Predicted $v$ (SPIKAN)')
cbar11 = plt.colorbar(im11, ax=axs[1, 1], fraction=0.046, pad=0.04)
cbar11.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# velocity magnitude
im12 = axs[1, 2].contourf(X_mesh, Y_mesh, vmag, levels=umag_levels, cmap='RdBu_r')
axs[1, 2].set_title(r'Predicted $|\mathbf{u}|$ (SPIKAN)')
cbar12 = plt.colorbar(im12, ax=axs[1, 2], fraction=0.046, pad=0.04)
cbar12.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# pressure (normalized) - P_norm already computed above
im13 = axs[1, 3].contourf(X_mesh, Y_mesh, P_norm, levels=p_levels, cmap='RdBu_r')
axs[1, 3].set_title(r'Predicted $p$ (SPIKAN)')
cbar13 = plt.colorbar(im13, ax=axs[1, 3], fraction=0.046, pad=0.04)
cbar13.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# Row 3: Absolute Differences
# Interpolate SPIKAN to OpenFOAM grid for comparison
U_interp = interpolate_to_ref_grid(U, X_mesh, Y_mesh, X_ref, Y_ref)
V_interp = interpolate_to_ref_grid(V, X_mesh, Y_mesh, X_ref, Y_ref)
vmag_interp = interpolate_to_ref_grid(vmag, X_mesh, Y_mesh, X_ref, Y_ref)
P_norm_interp = interpolate_to_ref_grid(P_norm, X_mesh, Y_mesh, X_ref, Y_ref)

# Calculate absolute differences
diff_U = np.abs(U_interp - U_ref)
diff_V = np.abs(V_interp - V_ref)
diff_vmag = np.abs(vmag_interp - UMAG_ref)
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

# velocity magnitude difference
im22 = axs[2, 2].contourf(X_ref, Y_ref, diff_vmag, levels=50, cmap='RdBu_r')
axs[2, 2].set_title(r'Absolute Error in $|\mathbf{u}|$')
cbar22 = plt.colorbar(im22, ax=axs[2, 2], fraction=0.046, pad=0.04)
cbar22.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# pressure difference
im23 = axs[2, 3].contourf(X_ref, Y_ref, diff_P, levels=50, cmap='RdBu_r')
axs[2, 3].set_title(r'Absolute Error in $p$')
cbar23 = plt.colorbar(im23, ax=axs[2, 3], fraction=0.046, pad=0.04)
cbar23.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# Add cylinder to all plots
for ax in axs.flat:
    circle = Circle(cylinder_center, cylinder_radius, fill=True, 
                   edgecolor='black', facecolor='white', linewidth=2)
    ax.add_artist(circle)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # Format axis ticks to 2 decimal places
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# Create output directory
os.makedirs('./results', exist_ok=True)

plt.tight_layout()
plt.savefig('./results/field_comparison_with_openfoam.png', dpi=300, bbox_inches='tight')
plt.close()

# Create streamline comparison plot
fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))

# FVM streamlines
U_stream_ref = np.where(np.isnan(U_ref), 0, U_ref)
V_stream_ref = np.where(np.isnan(V_ref), 0, V_ref)
axs2[0].streamplot(X_ref, Y_ref, U_stream_ref, V_stream_ref, density=2, color='k', linewidth=1)
axs2[0].set_title('Streamlines - FVM Reference')

# SPIKAN streamlines
U_stream = np.where(np.isnan(U), 0, U)
V_stream = np.where(np.isnan(V), 0, V)
axs2[1].streamplot(X_mesh, Y_mesh, U_stream, V_stream, density=2, color='k', linewidth=1)
axs2[1].set_title('Streamlines - SPIKAN')

# Add cylinders
for ax in axs2:
    circle = Circle(cylinder_center, cylinder_radius, fill=True, 
                   edgecolor='black', facecolor='white', linewidth=2)
    ax.add_artist(circle)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # Format axis ticks to 2 decimal places
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.savefig('./results/streamline_comparison.png', dpi=300)
plt.close()

# Plot centerline comparisons
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Vertical centerline (x=0.5, but avoid cylinder)
centerline_idx_x = np.argmin(np.abs(X_mesh[0, :] - 0.5))
centerline_idx_x_ref = np.argmin(np.abs(X_ref[0, :] - 0.5))

# Extract data along vertical centerline
y_line = Y_mesh[:, centerline_idx_x]
u_line = U[:, centerline_idx_x]

y_line_ref = Y_ref[:, centerline_idx_x_ref]
u_line_ref = U_ref[:, centerline_idx_x_ref]

# Plot only valid regions (outside cylinder)
valid_idx = (y_line < (cylinder_center[1] - cylinder_radius - 0.05)) | \
            (y_line > (cylinder_center[1] + cylinder_radius + 0.05))
valid_idx_ref = (y_line_ref < (cylinder_center[1] - cylinder_radius - 0.05)) | \
                (y_line_ref > (cylinder_center[1] + cylinder_radius + 0.05))

ax1.plot(u_line_ref[valid_idx_ref], y_line_ref[valid_idx_ref], 'k-', linewidth=2, label='FVM')
ax1.plot(u_line[valid_idx], y_line[valid_idx], 'b--', linewidth=2, label='SPIKAN')
ax1.set_xlabel(r'$u/U_0$')
ax1.set_ylabel(r'$y/L$')
ax1.set_title('Vertical Centerline Velocity (x=0.5)')
ax1.grid(True, alpha=0.3)
ax1.legend()
# Format axis ticks to 2 decimal places
ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# Horizontal centerline (y=0.5, but avoid cylinder)
centerline_idx_y = np.argmin(np.abs(Y_mesh[:, 0] - 0.5))
centerline_idx_y_ref = np.argmin(np.abs(Y_ref[:, 0] - 0.5))

# Extract data along horizontal centerline
x_line = X_mesh[centerline_idx_y, :]
v_line = V[centerline_idx_y, :]

x_line_ref = X_ref[centerline_idx_y_ref, :]
v_line_ref = V_ref[centerline_idx_y_ref, :]

# Plot only valid regions (outside cylinder)
valid_idx_h = (x_line < (cylinder_center[0] - cylinder_radius - 0.05)) | \
              (x_line > (cylinder_center[0] + cylinder_radius + 0.05))
valid_idx_h_ref = (x_line_ref < (cylinder_center[0] - cylinder_radius - 0.05)) | \
                  (x_line_ref > (cylinder_center[0] + cylinder_radius + 0.05))

ax2.plot(x_line_ref[valid_idx_h_ref], v_line_ref[valid_idx_h_ref], 'k-', linewidth=2, label='FVM')
ax2.plot(x_line[valid_idx_h], v_line[valid_idx_h], 'b--', linewidth=2, label='SPIKAN')
ax2.set_xlabel(r'$x/L$')
ax2.set_ylabel(r'$v/U_0$')
ax2.set_title('Horizontal Centerline Velocity (y=0.5)')
ax2.grid(True, alpha=0.3)
ax2.legend()
# Format axis ticks to 2 decimal places
ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.savefig('./results/centerline_comparison_with_openfoam.png', dpi=300)
plt.close()

# Calculate and print L2 errors
print("\nL2 Relative Errors (%):")
print(f"  u velocity: {calculate_l2_error(U_interp, U_ref):.2f}%")
print(f"  v velocity: {calculate_l2_error(V_interp, V_ref):.2f}%")
print(f"  velocity magnitude: {calculate_l2_error(vmag_interp, UMAG_ref):.2f}%")
print(f"  pressure (normalized): {calculate_l2_error(P_norm_interp, P_ref_norm):.2f}%")

# Plot loss history
plt.figure(figsize=(8, 6))
epochs = range(0, len(loss_history), 100)
plt.plot(epochs, loss_history[::100])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)
plt.title('Training Loss History')
plt.tight_layout()
plt.savefig('./results/loss_history.png', dpi=300)
plt.close()

# Print info
print(f"\nFinal loss value: {loss_history[-1]:.6e}")
print(f"Number of training epochs: {len(loss_history)}")
print(f"Reynolds number: {Re}")
print("\nPlots saved to ./results/")