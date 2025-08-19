import jax.numpy as jnp
import jax
from jax import vmap, jit
from functools import partial
import numpy as np
import sys
import os
import argparse
import ast
sys.path.insert(0, '../')

from KANWrapper import SF_KAN

# Problem parameters
L_range = (0.0, 1.0)
H_range = (0.0, 1.0)
cylinder_center = (0.5, 0.5)
cylinder_radius = 0.2
Re = 100.0  # Reynolds number

class CavityCylinder_SF_KAN(SF_KAN):
    def __init__(self, *args, Re=100.0, cylinder_center=(0.5, 0.5), cylinder_radius=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.Re = Re
        self.cylinder_center = jnp.array(cylinder_center)
        self.cylinder_radius = cylinder_radius

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params, state, xy_domain, xy_boundary):
        variables = {'params': params, 'state': state}
        
        # Compute predictions for all points at once
        all_points = jnp.vstack([xy_domain, xy_boundary])
        preds, _ = self.forward_pass(variables, all_points)
        
        # Split predictions for domain and boundary
        n_domain = xy_domain.shape[0]
        preds_domain = preds[:n_domain]
        preds_boundary = preds[n_domain:]
        
        # Physics loss (Navier-Stokes equations)
        def ns_residual(pred, x, y):
            u, v, p = pred
            
            # Compute all gradients at once
            grads = jax.jacrev(lambda inp: self.forward_pass(variables, inp.reshape(1, -1))[0].squeeze())(jnp.array([x, y]))
            du_dx, du_dy = grads[0]
            dv_dx, dv_dy = grads[1]
            dp_dx, dp_dy = grads[2]
            
            # Compute second derivatives
            d2u_dx2 = jax.jacfwd(lambda x: jax.jacfwd(lambda x: self.forward_pass(variables, jnp.array([[x, y]]))[0][0, 0])(x))(x)
            d2u_dy2 = jax.jacfwd(lambda y: jax.jacfwd(lambda y: self.forward_pass(variables, jnp.array([[x, y]]))[0][0, 0])(y))(y)
            d2v_dx2 = jax.jacfwd(lambda x: jax.jacfwd(lambda x: self.forward_pass(variables, jnp.array([[x, y]]))[0][0, 1])(x))(x)
            d2v_dy2 = jax.jacfwd(lambda y: jax.jacfwd(lambda y: self.forward_pass(variables, jnp.array([[x, y]]))[0][0, 1])(y))(y)
            
            continuity = du_dx + dv_dy
            momentum_x = u * du_dx + v * du_dy + dp_dx - (1./self.Re)*(d2u_dx2 + d2u_dy2)
            momentum_y = u * dv_dx + v * dv_dy + dp_dy - (1./self.Re)*(d2v_dx2 + d2v_dy2)
            return jnp.array([continuity, momentum_x, momentum_y])

        residuals = vmap(ns_residual)(preds_domain, xy_domain[:, 0], xy_domain[:, 1])
        physics_loss = jnp.mean(residuals**2)

        # Boundary loss
        def boundary_loss_fn(pred, x, y):
            u, v, p = pred
            L = 1.
            
            # Check if point is on cylinder
            dist_to_center = jnp.sqrt((x - self.cylinder_center[0])**2 + 
                                    (y - self.cylinder_center[1])**2)
            on_cylinder = jnp.abs(dist_to_center - self.cylinder_radius) < 1e-3
            
            # Box boundaries
            left_wall_loss = jnp.where(x == 0, u**2 + v**2, 0)
            right_wall_loss = jnp.where(x == L, u**2 + v**2, 0)
            bottom_wall_loss = jnp.where(y == 0, u**2 + v**2, 0)
            top_wall_loss = jnp.where(y == L, (u - 1.)**2 + v**2, 0)
            
            # Cylinder boundary (no-slip)
            cylinder_loss = jnp.where(on_cylinder, u**2 + v**2, 0)
            
            return left_wall_loss + right_wall_loss + bottom_wall_loss + top_wall_loss + cylinder_loss

        boundary_losses = vmap(boundary_loss_fn)(preds_boundary, xy_boundary[:, 0], xy_boundary[:, 1])
        boundary_loss = jnp.mean(boundary_losses)

        # Total loss (no regularization for now)
        total_loss = physics_loss + boundary_loss
        return total_loss


# Only run training if this file is executed directly
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train PIKAN for cavity flow with cylinder')
    parser.add_argument('--nx', type=int, default=100, help='Number of x grid points (default: 100)')
    parser.add_argument('--ny', type=int, default=100, help='Number of y grid points (default: 100)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs (default: 500)')
    parser.add_argument('--k', type=int, default=3, help='B-spline degree (default: 3)')
    parser.add_argument('--Re', type=float, default=100.0, help='Reynolds number (default: 100.0)')
    parser.add_argument('--layer_dims', type=str, default='[2,10,10,3]', 
                        help='Layer dimensions as list (default: [2,10,10,3])')
    
    args = parser.parse_args()
    
    # Parse layer_dims from string to list
    layer_dims = ast.literal_eval(args.layer_dims)
    
    # Extract arguments
    nx, ny = args.nx, args.ny
    num_epochs = args.epochs
    k = args.k
    Re = args.Re
    x = np.linspace(L_range[0], L_range[1], nx)
    y = np.linspace(H_range[0], H_range[1], ny)
    X, Y = np.meshgrid(x, y)

    # Filter out points inside the cylinder
    xy_all = np.column_stack((X.ravel(), Y.ravel()))
    distances = np.sqrt((xy_all[:, 0] - cylinder_center[0])**2 + 
                       (xy_all[:, 1] - cylinder_center[1])**2)
    outside_cylinder = distances > cylinder_radius
    xy_domain = xy_all[outside_cylinder]

    # Create boundary points
    # 1. Box boundaries
    x_boundary = np.concatenate([np.full(ny, L_range[0]), np.full(ny, L_range[1]), x, x])
    y_boundary = np.concatenate([y, y, np.full(nx, H_range[0]), np.full(nx, H_range[1])])
    xy_box_boundary = np.column_stack((x_boundary, y_boundary))

    # 2. Cylinder boundary
    n_cylinder = nx
    theta = np.linspace(0, 2*np.pi, n_cylinder, endpoint=False)
    x_cylinder = cylinder_center[0] + cylinder_radius * np.cos(theta)
    y_cylinder = cylinder_center[1] + cylinder_radius * np.sin(theta)
    xy_cylinder_boundary = np.column_stack((x_cylinder, y_cylinder))

    # Combine all boundary points
    xy_boundary = np.vstack([xy_box_boundary, xy_cylinder_boundary])

    # Model parameters
    init_lr = 1e-3

    # Create model
    model = CavityCylinder_SF_KAN(
        layer_dims=layer_dims,
        init_lr=init_lr,
        k=k,
        Re=Re,
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius
    )

    # Train the model
    print(f"Starting training for {num_epochs} epochs with nx={nx}, ny={ny}, k={k}, Re={Re}")
    print(f"Network architecture: {layer_dims}")
    
    # Track training time
    import time
    start_time = time.time()
    final_variables, loss_history = model.train(num_epochs=num_epochs, xy_domain=xy_domain, xy_boundary=xy_boundary)
    end_time = time.time()
    
    # Calculate iterations per second
    total_time = end_time - start_time
    iterations_per_sec = num_epochs / total_time
    ms_per_iter = 1000.0 / iterations_per_sec
    
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Average: {iterations_per_sec:.2f} iter/s ({ms_per_iter:.2f} ms/iter)")

    # Save only essential data
    output_data = {
        'parameters': {
            'Re': Re,
            'cylinder_center': cylinder_center,
            'cylinder_radius': cylinder_radius,
            'L_range': L_range,
            'H_range': H_range,
            'nx_train': nx,
            'ny_train': ny,
            'k': k,
            'layer_dims': layer_dims,
            'epochs': num_epochs
        },
        'training': {
            'loss_history': np.array(loss_history),
            'final_loss': float(loss_history[-1]),
            'total_time': total_time,
            'iterations_per_sec': iterations_per_sec,
            'ms_per_iter': ms_per_iter
        },
        'model': {
            'variables': final_variables
        }
    }

    # Create directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)

    # Save data
    save_path = f'./data/cavity_cylinder_pikan_Re_{Re}_nx{nx}_ny{ny}_epochs{num_epochs}_k{k}.npy'
    np.save(save_path, output_data)
    print(f"\nData saved to: {save_path}")
    print(f"Final loss: {loss_history[-1]:.4e}")

