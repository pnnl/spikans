import jax.numpy as jnp
import jax
from jax import vmap, jit, jacfwd
import optax
from functools import partial
import numpy as np
from tqdm import trange
from jax import random
import sys
import os
import argparse
import ast
sys.path.insert(0, '../')

from KAN import KAN

def create_interior_points(L_range, H_range, nx, ny):
    x = np.linspace(L_range[0], L_range[1], nx)
    y = np.linspace(H_range[0], H_range[1], ny)
    return x, y

def create_boundary_points(L_range, H_range, nx, ny):
    x = np.linspace(L_range[0], L_range[1], nx)
    y = np.linspace(H_range[0], H_range[1], ny)

    x_left = jnp.array([L_range[0]])
    y_left = y

    x_top = x
    y_top = jnp.array([H_range[1]])

    x_right = jnp.array([L_range[1]])
    y_right = y

    x_bottom = x
    y_bottom = jnp.array([H_range[0]])

    return (x_left, y_left), (x_top, y_top), (x_right, y_right), (x_bottom, y_bottom)

def create_cylinder_boundary_points(center, radius, n_points):
    theta = jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)
    x_cylinder = center[0] + radius * jnp.cos(theta)
    y_cylinder = center[1] + radius * jnp.sin(theta)
    return x_cylinder, y_cylinder

def create_interior_mask(x_interior, y_interior, cylinder_center, cylinder_radius):
    """Create a mask for interior points outside the cylinder"""
    X, Y = jnp.meshgrid(x_interior, y_interior, indexing='ij')
    dist_from_center = jnp.sqrt((X - cylinder_center[0])**2 + 
                                (Y - cylinder_center[1])**2)
    mask = dist_from_center > cylinder_radius  # True = valid point
    return mask

# Define the domain
L_range = (0.0, 1.0)
H_range = (0.0, 1.0)  # Square cavity

# Cylinder parameters
cylinder_center = (0.5, 0.5)
cylinder_radius = 0.2

# Global training data variables - will be initialized in main


class SF_KAN_Separable:
    def __init__(self, layer_dims, init_lr, Re = 100, k=5, r=1, residual='swish'):
        self.input_size = layer_dims[0] # input should always be 1 for separable PINNs
        self.out_size = layer_dims[-1]
        self.r = r
        self.layer_dims = [self.input_size] + layer_dims[1:-1] + [self.r * self.out_size]
        
        # Map string to activation function
        import flax.linen as nn
        activation_map = {
            'swish': nn.swish,
            'silu': nn.swish,  # silu is same as swish
            'tanh': nn.tanh,
            'relu': nn.relu,
            'sine': lambda x: jnp.sin(x)
        }
        residual_fn = activation_map.get(residual, nn.swish)
        
        self.model_x = KAN(layer_dims=self.layer_dims, k=k, const_spl=False, const_res=False, 
                          add_bias=True, grid_e=0.02, j='0', residual=residual_fn)
        self.model_y = KAN(layer_dims=self.layer_dims, k=k, const_spl=False, const_res=False, 
                          add_bias=True, grid_e=0.02, j='0', residual=residual_fn)
        
        key1, key2 = jax.random.split(jax.random.PRNGKey(10))
        self.variables_x = self.model_x.init(key1, jnp.ones([1, 1]))
        self.variables_y = self.model_y.init(key2, jnp.ones([1, 1]))
        
        self.optimizer = optax.adam(learning_rate=init_lr, nesterov=True)
        self.opt_state_x = self.optimizer.init(self.variables_x['params'])
        self.opt_state_y = self.optimizer.init(self.variables_y['params'])

        self.train_losses = []

    def interpolate_moments(self, mu_old, nu_old, new_shape):
        old_shape = mu_old.shape
        size = old_shape[0]
        old_j = old_shape[1]
        new_j = new_shape[1]
        
        old_indices = jnp.linspace(0, old_j - 1, old_j)
        new_indices = jnp.linspace(0, old_j - 1, new_j)

        interpolate_fn = lambda old_row: jnp.interp(new_indices, old_indices, old_row)

        mu_new = vmap(interpolate_fn)(mu_old)
        nu_new = vmap(interpolate_fn)(nu_old)
        
        return mu_new, nu_new

    def smooth_state_transition(self, old_state, params):
        adam_count = old_state[0].count
        adam_mu, adam_nu = old_state[0].mu, old_state[0].nu

        layer_keys = {k for k in adam_mu.keys() if k.startswith('layers_')}
        
        for key in layer_keys:
            c_shape = params[key]['c_basis'].shape
            mu_new0, nu_new0 = self.interpolate_moments(adam_mu[key]['c_basis'], adam_nu[key]['c_basis'], c_shape)
            adam_mu[key]['c_basis'], adam_nu[key]['c_basis'] = mu_new0, nu_new0

        adam_state = optax.ScaleByAdamState(adam_count, adam_mu, adam_nu)
        extra_state = optax.ScaleByScheduleState(adam_count)
        new_state = (adam_state, extra_state)

        return new_state

    def predict(self, x, y):
        variables_x, variables_y = self.variables_x, self.variables_y
        preds, _ = self.forward_pass(variables_x, variables_y, x, y)
        return preds
    
    def evaluate_single_point(self, variables_x, variables_y, x, y):
        """Evaluate network at a single (x, y) point"""
        # Get features from each network
        x_feat, _ = self.model_x.apply(variables_x, x.reshape(1, 1))
        y_feat, _ = self.model_y.apply(variables_y, y.reshape(1, 1))
        
        # Reshape to (out_size, r)
        x_feat = x_feat.reshape(self.out_size, self.r)
        y_feat = y_feat.reshape(self.out_size, self.r)
        
        # Compute output
        output = jnp.sum(x_feat * y_feat, axis=-1)  # (out_size,)
        return output

    @partial(jit, static_argnums=(0,))
    def forward_pass(self, variables_x, variables_y, x, y):
        preds_x, spl_regs_x = self.model_x.apply(variables_x, x[:, None])
        preds_y, spl_regs_y = self.model_y.apply(variables_y, y[:, None])
        
        preds_x = preds_x.reshape(-1, self.out_size, self.r)
        preds_y = preds_y.reshape(-1, self.out_size, self.r)
        preds = jnp.einsum('ijk,ljk->ilj', preds_x, preds_y)

        spl_regs = spl_regs_x + spl_regs_y
        
        return preds, spl_regs
    
    @partial(jit, static_argnums=(0,))
    def forward_pass_nonseparable(self, variables_x, variables_y, x_points, y_points):
        """Forward pass for non-separable points (e.g., cylinder boundary)"""
        def eval_point(xy):
            x, y = xy
            return self.evaluate_single_point(variables_x, variables_y, x, y)
        
        xy_points = jnp.stack([x_points, y_points], axis=-1)
        preds = vmap(eval_point)(xy_points)
        return preds

    @partial(jit, static_argnums=(0,))
    def loss(self, params_x, params_y, state_x, state_y, *args):
        variables_x = {'params': params_x, 'state': state_x}
        variables_y = {'params': params_y, 'state': state_y}
        return self.loss_fn(variables_x, variables_y, *args)

    @partial(jit, static_argnums=(0,))
    def train_step(self, params_x, params_y, state_x, state_y, opt_state_x, opt_state_y, *args):
        (loss_value, (physics_loss, boundary_loss)), grads = jax.value_and_grad(self.loss, has_aux=True, argnums=(0,1))(
            params_x, params_y, state_x, state_y, *args
        )
        grads_x, grads_y = grads

        updates_x, opt_state_x = self.optimizer.update(grads_x, opt_state_x)
        updates_y, opt_state_y = self.optimizer.update(grads_y, opt_state_y)

        params_x = optax.apply_updates(params_x, updates_x)
        params_y = optax.apply_updates(params_y, updates_y)

        return params_x, params_y, opt_state_x, opt_state_y, loss_value, physics_loss, boundary_loss

    def train(self, num_epochs, *args):
        params_x, state_x = self.variables_x['params'], self.variables_x['state']
        params_y, state_y = self.variables_y['params'], self.variables_y['state']
        opt_state_x, opt_state_y = self.opt_state_x, self.opt_state_y
        loss_history = []

        pbar = trange(num_epochs, smoothing=0.)
        for epoch in pbar:                
            params_x, params_y, opt_state_x, opt_state_y, loss_value, physics_loss, boundary_loss = self.train_step(
                params_x, params_y, state_x, state_y, opt_state_x, opt_state_y, *args
            )
            loss_history.append(loss_value)

            if epoch % 10 == 0:
                pbar.set_postfix({
                    'Total Loss': f"{loss_value:.4e}",
                    'Physics Loss': f"{physics_loss:.4e}",
                    'Boundary Loss': f"{boundary_loss:.4e}"
                })
        
        self.variables_x = {'params': params_x, 'state': state_x}
        self.variables_y = {'params': params_y, 'state': state_y}
        return loss_history


class CavityCylinder_SF_KAN_Separable(SF_KAN_Separable):
    def __init__(self, *args, Re=100.0, cylinder_center=(0.5, 0.5), cylinder_radius=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.Re = Re
        self.cylinder_center = cylinder_center
        self.cylinder_radius = cylinder_radius

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, variables_x, variables_y, x_interior, y_interior, interior_mask,
                x_left, y_left, x_top, y_top, x_right, y_right, x_bottom, y_bottom,
                x_cylinder, y_cylinder):
        
        # Compute physics residuals with masking
        residuals = self.compute_residuals(variables_x, variables_y, x_interior, y_interior)
        # Apply mask to residuals
        masked_residuals = residuals * interior_mask[..., None]
        # Compute mean only over valid points
        n_valid = jnp.sum(interior_mask)
        physics_loss = jnp.sum(jnp.square(masked_residuals)) / (n_valid * 3)

        # Boundary losses for cavity walls (separable)
        preds_left, _ = self.forward_pass(variables_x, variables_y, x_left, y_left)
        preds_top, _ = self.forward_pass(variables_x, variables_y, x_top, y_top)
        preds_right, _ = self.forward_pass(variables_x, variables_y, x_right, y_right)
        preds_bottom, _ = self.forward_pass(variables_x, variables_y, x_bottom, y_bottom)

        left_loss = jnp.mean(jnp.square(preds_left[..., 0]) + jnp.square(preds_left[..., 1]))
        top_loss = jnp.mean(jnp.square(preds_top[..., 0] - 1.0) + jnp.square(preds_top[..., 1]))
        right_loss = jnp.mean(jnp.square(preds_right[..., 0]) + jnp.square(preds_right[..., 1]))
        bottom_loss = jnp.mean(jnp.square(preds_bottom[..., 0]) + jnp.square(preds_bottom[..., 1]))

        # Cylinder boundary loss (non-separable)
        preds_cylinder = self.forward_pass_nonseparable(variables_x, variables_y, x_cylinder, y_cylinder)
        cylinder_loss = jnp.mean(jnp.square(preds_cylinder[:, 0]) + jnp.square(preds_cylinder[:, 1]))

        boundary_loss = left_loss + top_loss + right_loss + bottom_loss + 10.0 * cylinder_loss

        # Total loss
        total_loss = physics_loss + boundary_loss

        return total_loss, (physics_loss, boundary_loss)

    @partial(jit, static_argnums=(0,))
    def compute_residuals(self, variables_x, variables_y, x_interior, y_interior):
        def model_x_func(x):
            x_feat = self.model_x.apply(variables_x, x.reshape(-1, 1))[0]
            x_feat = x_feat.reshape(self.out_size, self.r) 
            return x_feat

        def model_y_func(y):
            y_feat = self.model_y.apply(variables_y, y.reshape(-1, 1))[0]
            y_feat = y_feat.reshape(self.out_size, self.r)
            return y_feat

        def model_x_grad(x):
            return jacfwd(model_x_func)(x)

        def model_y_grad(y):
            return jacfwd(model_y_func)(y)

        def model_x_hess(x):
            return jacfwd(jacfwd(model_x_func))(x)

        def model_y_hess(y):
            return jacfwd(jacfwd(model_y_func))(y)

        x_feats = vmap(model_x_func)(x_interior)
        y_feats = vmap(model_y_func)(y_interior)
        x_grads = vmap(model_x_grad)(x_interior)
        y_grads = vmap(model_y_grad)(y_interior)
        x_hess = vmap(model_x_hess)(x_interior)
        y_hess = vmap(model_y_hess)(y_interior)

        u_x, v_x, p_x = x_feats[:, 0, :], x_feats[:, 1, :], x_feats[:, 2, :]
        u_y, v_y, p_y = y_feats[:, 0, :], y_feats[:, 1, :], y_feats[:, 2, :]

        du_x_dx, dv_x_dx, dp_x_dx = x_grads[:, 0, :], x_grads[:, 1, :], x_grads[:, 2, :]
        du_y_dy, dv_y_dy, dp_y_dy = y_grads[:, 0, :], y_grads[:, 1, :], y_grads[:, 2, :]

        d2u_x_dx2, d2v_x_dx2 = x_hess[:, 0, :], x_hess[:, 1, :]
        d2u_y_dy2, d2v_y_dy2 = y_hess[:, 0, :], y_hess[:, 1, :]

        u = jnp.einsum('ir,jr->ij', u_x, u_y)
        v = jnp.einsum('ir,jr->ij', v_x, v_y)
        p = jnp.einsum('ir,jr->ij', p_x, p_y)

        du_dx = jnp.einsum('ir,jr->ij', du_x_dx, u_y)
        du_dy = jnp.einsum('ir,jr->ij', u_x, du_y_dy)
        dv_dx = jnp.einsum('ir,jr->ij', dv_x_dx, v_y)
        dv_dy = jnp.einsum('ir,jr->ij', v_x, dv_y_dy)
        dp_dx = jnp.einsum('ir,jr->ij', dp_x_dx, p_y)
        dp_dy = jnp.einsum('ir,jr->ij', p_x, dp_y_dy)

        d2u_dx2 = jnp.einsum('ir,jr->ij', d2u_x_dx2, u_y)
        d2u_dy2 = jnp.einsum('ir,jr->ij', u_x, d2u_y_dy2)
        d2v_dx2 = jnp.einsum('ir,jr->ij', d2v_x_dx2, v_y)
        d2v_dy2 = jnp.einsum('ir,jr->ij', v_x, d2v_y_dy2)

        continuity = du_dx + dv_dy
        momentum_x = u * du_dx + v * du_dy + dp_dx - (1/self.Re) * (d2u_dx2 + d2u_dy2)
        momentum_y = u * dv_dx + v * dv_dy + dp_dy - (1/self.Re) * (d2v_dx2 + d2v_dy2)

        return jnp.stack([continuity, momentum_x, momentum_y], axis=-1)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train SPIKAN for cavity flow with cylinder')
    parser.add_argument('--nx', type=int, default=200, help='Number of x grid points (default: 200)')
    parser.add_argument('--ny', type=int, default=200, help='Number of y grid points (default: 200)')
    parser.add_argument('--epochs', type=int, default=100000, help='Number of training epochs (default: 100000)')
    parser.add_argument('--k', type=int, default=5, help='B-spline degree (default: 5)')
    parser.add_argument('--Re', type=float, default=100.0, help='Reynolds number (default: 100.0)')
    parser.add_argument('--r', type=int, default=10, help='Latent dimension (default: 10)')
    parser.add_argument('--layer_dims', type=str, default='[1,5,5,3]', 
                        help='Layer dimensions as list (default: [1,5,5,3])')
    parser.add_argument('--activation', type=str, default='silu', 
                        choices=['silu', 'tanh', 'relu', 'sine'],
                        help='Activation function (default: silu)')
    
    args = parser.parse_args()
    
    # Parse layer_dims from string to list
    layer_dims = ast.literal_eval(args.layer_dims)
    
    # Extract arguments
    nx, ny = args.nx, args.ny
    num_epochs = args.epochs
    k = args.k
    Re = args.Re
    r = args.r
    activation = args.activation
    init_lr = 1e-3
    
    # Create interior points using nx and ny
    x_interior, y_interior = create_interior_points(L_range, H_range, nx, ny)
    
    # Create interior mask
    interior_mask = create_interior_mask(x_interior, y_interior, cylinder_center, cylinder_radius)
    
    # Create boundary points using nx and ny
    (x_left, y_left), (x_top, y_top), (x_right, y_right), (x_bottom, y_bottom) = create_boundary_points(L_range, H_range, nx, ny)
    
    # Create cylinder boundary points
    n_cylinder = nx
    x_cylinder, y_cylinder = create_cylinder_boundary_points(cylinder_center, cylinder_radius, n_cylinder)

    # Create model
    model = CavityCylinder_SF_KAN_Separable(
        layer_dims=layer_dims,
        init_lr=init_lr,
        Re=Re,
        k=k,
        r=r,
        residual=activation,
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius
    )

    # Train model
    print(f"Starting training for {num_epochs} epochs with nx={nx}, ny={ny}, k={k}, Re={Re}, r={r}, activation={activation}")
    print(f"Network architecture: {layer_dims}")
    
    # Track training time
    import time
    start_time = time.time()
    loss_history = model.train(
        num_epochs,
        x_interior, y_interior, interior_mask,
        x_left, y_left,
        x_top, y_top,
        x_right, y_right,
        x_bottom, y_bottom,
        x_cylinder, y_cylinder
    )
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
            'r': r,
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
            'variables_x': model.variables_x,
            'variables_y': model.variables_y
        }
    }

    # Save data
    save_path = f'./data/cavity_cylinder_spikan_Re_{Re}_nx{nx}_ny{ny}_epochs{num_epochs}_r{r}_k{k}_{activation}.npy'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, output_data)
    print(f"\nData saved to: {save_path}")
    print(f"Final loss: {loss_history[-1]:.4e}")


