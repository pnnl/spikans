#Copyright (c) 2024 Spyros Rigas, Michalis Papachristou
#Adapted from: https://github.com/srigas/jaxKAN
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import optax
from jax import grad, vmap, jit
from functools import partial
from tqdm import trange
from flax import linen as nn
from flax.linen import initializers
import time 


from KAN import KAN

class SF_KAN(ABC):
    def __init__(self, layer_dims, init_lr,  k=3):
        self.modelLF = KAN(layer_dims=layer_dims, k=k, const_spl=False, const_res=False, add_bias=True, grid_e=0.02, j='0')
        # self.lr_scales = dict(zip(boundaries, scales))
        # self.grid_upds = dict(zip(boundaries, grid_vals))
        self.layer_dims = layer_dims

        # schedule = optax.piecewise_constant_schedule(
        #     init_value=init_lr,
        #     boundaries_and_scales=self.lr_scales
        # )
        
        key = jax.random.PRNGKey(10)
        self.variables = self.modelLF.init(key, jnp.ones([1, layer_dims[0]]))
        variable_params = self.variables['params']
            
        self.optimizer = optax.adam(learning_rate=init_lr, nesterov=True)
        self.opt_state = self.optimizer.init(variable_params)

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

    @partial(jit, static_argnums=(0,))
    def forward_pass(self, variables, x):
        preds, spl_regs = self.modelLF.apply(variables, x)
                
        return preds, spl_regs

    @abstractmethod
    def loss_fn(self, params, state, *args):
        pass

    @partial(jit, static_argnums=(0,))
    def loss(self, params, state, *args):
        loss_value = self.loss_fn(params, state, *args)        
        return loss_value

    @partial(jit, static_argnums=(0,))
    def train_step(self, params, state, opt_state, xy_domain, xy_boundary):
        loss_value, grads = jax.value_and_grad(self.loss)(params, state, xy_domain, xy_boundary)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    @partial(jit, static_argnums=(0,))
    def train_epoch(self, params, state, opt_state, xy_domain, xy_boundary, xy_ic):
        params, opt_state, loss_value = self.train_step(params, state, opt_state, xy_domain, xy_boundary)
        return params, state, opt_state, loss_value

    def train(self, num_epochs, xy_domain, xy_boundary):
        params, state = self.variables['params'], self.variables['state']
        opt_state = self.optimizer.init(params)
        loss_history = []

        pbar = trange(num_epochs, smoothing=0.)
        for epoch in pbar:
            # if epoch in self.grid_upds.keys():
            #     print(f"Epoch {epoch+1}: Performing grid update")
            #     G_new = self.grid_upds[epoch]
            #     updated_variables = self.modelLF.apply({'params': params, 'state': state}, xy_domain, G_new, method=self.modelLF.update_grids)
                
            #     params, state = updated_variables['params'], updated_variables['state']
            #     opt_state = self.smooth_state_transition(opt_state, params)
            params, opt_state, loss_value = self.train_step(params, state, opt_state, xy_domain, xy_boundary)
            loss_history.append(loss_value)

            if epoch % 10 == 0:
                pbar.set_postfix({'Loss': "{0:.4e}".format(loss_value)})
        
        return {'params': params, 'state': state}, loss_history

