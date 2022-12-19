import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config
from jax import lax, random, jit, vmap
from jax.example_libraries.stax import serial, Dense, Relu
from jax.nn.initializers import zeros
from jax.example_libraries import optimizers

from data import make_sampler
from net import make_vec_field_net
from flow import make_cond_flow, NeuralODE
from loss import make_loss
from energy import energy_fun, make_free_energy

import itertools
import time
from functools import partial
import matplotlib.pyplot as plt

if __name__ == '__main__':
    config.update("jax_enable_x64", True)
    rng = random.PRNGKey(42)
    
    '''setting hyperparameters'''
    beta = 10
    n = 6
    dim = 2

    sigma_min = 0.01
    num_epochs, batch_size = 20, 1024
    sample_size = 4096

    optimizer = optimizers.adam
    step_size = 1e-2

    '''datasets'''
    data_size = 102400
    data_name = 'mcmc'
    data_rng, rng = random.split(rng)

    sampler = make_sampler(data_size, data_name)
    X0, X1 = sampler(data_rng, beta, n, dim)

    '''building networks'''
    init_rng, rng = random.split(random.PRNGKey(42))
    params, vec_field_net = make_vec_field_net(init_rng, n*dim)
    batched_cond_flow, cond_vec_field = make_cond_flow(sigma_min)

    '''initializing the sampler and logp calculator'''
    forward, reverse, batched_sample_fun, logp_fun = NeuralODE(vec_field_net, n*dim)
    free_energy = make_free_energy(energy_fun, batched_sample_fun, logp_fun, n, dim, beta)

    '''initializing the loss function'''
    loss = make_loss(vec_field_net, cond_vec_field)
    value_and_grad = jax.value_and_grad(loss, argnums=0, has_aux=True)

    '''training with samples'''
    def training(rng, num_epochs, optimizer, step_size, params, X0, X1):
        
        def step(rng, i, opt_state, x0, x1):
            params = get_params(opt_state)

            x0 = x0.reshape(-1, n*dim)
            x1 = x1.reshape(-1, n*dim)

            t = random.uniform(rng, (batch_size,))
            x = batched_cond_flow(x0, x1, t)

            value, grad = value_and_grad(params, x, x1, t)
            return opt_update(i, grad, opt_state), value
        
        opt_init, opt_update, get_params = optimizer(step_size)
        opt_state = opt_init(params)
        itercount = itertools.count()
        loss_history = []
        for epoch in range(num_epochs):
            permute_rng, step_rng, rng = random.split(rng, 3)
            X0 = random.permutation(permute_rng, X0)
            X1 = random.permutation(permute_rng, X1)
            for batch_index in range(0, len(X1), batch_size):
                opt_state, (d_mean, d_err) = step(step_rng, next(itercount), opt_state, X0[batch_index:batch_index+batch_size], X1[batch_index:batch_index+batch_size])
                loss_history.append([d_mean, d_err])
        
        return get_params(opt_state), loss_history
    
    trained_params, loss_history = training(rng, num_epochs, optimizer, step_size, params, X0, X1)

    '''drawing samples'''
    start = time.time()
    fe_rng, rng = random.split(rng)
    fe, fe_err, _ = free_energy(fe_rng, params, sample_size)
    end = time.time()
    running_time = end - start
    print('free energy using untrained model: %f ± %f' %(fe, fe_err))
    print('time cost: %.5f sec' %running_time)

    start = time.time()
    fe_rng, rng = random.split(rng)
    fe, fe_err, X_syn = free_energy(fe_rng, trained_params, sample_size)
    end = time.time()
    running_time = end - start
    print('free energy using trained model: %f ± %f' %(fe, fe_err))
    print('time cost: %.5f sec' %running_time)

    '''plotting'''
    plot_range = [(-2, 2), (-2, 2)]
    n_bins = 100

    X_syn_show = X_syn.reshape(-1, dim)

    fig = plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist2d(X_syn_show[:, 0], X_syn_show[:, 1], bins=n_bins, range=plot_range)

    plt.subplot(1, 2, 2)
    y = jnp.reshape(jnp.array(loss_history), (-1, 2))
    plt.errorbar(jnp.arange(y.shape[0]), y[:, 0], yerr=y[:, 1], marker='o', capsize=8, label='couple')
    plt.legend()

    plt.savefig('%s_batchsize%i_epoch%i_beta%i_n%i_dim%i_step%f.png' % (data_name, batch_size, num_epochs, beta, n, dim, step_size))