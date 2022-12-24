import jax
import jax.numpy as jnp
from jax import random
from jax.config import config
from jax.example_libraries import optimizers

from data import make_sampler
from net import make_vec_field_net
from flow import NeuralODE
from loss import make_loss
from energy import energy_fun, make_free_energy

import itertools
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    config.update("jax_enable_x64", True)
    rng = random.PRNGKey(42)

    import argparse
    parser = argparse.ArgumentParser(description='')

    group = parser.add_argument_group('learning parameters')
    group.add_argument('-epoch', type=int, default=15, help='')
    group.add_argument('-batchsize', type=int, default=4096, help='')
    group.add_argument('-samplesize', type=int, default=4096, help='')
    group.add_argument('-step', type=float, default=1e-2, help='')

    group = parser.add_argument_group('datasets')
    group.add_argument('-datasize', type=int, default=102400, help='')
    group.add_argument('-name', type=str, default='mcmc', help='')

    group = parser.add_argument_group('physics parameters')
    group.add_argument('-n', type=int, default=6, help='The number of particles')
    group.add_argument('-dim', type=int, default=2, help='The dimensions of the system')
    group.add_argument('-beta', type=float, default=10.0, help='')

    args = parser.parse_args()

    '''generating datasets'''
    sampler = make_sampler(args.datasize, args.name)

    start = time.time()
    data_rng, rng = random.split(rng)
    X0, X1 = sampler(data_rng, args.beta, args.n, args.dim)
    end = time.time()
    running_time = end - start
    print('training set sampling time: %.5f sec' %running_time)

    '''building networks'''
    init_rng, rng = random.split(random.PRNGKey(42))
    params, vec_field_net = make_vec_field_net(init_rng, args.n*args.dim)

    '''initializing the sampler and logp calculator'''
    forward, reverse, batched_sample_fun, logp_fun = NeuralODE(vec_field_net, args.n*args.dim)
    free_energy = make_free_energy(energy_fun, batched_sample_fun, logp_fun, args.n, args.dim, args.beta)

    '''initializing the loss function'''
    loss = make_loss(vec_field_net)
    value_and_grad = jax.value_and_grad(loss, argnums=0, has_aux=True)

    '''training with samples'''
    def training(rng, num_epochs, params, X0, X1, step_size, optimizer=optimizers.adam):
        
        def step(rng, i, opt_state, x0, x1):
            params = get_params(opt_state)

            x0 = x0.reshape(-1, args.n*args.dim)
            x1 = x1.reshape(-1, args.n*args.dim)
            t = random.uniform(rng, (args.batchsize,))

            value, grad = value_and_grad(params, x0, x1, t)
            return opt_update(i, grad, opt_state), value
        
        opt_init, opt_update, get_params = optimizer(step_size)
        opt_state = opt_init(params)
        itercount = itertools.count()
        loss_history = []
        for epoch in range(num_epochs):
            permute_rng, step_rng, rng = random.split(rng, 3)
            X0 = random.permutation(permute_rng, X0)
            X1 = random.permutation(permute_rng, X1)
            for batch_index in range(0, len(X1), args.batchsize):
                opt_state, (d_mean, d_err) = step(step_rng, next(itercount), opt_state, X0[batch_index:batch_index+args.batchsize], X1[batch_index:batch_index+args.batchsize])
                loss_history.append([d_mean, d_err])
        
        return get_params(opt_state), loss_history

    start = time.time()
    trained_params, loss_history = training(rng, args.epoch, params, X0, X1, args.step)
    end = time.time()
    running_time = end - start
    print('training time: %.5f sec' %running_time)

    '''drawing samples'''
    start = time.time()
    fe_rng, rng = random.split(rng)
    fe, fe_err, _ = free_energy(fe_rng, params, args.samplesize)
    end = time.time()
    running_time = end - start
    print('free energy using untrained model: %f ± %f' %(fe, fe_err))
    print('importance sampling time: %.5f sec' %running_time)

    start = time.time()
    fe_rng, rng = random.split(rng)
    fe, fe_err, X_syn = free_energy(fe_rng, trained_params, args.samplesize)
    end = time.time()
    running_time = end - start
    print('free energy using trained model: %f ± %f' %(fe, fe_err))
    print('importance sampling time: %.5f sec' %running_time)

    print('training loops: %i' %loss_history.shape[0])

    '''plotting'''
    plot_range = [(-2, 2), (-2, 2)]
    n_bins = 100

    X_syn_show = X_syn.reshape(-1, args.dim)

    fig = plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist2d(X_syn_show[:, 0], X_syn_show[:, 1], bins=n_bins, range=plot_range)

    plt.subplot(1, 2, 2)
    y = jnp.reshape(jnp.array(loss_history), (-1, 2))
    plt.errorbar(jnp.arange(y.shape[0]), y[:, 0], yerr=y[:, 1], marker='o', capsize=8)

    plt.savefig('%s_batchsize%i_epoch%i_samplesize%i_beta%i_n%i_spatial_dim%i_step%f.png' \
                % (args.name, args.batchsize, args.epoch, args.samplesize, args.beta, args.n, args.dim, args.step))