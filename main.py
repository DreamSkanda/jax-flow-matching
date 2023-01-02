import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import optax
import haiku as hk

from data import make_sampler
from net import make_vec_field_net, make_backflow, make_transformer
from flow import NeuralODE
from loss import make_loss
from energy import energy_fun, make_free_energy

from typing import NamedTuple
import itertools
import time
import matplotlib.pyplot as plt

class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

if __name__ == '__main__':
    rng = random.PRNGKey(42)

    import argparse
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-ds", default='datasets/',help="where to load the datasets for training")
    parser.add_argument("-fig", default='figures/',help="where to store the figures")

    group = parser.add_argument_group('learning parameters')
    group.add_argument('-epoch', type=int, default=15, help='')
    group.add_argument('-batchsize', type=int, default=4096, help='')
    group.add_argument('-samplesize', type=int, default=4096, help='')
    group.add_argument('-lr', type=float, default=1e-3, help='learning rate')

    group = parser.add_argument_group('datasets')
    group.add_argument('-datasize', type=int, default=102400, help='')
    group.add_argument('-name', type=str, default='mcmc', help='')

    group = parser.add_argument_group('network parameters')
    group.add_argument('-channel', type=int, default=512, help='The channels in a middle layer')
    group.add_argument('-numlayers', type=int, default=4, help='The number of layers')
    group.add_argument('-nheads', type=int, default=8, help='')
    group.add_argument('-keysize', type=int, default=16, help='')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-backflow', action='store_true', help='Use backflow')
    group.add_argument('-transformer', action='store_true', help='Use transformer')
    group.add_argument('-mlp', action='store_true', help='mlp')
    group.add_argument('-emlp', action='store_true', help='emlp')

    group = parser.add_argument_group('physics parameters')
    group.add_argument('-n', type=int, default=6, help='The number of particles')
    group.add_argument('-dim', type=int, default=2, help='The dimensions of the system')
    group.add_argument('-beta', type=float, default=10.0, help='')

    args = parser.parse_args()

####################################################################################

    print("\n========== Load dataset ==========")
    
    import os
    if not os.path.exists(args.ds):
        os.makedirs(args.ds)

    file = args.ds + args.name + 'datasize%i_beta%i_n%i_dim%i.npz' % (args.datasize, args.beta, args.n, args.dim)

    if os.path.isfile(file):
        data = jnp.load(file)
        X0, X1 = data['X0'], data['X1']

        print("File %s has been loaded." % file)
    else:
        print("File %s does not exist." % file)

        print("\n========== Generate dataset using MCMC ==========")
        
        sampler = make_sampler(args.datasize, args.name)
        data_rng, rng = random.split(rng)

        start = time.time()
        X0, X1 = sampler(data_rng, args.beta, args.n, args.dim)
        end = time.time()
        running_time = end - start
        
        print("sampling time: %.5f sec" % running_time)

        jnp.savez(file, X0=X0, X1=X1)

        print("File %s has been generated." % file)

####################################################################################

    '''building networks'''
    init_rng, rng = random.split(rng)
    if args.backflow:
        print('\n========== Construct backflow network ==========')
        params, vec_field_net = make_backflow(init_rng, args.n, args.dim, [args.channel]*args.numlayers)
        net_name = 'backflow'
    elif args.transformer:
        print("\n========== Construct transformer network ==========")
        params, vec_field_net = make_transformer(init_rng, args.n, args.dim, args.nheads, args.numlayers, args.keysize)
        net_name = 'transformer'
    elif args.mlp:
        print("\n========== Construct mlp network ==========")
        params, vec_field_net = make_vec_field_net(init_rng, args.n, args.dim, ch=args.channel, num_layers=args.numlayers, symmetry=False)
        net_name = 'mlp'
    elif args.emlp:
        print("\n========== Construct emlp network ==========")
        params, vec_field_net = make_vec_field_net(init_rng, args.n, args.dim, ch=args.channel, num_layers=args.numlayers, symmetry=True)
        net_name = 'emlp'
    else:
        raise ValueError("what model ?")

    '''initializing the sampler and logp calculator'''
    forward, reverse, batched_sample_fun, logp_fun = NeuralODE(vec_field_net, args.n*args.dim)
    free_energy = make_free_energy(energy_fun, batched_sample_fun, logp_fun, args.n, args.dim, args.beta)

    '''initializing the loss function'''
    loss = make_loss(vec_field_net)
    value_and_grad = jax.value_and_grad(loss, argnums=0, has_aux=True)

####################################################################################

    '''training with samples'''
    def training(rng, num_epochs, init_params, X0, X1, learning_rate, type_optim = optax.adam):
        
        @jax.jit
        def step(rng, i, state, x0, x1):
            rng, sample_rng = random.split(rng)
            x0 = random.normal(sample_rng, (x1.shape[0], x1.shape[1]))

            t = random.uniform(rng, (args.batchsize,))

            value, grad = value_and_grad(state.params, x0, x1, t)

            updates, opt_state = optimizer.update(grad, state.opt_state)
            params = optax.apply_updates(state.params, updates)

            return TrainingState(params, opt_state), value
        
        optimizer = type_optim(learning_rate)
        init_opt_state = optimizer.init(init_params)

        state = TrainingState(init_params, init_opt_state)

        itercount = itertools.count()
        loss_history = []
        for epoch in range(num_epochs):
            permute_rng, rng = random.split(rng)
            X0 = random.permutation(permute_rng, X0)
            X1 = random.permutation(permute_rng, X1)
            for batch_index in range(0, len(X1), args.batchsize):
                step_rng, rng = random.split(rng)
                state, (d_mean, d_err) = step(step_rng, next(itercount), state, X0[batch_index:batch_index+args.batchsize], X1[batch_index:batch_index+args.batchsize])
                print (epoch, d_mean)
                loss_history.append([d_mean, d_err])
        
        return state.params, loss_history

    print("\n========== Training ==========")

    start = time.time()
    trained_params, loss_history = training(rng, args.epoch, params, X0, X1, args.lr)
    end = time.time()
    running_time = end - start
    print('training time: %.5f sec' %running_time)

####################################################################################

    print("\n========== Calculating free energy ==========")

    '''drawing samples'''
    #start = time.time()
    #fe_rng, rng = random.split(rng)
    #fe, fe_err, _, f, f_err = free_energy(fe_rng, params, args.samplesize)
    #end = time.time()
    #running_time = end - start
    #print('free energy using untrained model: %f ± %f' %(fe, fe_err))
    #print('variational free energy using untrained model: %f ± %f' %(f, f_err))
    #print('importance sampling time: %.5f sec' %running_time)

    start = time.time()
    fe_rng, rng = random.split(rng)
    fe, fe_err, X_syn, f, f_err = free_energy(fe_rng, trained_params, args.samplesize)
    end = time.time()
    running_time = end - start
    print('free energy using trained model: %f ± %f' %(fe, fe_err))
    print('variational free energy using trained model: %f ± %f' %(f, f_err))
    print('importance sampling time: %.5f sec' %running_time)

    print("training loops: %i" % len(loss_history))

    '''plotting'''
    plot_range = [(-2, 2), (-2, 2)]
    n_bins = 100

    X_syn_show = X_syn.reshape(-1, args.dim)

    fig = plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist2d(X_syn_show[:, 0], X_syn_show[:, 1], bins=n_bins, range=plot_range, density=True, cmap="inferno")
    plt.subplot(1, 2, 2)
    y = jnp.reshape(jnp.array(loss_history), (-1, 2))
    plt.errorbar(jnp.arange(y.shape[0]), y[:, 0], yerr=y[:, 1], marker='o', capsize=8)

    if not os.path.exists(args.fig):
        os.makedirs(args.fig)

    if args.tranformer:
        plt.savefig('haiku_%s_batchsize%i_epoch%i_samplesize%i_head%i_layer%i_key%i_beta%i_n%i_dim%i_lr%f.png' \
                    % (net_name, args.batchsize, args.epoch, args.samplesize, args.nheads, args.numlayers, args.keysize, args.beta, args.n, args.dim, args.lr))
    else:
        plt.savefig('haiku_%s_batchsize%i_epoch%i_samplesize%i_ch%i_layer%i_beta%i_n%i_dim%i_lr%f.png' \
                    % (net_name, args.batchsize, args.epoch, args.samplesize, args.channel, args.numlayers, args.beta, args.n, args.dim, args.lr))
