import jax
import jax.numpy as jnp
import numpy as np
from jax import random, jit, vmap
from jax.example_libraries.stax import serial, Dense, Relu
from jax.nn.initializers import zeros
from jax.example_libraries import optimizers

from flow import make_cond_flow, NeuralODE
from loss import make_loss

from sklearn import datasets, preprocessing
import itertools
from functools import partial
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # loading datasets
    n_samples = 100000
    plot_range = [(-2, 2), (-2, 2)]
    n_bins = 100

    scaler = preprocessing.StandardScaler()
    X, _ = datasets.make_moons(n_samples=n_samples, noise=.05)
    X = scaler.fit_transform(X)

    # hyperparameters
    dim = X.shape[-1]
    sigma_min = 0.01
    num_epochs, batch_size = 30, 10000

    # building networks
    def make_vec_field_net(rng):
        net_init, net_apply = serial(Dense(512), Relu, Dense(512), Relu, Dense(dim, W_init=zeros, b_init=zeros))
        in_shape = (-1, dim+1)
        _, net_params = net_init(rng, in_shape)

        def net_with_t(params, x, t):
            return net_apply(params, jnp.concatenate((x,t.reshape(1))))
        
        return net_params, net_with_t

    init_rng, rng = random.split(random.PRNGKey(42))
    params, vec_field_net = make_vec_field_net(init_rng)
    batched_cond_flow, cond_vec_field = make_cond_flow(sigma_min)

    # initializing the sampler and logp calculator
    batched_sampler, batched_logp = NeuralODE(vec_field_net, batch_size, dim)

    # initializing the loss function
    loss = make_loss(vec_field_net, cond_vec_field)
    value_and_grad = jax.value_and_grad(loss, argnums=0, has_aux=True)

    # initializing the optimizer
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)
    opt_state = opt_init(params)

    # training step
    def step(rng, i, opt_state, inputs):
        params = get_params(opt_state)

        n_rng, u_rng = random.split(rng)
        t = random.uniform(u_rng, (batch_size,))
        x = batched_cond_flow(random.normal(n_rng, (batch_size, dim)), inputs, t)

        value, grad = value_and_grad(params, x, inputs, t)
        return opt_update(i, grad, opt_state), value

    # training
    itercount = itertools.count()
    loss_history = []
    for epoch in range(num_epochs):
        permute_rng, step_rng, rng = random.split(rng, 3)
        X = random.permutation(permute_rng, X)

        for batch_index in range(0, len(X), batch_size):
            opt_state, (d_mean, d_err) = step(step_rng, next(itercount), opt_state, X[batch_index:batch_index+batch_size])
            loss_history.append([d_mean, d_err])
            print(d_mean, d_err)
    
    params = get_params(opt_state)
    sample_rng, rng = random.split(rng)
    
    X_syn = batched_sampler(sample_rng, params)

    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist2d(X_syn[:, 0], X_syn[:, 1], bins=n_bins, range=plot_range)

    plt.subplot(1, 2, 2)
    y = np.reshape(np.array(loss_history), (-1, 2))
    plt.errorbar(np.arange(y.shape[0]), y[:, 0], yerr=y[:, 1], marker='o', capsize=8)

    plt.savefig('batchsize%i_epoch%i.png' % (batch_size, num_epochs))