import jax
import jax.numpy as jnp
import optax
import haiku as hk

import checkpoint
import os
from typing import NamedTuple
import itertools

class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

def train(rng, value_and_grad, hyperparams, params, data, lr, path):

    (num_epochs, num_iterations, batchsize) = hyperparams
    assert (len(data)//batchsize==num_iterations and len(data)%batchsize==0)

    @jax.jit
    def step(rng, i, state, x1):
        sample_rng, rng = jax.random.split(rng)
        x0 = jax.random.normal(sample_rng, x1.shape)
        t = jax.random.uniform(rng, (batchsize,))

        value, grad = value_and_grad(state.params, x0, x1, t)

        updates, opt_state = optimizer.update(grad, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(params, opt_state), value
    
    optimizer = optax.adam(lr)
    init_opt_state = optimizer.init(params)
    state = TrainingState(params, init_opt_state)

    log_filename = os.path.join(path, "loss.txt")
    f = open(log_filename, "w", buffering=1, newline="\n")
    itercount = itertools.count()
    for epoch in range(1, num_epochs+1):
        permute_rng, rng = jax.random.split(rng)
        data = jax.random.permutation(permute_rng, data)

        total_loss = 0.0
        counter = 0 
        for batch_index in range(0, num_iterations, batchsize):
            x1 = data[batch_index:batch_index+batchsize]

            step_rng, rng = jax.random.split(rng)
            state, d_mean = step(step_rng, next(itercount), state, x1)
            total_loss += d_mean
            counter += 1

        f.write( ("%6d" + "  %.6f" + "\n") % (epoch, total_loss/counter) )

        if epoch % 100 == 0:
            ckpt = {"params": state.params,
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return state.params


def train_and_evaluate(rng, loss, value_and_grad, hyperparams, params, training_data, validation_data, lr, path):
    
    (num_epochs, num_iterations, batchsize) = hyperparams

    training_X0, training_X1 = training_data
    validation_X0, validation_X1 = validation_data

    validation_batchsize = len(validation_X1)//num_iterations

    assert (len(training_X1)//batchsize==num_iterations and len(training_X1)%batchsize==0)
    assert (len(validation_X1)//validation_batchsize==num_iterations and len(validation_X1)%validation_batchsize==0)

    @jax.jit
    def train_step(rng, i, state, x0, x1):
        t = jax.random.uniform(rng, (x0.shape[0],))

        value, grad = value_and_grad(state.params, x0, x1, t)

        updates, opt_state = optimizer.update(grad, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(params, opt_state), value

    @jax.jit
    def test_step(rng, i, state, x0, x1):
        t = jax.random.uniform(rng, (x0.shape[0],))
        value = loss(state.params, x0, x1, t)
        return value
    
    optimizer = optax.adam(lr)
    init_opt_state = optimizer.init(params)
    state = TrainingState(params, init_opt_state)

    log_filename = os.path.join(path, "loss.txt")
    f = open(log_filename, "w", buffering=1, newline="\n")
    itercount = itertools.count()
    for epoch in range(1, num_epochs+1):
        permute_rng, rng = jax.random.split(rng)
        training_X0 = jax.random.permutation(permute_rng, training_X0)
        training_X1 = jax.random.permutation(permute_rng, training_X1)

        train_loss = 0.0
        validation_loss = 0.0
        counter = 0
        # train
        for batch_index in range(0, num_iterations, batchsize):
            x0 = training_X0[batch_index:batch_index+batchsize]
            x1 = training_X1[batch_index:batch_index+batchsize]

            step_rng, rng = jax.random.split(rng)
            state, d_mean = train_step(step_rng, next(itercount), state, x0, x1)
            train_loss += d_mean
            counter += 1
        
        # test
        for batch_index in range(0, num_iterations, validation_batchsize):
            x0 = validation_X0[batch_index:batch_index+validation_batchsize]
            x1 = validation_X1[batch_index:batch_index+validation_batchsize]

            step_rng, rng = jax.random.split(rng)
            d_mean = test_step(step_rng, next(itercount), state, x0, x1)
            validation_loss += d_mean

        f.write( ("%6d" + "  %.6f" + "  %.6f" + "\n") % (epoch, train_loss/counter, validation_loss/counter) )

        if epoch % 100 == 0:
            ckpt = {"params": state.params,
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return state.params