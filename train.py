import jax
import optax
import haiku as hk

import checkpoint
import os
from typing import NamedTuple
import itertools

class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

def train(rng, value_and_grad, num_epochs, batchsize, params, X0, X1, lr, path, coupled=True):
    
    assert (len(X1)%batchsize==0)

    @jax.jit
    def step(rng, i, state, x0, x1):
        t_rng, sample_rng = jax.random.split(rng)
        if not coupled:
            x0 = jax.random.normal(sample_rng, x1.shape)

        t = jax.random.uniform(t_rng, (batchsize,))

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
        X0 = jax.random.permutation(permute_rng, X0)
        X1 = jax.random.permutation(permute_rng, X1)

        total_loss = 0.0
        counter = 0 
        for batch_index in range(0, len(X1), batchsize):
            x0, x1 = X0[batch_index:batch_index+batchsize], X1[batch_index:batch_index+batchsize]

            step_rng, rng = jax.random.split(rng)
            state, (d_mean, d_err) = step(step_rng, next(itercount), state, x0, x1)
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